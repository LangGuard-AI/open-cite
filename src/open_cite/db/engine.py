"""
SQLAlchemy engine singleton, session factory, and database lifecycle.

Supports SQLite (local dev), PostgreSQL, and Databricks SQL (via Unity Catalog).
"""

import logging
import os
from typing import Optional, Dict, Any

from sqlalchemy import create_engine, event, text as sqltext
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from .models import Base

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None

# Token forwarded from the Databricks App proxy (``x-forwarded-access-token``).
# When set, new DB connections prefer this over static tokens or SDK auth.
_forwarded_access_token: Optional[str] = None


def set_forwarded_token(token: str):
    """Store the latest ``x-forwarded-access-token`` for DB connections."""
    global _forwarded_access_token
    _forwarded_access_token = token
    logger.info("[db] Forwarded access token updated (length=%d)", len(token))

# Default catalog/schema for Databricks SQL persistence
# Uses the warehouse's default catalog (typically "main") with an "opencite" schema.
_DATABRICKS_CATALOG = "workspace"
_DATABRICKS_SCHEMA = "opencite"


def _build_databricks_engine_args() -> Optional[Dict[str, Any]]:
    """Build ``databricks.sql.connect()`` kwargs for Databricks SQL.

    Returns a dict of kwargs if running on Databricks with a SQL
    warehouse available, else ``None``.
    """
    host = os.getenv("DATABRICKS_HOST")
    logger.info("[db] _build_databricks_engine_args: DATABRICKS_HOST=%s", host)
    if not host:
        logger.info("[db] No DATABRICKS_HOST — skipping Databricks SQL")
        return None

    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    logger.info("[db] DATABRICKS_WAREHOUSE_ID=%s", warehouse_id)

    if not warehouse_id:
        logger.info("[db] No DATABRICKS_WAREHOUSE_ID, attempting SDK auto-discovery...")
        try:
            from databricks.sdk import WorkspaceClient
            logger.info("[db] Creating WorkspaceClient for warehouse discovery...")
            w = WorkspaceClient()
            logger.info("[db] WorkspaceClient created, listing warehouses...")
            all_warehouses = list(w.warehouses.list())
            logger.info("[db] Found %d warehouses", len(all_warehouses))
            for wh in all_warehouses:
                state_val = wh.state.value if wh.state else "unknown"
                logger.info("[db]   Warehouse: name=%s, id=%s, state=%s", wh.name, wh.id, state_val)
                if not warehouse_id and state_val == "RUNNING":
                    warehouse_id = wh.id
                    logger.info("[db] Selected running warehouse: %s (%s)", wh.name, wh.id)
            if not warehouse_id and all_warehouses:
                warehouse_id = all_warehouses[0].id
                logger.info("[db] No running warehouse, falling back to first: %s", warehouse_id)
        except Exception as exc:
            logger.warning("[db] SDK warehouse auto-discovery failed: %s", exc, exc_info=True)

    if not warehouse_id:
        logger.warning("[db] No SQL warehouse available — cannot use Databricks SQL")
        return None

    hostname = host.replace("https://", "").replace("http://", "").rstrip("/")
    http_path = f"/sql/1.0/warehouses/{warehouse_id}"
    catalog = os.getenv("OPENCITE_DATABRICKS_CATALOG", _DATABRICKS_CATALOG)
    schema = os.getenv("OPENCITE_DATABRICKS_SCHEMA", _DATABRICKS_SCHEMA)

    connect_args = {
        "server_hostname": hostname,
        "http_path": http_path,
        "catalog": catalog,
        "schema": schema,
    }

    logger.info("[db] Databricks SQL connect_args: server_hostname=%s, http_path=%s, catalog=%s, schema=%s",
                hostname, http_path, catalog, schema)
    return connect_args


def _build_database_url() -> str:
    """Resolve the database URL from environment variables.

    Priority:
    1. ``OPENCITE_DATABASE_URL`` env var (full SQLAlchemy URL)
    2. ``OPENCITE_DB_PATH`` env var → ``sqlite:///{path}``
    3. Default ``sqlite:///./opencite.db``

    Note: Databricks SQL is handled separately in ``get_engine()``.
    """
    url = os.getenv("OPENCITE_DATABASE_URL")
    if url:
        logger.info("[db] Using OPENCITE_DATABASE_URL: %s", url.split("@")[-1] if "@" in url else url)
        return url

    db_path = os.getenv("OPENCITE_DB_PATH", "./opencite.db")
    logger.info("[db] Using SQLite: OPENCITE_DB_PATH=%s", db_path)
    return f"sqlite:///{db_path}"


def get_engine():
    """Return the singleton SQLAlchemy engine, creating it on first call."""
    global _engine
    if _engine is not None:
        return _engine

    logger.info("[db] get_engine() called — creating engine...")

    # Try Databricks SQL first (auto-detected from environment)
    db_connect_args = _build_databricks_engine_args()
    if db_connect_args is not None:
        logger.info("[db] Databricks SQL detected, setting up creator-based engine...")
        try:
            import databricks.sql as dbsql
            logger.info("[db] databricks.sql imported successfully")
        except ImportError as exc:
            logger.error("[db] Failed to import databricks.sql: %s", exc)
            logger.info("[db] Falling back to non-Databricks engine")
            db_connect_args = None

    if db_connect_args is not None:
        import databricks.sql as dbsql

        # Resolve auth: static token or SDK OAuth (refreshed per connection)
        static_token = os.getenv("DATABRICKS_TOKEN")
        sdk_config = None
        if static_token:
            logger.info("[db] Auth: using static DATABRICKS_TOKEN")
        else:
            logger.info("[db] Auth: no DATABRICKS_TOKEN, trying SDK OAuth...")
            try:
                from databricks.sdk import WorkspaceClient
                sdk_config = WorkspaceClient().config
                logger.info("[db] Auth: SDK WorkspaceClient created, auth_type=%s", sdk_config.auth_type)
                # Test that authenticate() works
                # SDK API varies: some versions take a headers dict, others return headers
                try:
                    test_headers = {}
                    sdk_config.authenticate(test_headers)
                except TypeError:
                    test_headers = sdk_config.authenticate() or {}
                test_auth = test_headers.get("Authorization", "")
                if test_auth and test_auth.startswith("Bearer "):
                    logger.info("[db] Auth: SDK authenticate() returned Bearer token (length=%d)", len(test_auth))
                else:
                    logger.warning("[db] Auth: SDK authenticate() did NOT return a valid Bearer token: %s", repr(test_auth[:50] if test_auth else test_auth))
                    logger.info("[db] Auth: SDK config details: host=%s, auth_type=%s, client_id=%s",
                                getattr(sdk_config, 'host', None),
                                getattr(sdk_config, 'auth_type', None),
                                os.getenv('DATABRICKS_CLIENT_ID', 'not_set'))
            except Exception as exc:
                logger.warning("[db] Auth: SDK OAuth setup failed: %s", exc, exc_info=True)

        if not static_token and not sdk_config:
            logger.error("[db] Auth: no auth method available — cannot connect to Databricks SQL")
            logger.info("[db] Falling back to non-Databricks engine")
        else:
            def _databricks_creator():
                """Create a fresh databricks-sql connection with current auth."""
                kwargs = dict(db_connect_args)
                if _forwarded_access_token:
                    kwargs["access_token"] = _forwarded_access_token
                elif static_token:
                    kwargs["access_token"] = static_token
                elif sdk_config:
                    try:
                        headers = {}
                        sdk_config.authenticate(headers)
                    except TypeError:
                        headers = sdk_config.authenticate() or {}
                    auth = headers.get("Authorization", "")
                    if auth.startswith("Bearer "):
                        kwargs["access_token"] = auth[7:]
                    else:
                        logger.error("[db] creator: SDK authenticate() did not return Bearer token: %s", repr(auth[:50] if auth else auth))

                # Fail-fast if no access token — prevents OAuth callback flow
                if not kwargs.get("access_token"):
                    raise RuntimeError("No access_token available for Databricks SQL connection")

                logger.info("[db] creator: calling databricks.sql.connect(server_hostname=%s, http_path=%s, catalog=%s, schema=%s, access_token=%s)",
                            kwargs.get("server_hostname"), kwargs.get("http_path"),
                            kwargs.get("catalog"), kwargs.get("schema"),
                            "present" if kwargs.get("access_token") else "MISSING")
                try:
                    conn = dbsql.connect(**kwargs)
                    logger.info("[db] creator: connection established successfully")
                    return conn
                except Exception as exc:
                    logger.error("[db] creator: databricks.sql.connect() failed: %s", exc, exc_info=True)
                    raise

            try:
                _engine = create_engine(
                    "databricks://",
                    creator=_databricks_creator,
                    pool_pre_ping=True,
                )
                logger.info("[db] SQLAlchemy engine created (Databricks SQL) — testing connection...")
                # Don't test here; let init_db() do it via create_all()
                return _engine
            except Exception as exc:
                logger.error("[db] Failed to create Databricks SQLAlchemy engine: %s", exc, exc_info=True)
                logger.info("[db] Falling back to non-Databricks engine")

    # Fall back to explicit URL or SQLite
    url = _build_database_url()
    is_sqlite = url.startswith("sqlite")

    if is_sqlite:
        _engine = create_engine(
            url,
            poolclass=NullPool,
        )

        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

        logger.info("[db] SQLAlchemy engine created (SQLite, NullPool+WAL): %s", url)
    else:
        _engine = create_engine(
            url,
            poolclass=QueuePool,
            pool_pre_ping=True,
            pool_size=5,
        )
        logger.info("[db] SQLAlchemy engine created (PostgreSQL): %s", url.split("@")[-1] if "@" in url else url)

    return _engine


def get_session():
    """Return a thread-scoped session bound to the singleton engine."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        factory = sessionmaker(bind=engine)
        _session_factory = scoped_session(factory)
    return _session_factory()


def _drop_table(engine, table_name: str):
    """Drop a table, using fully-qualified name on Databricks."""
    if engine.dialect.name == "databricks":
        catalog = os.getenv("OPENCITE_DATABRICKS_CATALOG", _DATABRICKS_CATALOG)
        schema = os.getenv("OPENCITE_DATABRICKS_SCHEMA", _DATABRICKS_SCHEMA)
        fqn = f"`{catalog}`.`{schema}`.`{table_name}`"
    else:
        fqn = table_name
    with engine.connect() as conn:
        conn.execute(sqltext(f"DROP TABLE IF EXISTS {fqn}"))
        conn.commit()
    logger.info("[db] Dropped table %s", fqn)


def _run_migrations(engine):
    """One-time schema migrations for tables whose columns have changed."""
    from sqlalchemy import inspect as sa_inspect
    try:
        insp = sa_inspect(engine)

        # Lineage: synthetic `id` PK → composite PK (source_id, target_id, relationship_type)
        if insp.has_table("lineage"):
            columns = {c["name"] for c in insp.get_columns("lineage")}
            if "id" in columns:
                logger.info("[db] Migrating lineage table: dropping old schema with 'id' column")
                _drop_table(engine, "lineage")

        # Agents: removed `confidence` column
        if insp.has_table("agents"):
            columns = {c["name"] for c in insp.get_columns("agents")}
            if "confidence" in columns:
                logger.info("[db] Migrating agents table: dropping old schema with 'confidence' column")
                _drop_table(engine, "agents")

    except Exception as exc:
        logger.warning("[db] Migration check failed (non-fatal): %s", exc)


def init_db():
    """Create all tables that don't already exist."""
    logger.info("[db] init_db() called — creating tables...")
    try:
        engine = get_engine()
        logger.info("[db] Engine dialect: %s", engine.dialect.name)

        # On Databricks, try to ensure the schema exists before creating tables.
        # This is best-effort — if permissions block it, the schema must already exist.
        if engine.dialect.name == "databricks":
            catalog = os.getenv("OPENCITE_DATABRICKS_CATALOG", _DATABRICKS_CATALOG)
            schema = os.getenv("OPENCITE_DATABRICKS_SCHEMA", _DATABRICKS_SCHEMA)
            logger.info("[db] Ensuring Databricks schema %s.%s exists...", catalog, schema)
            try:
                with engine.connect() as conn:
                    conn.execute(sqltext(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`"))
                    conn.commit()
                logger.info("[db] Databricks schema ready")
            except Exception as exc:
                logger.warning("[db] Could not create schema %s.%s (may already exist): %s",
                               catalog, schema, exc)

        # One-time schema migrations for changed tables
        _run_migrations(engine)

        if engine.dialect.name == "databricks":
            # Databricks checkfirst/has_table can be unreliable — create each
            # table individually and tolerate "already exists" errors.
            for table in Base.metadata.sorted_tables:
                try:
                    table.create(engine)
                    logger.info("[db] Created table: %s", table.name)
                except Exception as table_exc:
                    if "ALREADY_EXISTS" in str(table_exc):
                        logger.info("[db] Table %s already exists, skipping", table.name)
                    else:
                        raise
        else:
            Base.metadata.create_all(engine)
        logger.info("[db] Database tables created/verified successfully")
    except Exception as exc:
        logger.error("[db] init_db() failed: %s", exc, exc_info=True)
        raise


def close_db():
    """Dispose the engine and remove the scoped session."""
    global _engine, _session_factory
    if _session_factory is not None:
        _session_factory.remove()
        _session_factory = None
    if _engine is not None:
        _engine.dispose()
        _engine = None
        logger.info("[db] Database engine disposed")
