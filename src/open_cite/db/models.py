"""
SQLAlchemy ORM model definitions for all OpenCITE tables.

Uses SQLAlchemy 2.0 DeclarativeBase. ``sqlalchemy.JSON`` maps to TEXT on
SQLite, JSONB on PostgreSQL, and STRING on Databricks.

Databricks SQL compatibility notes:
- ``BigInteger`` + ``Identity()`` instead of ``Integer`` + ``autoincrement``
  (Databricks only supports BIGINT GENERATED ALWAYS AS IDENTITY)
- ``ForeignKey`` constraints carry explicit ``name=`` (required by dialect)
- Foreign keys are informational only on Databricks (not enforced)
"""

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.types import JSON
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import CreateIndex, DropIndex


# Databricks SQL has no native JSON type — render as STRING instead.
@compiles(JSON, "databricks")
def _compile_json_databricks(type_, compiler, **kw):
    return "STRING"


# Databricks Unity Catalog does not support CREATE INDEX / DROP INDEX.
@compiles(CreateIndex, "databricks")
def _skip_create_index_databricks(element, compiler, **kw):
    return "SELECT 1"


@compiles(DropIndex, "databricks")
def _skip_drop_index_databricks(element, compiler, **kw):
    return "SELECT 1"


# Databricks dialect lacks the _json_serializer/_json_deserializer attrs
# that SQLAlchemy's JSON type expects.  Patch them in so JSON columns work
# (uses default json.loads / json.dumps).
try:
    from databricks.sqlalchemy import DatabricksDialect
    if not hasattr(DatabricksDialect, "_json_deserializer"):
        DatabricksDialect._json_deserializer = None
    if not hasattr(DatabricksDialect, "_json_serializer"):
        DatabricksDialect._json_serializer = None
except ImportError:
    pass


class Base(DeclarativeBase):
    pass


# =========================================================================
# Table group 1 — Plugin Configs (OPENCITE_PERSIST_PLUGINS)
# =========================================================================


class PluginConfig(Base):
    __tablename__ = "plugin_configs"

    instance_id = Column(String, primary_key=True)
    plugin_type = Column(String, nullable=False, index=True)
    display_name = Column(String, nullable=False)
    config = Column(JSON, nullable=False)
    auto_start = Column(Boolean, default=False)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)


# =========================================================================
# Table group 2 — Discovered Assets (OPENCITE_PERSISTENCE_ENABLED)
# =========================================================================


class Tool(Base):
    __tablename__ = "tools"

    name = Column(String, primary_key=True)
    models = Column(JSON, nullable=False)
    trace_count = Column(Integer, default=0)
    metadata_ = Column("metadata", JSON)
    last_updated = Column(String, nullable=False)


class Model(Base):
    __tablename__ = "models"

    name = Column(String, primary_key=True)
    provider = Column(String)
    tools = Column(JSON, nullable=False)
    usage_count = Column(Integer, default=0)
    last_updated = Column(String, nullable=False)


class Agent(Base):
    __tablename__ = "agents"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, index=True)
    tools_used = Column(JSON)
    models_used = Column(JSON)
    first_seen = Column(String, nullable=False)
    last_seen = Column(String, nullable=False)
    metadata_ = Column("metadata", JSON)


class DownstreamSystem(Base):
    __tablename__ = "downstream_systems"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, index=True)
    type = Column(String, default="unknown")
    endpoint = Column(String)
    tools_connecting = Column(JSON)
    first_seen = Column(String, nullable=False)
    last_seen = Column(String, nullable=False)
    metadata_ = Column("metadata", JSON)


class Lineage(Base):
    __tablename__ = "lineage"

    source_id = Column(String, primary_key=True)
    target_id = Column(String, primary_key=True)
    relationship_type = Column(String, primary_key=True)
    source_type = Column(String, nullable=False)
    target_type = Column(String, nullable=False)
    weight = Column(Integer, default=1)
    first_seen = Column(String, nullable=False)
    last_seen = Column(String, nullable=False)


class McpServer(Base):
    __tablename__ = "mcp_servers"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    transport = Column(String)
    endpoint = Column(String)
    command = Column(String)
    args = Column(JSON)
    env = Column(JSON)
    source_file = Column(String)
    source_env_var = Column(String)
    metadata_ = Column("metadata", JSON)
    last_updated = Column(String, nullable=False)

    tools = relationship("McpTool", back_populates="server", cascade="all, delete-orphan")
    resources = relationship("McpResource", back_populates="server", cascade="all, delete-orphan")


class McpTool(Base):
    __tablename__ = "mcp_tools"

    id = Column(String, primary_key=True)
    server_id = Column(String, ForeignKey("mcp_servers.id", name="fk_mcp_tools_server"), nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(String)
    schema_ = Column("schema", JSON)
    usage = Column(JSON)
    metadata_ = Column("metadata", JSON)
    last_updated = Column(String, nullable=False)

    server = relationship("McpServer", back_populates="tools")


class McpResource(Base):
    __tablename__ = "mcp_resources"

    id = Column(String, primary_key=True)
    server_id = Column(String, ForeignKey("mcp_servers.id", name="fk_mcp_resources_server"), nullable=False, index=True)
    uri = Column(String, nullable=False)
    name = Column(String)
    type = Column(String)
    mime_type = Column(String)
    description = Column(String)
    usage = Column(JSON)
    metadata_ = Column("metadata", JSON)
    last_updated = Column(String, nullable=False)

    server = relationship("McpServer", back_populates="resources")


class DiscoveryStatus(Base):
    __tablename__ = "discovery_status"

    key = Column(String, primary_key=True)
    value = Column(JSON, nullable=False)


# =========================================================================
# Table group 3 — ID Mappings (OPENCITE_PERSIST_MAPPINGS)
# =========================================================================


class AssetIdMapping(Base):
    __tablename__ = "asset_id_mappings"

    id = Column(BigInteger, Identity(always=True), primary_key=True)
    plugin_name = Column(String, nullable=False, index=True)
    attributes = Column(JSON, nullable=False)
    identity = Column(JSON, nullable=False)
    match_type = Column(String, default="all")
    created_at = Column(String, nullable=False)
