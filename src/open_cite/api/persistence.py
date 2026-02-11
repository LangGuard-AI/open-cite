"""
OpenCITE SQLite Persistence Layer.

Provides durable storage for discovered assets, traces, and mappings.
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 3


class PersistenceManager:
    """
    SQLite-based persistence for OpenCITE data.

    Thread-safe storage for traces, tools, models, MCP entities, and mappings.
    """

    def __init__(self, db_path: str):
        """
        Initialize the persistence manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._local = threading.local()

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self._init_schema()
        logger.info(f"Persistence initialized at {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Get a thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.conn.execute("PRAGMA foreign_keys = ON")

        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise

    def _init_schema(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Schema version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)

            # Check current version
            cursor.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()
            current_version = row[0] if row else 0

            if current_version < SCHEMA_VERSION:
                self._migrate_schema(cursor, current_version)
                cursor.execute("DELETE FROM schema_version")
                cursor.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,)
                )

            conn.commit()

    def _migrate_schema(self, cursor, from_version: int):
        """Apply schema migrations."""
        if from_version < 1:
            # Initial schema
            # Note: We do NOT persist raw traces - only discovered assets

            # Tools table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tools (
                    name TEXT PRIMARY KEY,
                    models JSON NOT NULL,
                    trace_count INTEGER DEFAULT 0,
                    metadata JSON,
                    last_updated TEXT NOT NULL
                )
            """)

            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    name TEXT PRIMARY KEY,
                    provider TEXT,
                    tools JSON NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    last_updated TEXT NOT NULL
                )
            """)

            # MCP Servers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mcp_servers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    transport TEXT,
                    endpoint TEXT,
                    command TEXT,
                    args JSON,
                    env JSON,
                    source_file TEXT,
                    source_env_var TEXT,
                    metadata JSON,
                    last_updated TEXT NOT NULL
                )
            """)

            # MCP Tools table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mcp_tools (
                    id TEXT PRIMARY KEY,
                    server_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    schema JSON,
                    usage JSON,
                    metadata JSON,
                    last_updated TEXT NOT NULL,
                    FOREIGN KEY (server_id) REFERENCES mcp_servers(id)
                )
            """)

            # MCP Resources table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mcp_resources (
                    id TEXT PRIMARY KEY,
                    server_id TEXT NOT NULL,
                    uri TEXT NOT NULL,
                    name TEXT,
                    type TEXT,
                    mime_type TEXT,
                    description TEXT,
                    usage JSON,
                    metadata JSON,
                    last_updated TEXT NOT NULL,
                    FOREIGN KEY (server_id) REFERENCES mcp_servers(id)
                )
            """)

            # Tool mappings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plugin_name TEXT NOT NULL,
                    attributes JSON NOT NULL,
                    identity JSON NOT NULL,
                    match_type TEXT DEFAULT 'all',
                    created_at TEXT NOT NULL
                )
            """)

            # Discovery status table (for resuming after restart)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS discovery_status (
                    key TEXT PRIMARY KEY,
                    value JSON NOT NULL
                )
            """)

            # Create indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_tools_last_updated ON tools(last_updated)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_mcp_tools_server ON mcp_tools(server_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_mcp_resources_server ON mcp_resources(server_id)"
            )

            logger.info("Created database schema v1")

        if from_version < 2:
            # Schema v2: Add plugin_instances table for multi-instance support
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS plugin_instances (
                    instance_id TEXT PRIMARY KEY,
                    plugin_type TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    config JSON NOT NULL,
                    status TEXT DEFAULT 'stopped',
                    auto_start INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_plugin_instances_type ON plugin_instances(plugin_type)"
            )

            logger.info("Created database schema v2 (plugin_instances)")

        if from_version < 3:
            # Schema v3: Add agents, downstream_systems, lineage tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    confidence TEXT DEFAULT 'medium',
                    tools_used JSON,
                    models_used JSON,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    metadata JSON
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS downstream_systems (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT DEFAULT 'unknown',
                    endpoint TEXT,
                    tools_connecting JSON,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    metadata JSON
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    weight INTEGER DEFAULT 1,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    UNIQUE(source_id, target_id, relationship_type)
                )
            """)

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_downstream_name ON downstream_systems(name)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_lineage_source ON lineage(source_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_lineage_target ON lineage(target_id)"
            )

            logger.info("Created database schema v3 (agents, downstream_systems, lineage)")

    # =========================================================================
    # Plugin Instance Operations
    # =========================================================================

    def save_plugin_instance(
        self,
        instance_id: str,
        plugin_type: str,
        display_name: str,
        config: Dict[str, Any],
        status: str = 'stopped',
        auto_start: bool = False,
    ):
        """
        Save or update a plugin instance configuration.

        Args:
            instance_id: Unique identifier for this instance
            plugin_type: Type of plugin (e.g., 'databricks', 'opentelemetry')
            display_name: Human-readable name
            config: Plugin configuration (credentials should be masked or omitted)
            status: Current status ('running', 'stopped', 'error')
            auto_start: Whether to auto-start on API startup
        """
        with self._get_connection() as conn:
            now = datetime.utcnow().isoformat()
            conn.execute("""
                INSERT INTO plugin_instances (
                    instance_id, plugin_type, display_name, config, status,
                    auto_start, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(instance_id) DO UPDATE SET
                    plugin_type = ?,
                    display_name = ?,
                    config = ?,
                    status = ?,
                    auto_start = ?,
                    updated_at = ?
            """, (
                instance_id, plugin_type, display_name, json.dumps(config),
                status, 1 if auto_start else 0, now, now,
                # Update values
                plugin_type, display_name, json.dumps(config),
                status, 1 if auto_start else 0, now,
            ))
            conn.commit()
            logger.debug(f"Saved plugin instance: {instance_id}")

    def load_plugin_instances(
        self,
        plugin_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load plugin instances, optionally filtered by type.

        Args:
            plugin_type: Optional plugin type to filter by

        Returns:
            List of plugin instance configurations
        """
        with self._get_connection() as conn:
            if plugin_type:
                cursor = conn.execute(
                    "SELECT * FROM plugin_instances WHERE plugin_type = ? ORDER BY created_at",
                    (plugin_type,)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM plugin_instances ORDER BY created_at"
                )

            return [
                {
                    'instance_id': row['instance_id'],
                    'plugin_type': row['plugin_type'],
                    'display_name': row['display_name'],
                    'config': json.loads(row['config']),
                    'status': row['status'],
                    'auto_start': bool(row['auto_start']),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                }
                for row in cursor
            ]

    def get_plugin_instance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific plugin instance by ID.

        Args:
            instance_id: The instance ID to look up

        Returns:
            Plugin instance configuration or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM plugin_instances WHERE instance_id = ?",
                (instance_id,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    'instance_id': row['instance_id'],
                    'plugin_type': row['plugin_type'],
                    'display_name': row['display_name'],
                    'config': json.loads(row['config']),
                    'status': row['status'],
                    'auto_start': bool(row['auto_start']),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                }
            return None

    def update_instance_status(self, instance_id: str, status: str) -> bool:
        """
        Update the status of a plugin instance.

        Args:
            instance_id: The instance ID to update
            status: New status ('running', 'stopped', 'error')

        Returns:
            True if updated, False if instance not found
        """
        with self._get_connection() as conn:
            now = datetime.utcnow().isoformat()
            cursor = conn.execute(
                "UPDATE plugin_instances SET status = ?, updated_at = ? WHERE instance_id = ?",
                (status, now, instance_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_plugin_instance(self, instance_id: str) -> bool:
        """
        Delete a plugin instance configuration.

        Args:
            instance_id: The instance ID to delete

        Returns:
            True if deleted, False if instance not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM plugin_instances WHERE instance_id = ?",
                (instance_id,)
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted plugin instance: {instance_id}")
            return deleted

    # =========================================================================
    # Tool Operations
    # =========================================================================

    def save_tool(self, name: str, models: List[str], trace_count: int,
                  metadata: Optional[Dict] = None):
        """Save or update a discovered tool."""
        with self._get_connection() as conn:
            now = datetime.utcnow().isoformat()
            conn.execute("""
                INSERT INTO tools (name, models, trace_count, metadata, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    models = ?,
                    trace_count = ?,
                    metadata = ?,
                    last_updated = ?
            """, (
                name, json.dumps(models), trace_count, json.dumps(metadata or {}), now,
                json.dumps(models), trace_count, json.dumps(metadata or {}), now
            ))
            conn.commit()

    def load_tools(self) -> Dict[str, Dict[str, Any]]:
        """Load all tools."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name, models, trace_count, metadata FROM tools"
            )
            tools = {}
            for row in cursor:
                tools[row['name']] = {
                    'models': set(json.loads(row['models'])),
                    'traces': [],  # Traces are loaded separately if needed
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'trace_count': row['trace_count'],
                }
            return tools

    # =========================================================================
    # Model Operations
    # =========================================================================

    def save_model(self, name: str, provider: str, tools: List[str],
                   usage_count: int):
        """Save or update a discovered model."""
        with self._get_connection() as conn:
            now = datetime.utcnow().isoformat()
            conn.execute("""
                INSERT INTO models (name, provider, tools, usage_count, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    provider = ?,
                    tools = ?,
                    usage_count = ?,
                    last_updated = ?
            """, (
                name, provider, json.dumps(tools), usage_count, now,
                provider, json.dumps(tools), usage_count, now
            ))
            conn.commit()

    def load_models(self) -> Dict[str, Dict[str, Any]]:
        """Load all models."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name, provider, tools, usage_count FROM models"
            )
            return {
                row['name']: {
                    'provider': row['provider'],
                    'tools': set(json.loads(row['tools'])),
                    'usage_count': row['usage_count'],
                }
                for row in cursor
            }

    # =========================================================================
    # Agent Operations
    # =========================================================================

    def save_agent(self, agent_id: str, name: str, confidence: str = 'medium',
                   tools_used: Optional[List[str]] = None,
                   models_used: Optional[List[str]] = None,
                   first_seen: Optional[str] = None,
                   metadata: Optional[Dict] = None):
        """Save or update a discovered agent."""
        with self._get_connection() as conn:
            now = datetime.utcnow().isoformat()
            conn.execute("""
                INSERT INTO agents (id, name, confidence, tools_used, models_used,
                                    first_seen, last_seen, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = ?,
                    confidence = ?,
                    tools_used = ?,
                    models_used = ?,
                    last_seen = ?,
                    metadata = ?
            """, (
                agent_id, name, confidence,
                json.dumps(tools_used or []),
                json.dumps(models_used or []),
                first_seen or now, now,
                json.dumps(metadata or {}),
                # Update values
                name, confidence,
                json.dumps(tools_used or []),
                json.dumps(models_used or []),
                now,
                json.dumps(metadata or {}),
            ))
            conn.commit()

    def load_agents(self) -> Dict[str, Dict[str, Any]]:
        """Load all agents."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM agents")
            agents = {}
            for row in cursor:
                agents[row['id']] = {
                    'id': row['id'],
                    'name': row['name'],
                    'confidence': row['confidence'],
                    'tools_used': json.loads(row['tools_used']) if row['tools_used'] else [],
                    'models_used': json.loads(row['models_used']) if row['models_used'] else [],
                    'first_seen': row['first_seen'],
                    'last_seen': row['last_seen'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                }
            return agents

    # =========================================================================
    # Downstream System Operations
    # =========================================================================

    def save_downstream_system(self, system_id: str, name: str,
                               system_type: str = 'unknown',
                               endpoint: Optional[str] = None,
                               tools_connecting: Optional[List[str]] = None,
                               first_seen: Optional[str] = None,
                               metadata: Optional[Dict] = None):
        """Save or update a discovered downstream system."""
        with self._get_connection() as conn:
            now = datetime.utcnow().isoformat()
            conn.execute("""
                INSERT INTO downstream_systems (id, name, type, endpoint, tools_connecting,
                                                first_seen, last_seen, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = ?,
                    type = ?,
                    endpoint = ?,
                    tools_connecting = ?,
                    last_seen = ?,
                    metadata = ?
            """, (
                system_id, name, system_type, endpoint,
                json.dumps(tools_connecting or []),
                first_seen or now, now,
                json.dumps(metadata or {}),
                # Update values
                name, system_type, endpoint,
                json.dumps(tools_connecting or []),
                now,
                json.dumps(metadata or {}),
            ))
            conn.commit()

    def load_downstream_systems(self) -> Dict[str, Dict[str, Any]]:
        """Load all downstream systems."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM downstream_systems")
            systems = {}
            for row in cursor:
                systems[row['id']] = {
                    'id': row['id'],
                    'name': row['name'],
                    'type': row['type'],
                    'endpoint': row['endpoint'],
                    'tools_connecting': json.loads(row['tools_connecting']) if row['tools_connecting'] else [],
                    'first_seen': row['first_seen'],
                    'last_seen': row['last_seen'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                }
            return systems

    # =========================================================================
    # Lineage Operations
    # =========================================================================

    def save_lineage(self, source_id: str, source_type: str,
                     target_id: str, target_type: str,
                     relationship_type: str, weight: int = 1):
        """Save or update a lineage relationship."""
        with self._get_connection() as conn:
            now = datetime.utcnow().isoformat()
            conn.execute("""
                INSERT INTO lineage (source_id, source_type, target_id, target_type,
                                     relationship_type, weight, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, relationship_type) DO UPDATE SET
                    weight = weight + 1,
                    last_seen = ?
            """, (
                source_id, source_type, target_id, target_type,
                relationship_type, weight, now, now,
                now,
            ))
            conn.commit()

    def load_lineage(self, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load lineage relationships, optionally filtered by source."""
        with self._get_connection() as conn:
            if source_id:
                cursor = conn.execute(
                    "SELECT * FROM lineage WHERE source_id = ? OR target_id = ?",
                    (source_id, source_id)
                )
            else:
                cursor = conn.execute("SELECT * FROM lineage")

            return [
                {
                    'source_id': row['source_id'],
                    'source_type': row['source_type'],
                    'target_id': row['target_id'],
                    'target_type': row['target_type'],
                    'relationship_type': row['relationship_type'],
                    'weight': row['weight'],
                    'first_seen': row['first_seen'],
                    'last_seen': row['last_seen'],
                }
                for row in cursor
            ]

    # =========================================================================
    # MCP Server Operations
    # =========================================================================

    def save_mcp_server(self, server_id: str, name: str, **kwargs):
        """Save or update an MCP server."""
        with self._get_connection() as conn:
            now = datetime.utcnow().isoformat()
            conn.execute("""
                INSERT INTO mcp_servers (
                    id, name, transport, endpoint, command, args, env,
                    source_file, source_env_var, metadata, last_updated
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = ?,
                    transport = ?,
                    endpoint = ?,
                    command = ?,
                    args = ?,
                    env = ?,
                    source_file = ?,
                    source_env_var = ?,
                    metadata = ?,
                    last_updated = ?
            """, (
                server_id, name,
                kwargs.get('transport'),
                kwargs.get('endpoint'),
                kwargs.get('command'),
                json.dumps(kwargs.get('args')) if kwargs.get('args') else None,
                json.dumps(kwargs.get('env')) if kwargs.get('env') else None,
                kwargs.get('source_file'),
                kwargs.get('source_env_var'),
                json.dumps(kwargs.get('metadata', {})),
                now,
                # Update values
                name,
                kwargs.get('transport'),
                kwargs.get('endpoint'),
                kwargs.get('command'),
                json.dumps(kwargs.get('args')) if kwargs.get('args') else None,
                json.dumps(kwargs.get('env')) if kwargs.get('env') else None,
                kwargs.get('source_file'),
                kwargs.get('source_env_var'),
                json.dumps(kwargs.get('metadata', {})),
                now,
            ))
            conn.commit()

    def load_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """Load all MCP servers."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM mcp_servers")
            servers = {}
            for row in cursor:
                servers[row['id']] = {
                    'id': row['id'],
                    'name': row['name'],
                    'transport': row['transport'],
                    'endpoint': row['endpoint'],
                    'command': row['command'],
                    'args': json.loads(row['args']) if row['args'] else None,
                    'env': json.loads(row['env']) if row['env'] else None,
                    'source_file': row['source_file'],
                    'source_env_var': row['source_env_var'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                }
            return servers

    # =========================================================================
    # MCP Tool Operations
    # =========================================================================

    def save_mcp_tool(self, tool_id: str, server_id: str, name: str, **kwargs):
        """Save or update an MCP tool."""
        with self._get_connection() as conn:
            now = datetime.utcnow().isoformat()
            conn.execute("""
                INSERT INTO mcp_tools (
                    id, server_id, name, description, schema, usage, metadata, last_updated
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    server_id = ?,
                    name = ?,
                    description = ?,
                    schema = ?,
                    usage = ?,
                    metadata = ?,
                    last_updated = ?
            """, (
                tool_id, server_id, name,
                kwargs.get('description'),
                json.dumps(kwargs.get('schema')) if kwargs.get('schema') else None,
                json.dumps(kwargs.get('usage')) if kwargs.get('usage') else None,
                json.dumps(kwargs.get('metadata', {})),
                now,
                # Update values
                server_id, name,
                kwargs.get('description'),
                json.dumps(kwargs.get('schema')) if kwargs.get('schema') else None,
                json.dumps(kwargs.get('usage')) if kwargs.get('usage') else None,
                json.dumps(kwargs.get('metadata', {})),
                now,
            ))
            conn.commit()

    def load_mcp_tools(self) -> Dict[str, Dict[str, Any]]:
        """Load all MCP tools."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM mcp_tools")
            tools = {}
            for row in cursor:
                tools[row['id']] = {
                    'id': row['id'],
                    'server_id': row['server_id'],
                    'name': row['name'],
                    'description': row['description'],
                    'schema': json.loads(row['schema']) if row['schema'] else None,
                    'usage': json.loads(row['usage']) if row['usage'] else None,
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                }
            return tools

    # =========================================================================
    # MCP Resource Operations
    # =========================================================================

    def save_mcp_resource(self, resource_id: str, server_id: str, uri: str, **kwargs):
        """Save or update an MCP resource."""
        with self._get_connection() as conn:
            now = datetime.utcnow().isoformat()
            conn.execute("""
                INSERT INTO mcp_resources (
                    id, server_id, uri, name, type, mime_type, description,
                    usage, metadata, last_updated
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    server_id = ?,
                    uri = ?,
                    name = ?,
                    type = ?,
                    mime_type = ?,
                    description = ?,
                    usage = ?,
                    metadata = ?,
                    last_updated = ?
            """, (
                resource_id, server_id, uri,
                kwargs.get('name'),
                kwargs.get('type'),
                kwargs.get('mime_type'),
                kwargs.get('description'),
                json.dumps(kwargs.get('usage')) if kwargs.get('usage') else None,
                json.dumps(kwargs.get('metadata', {})),
                now,
                # Update values
                server_id, uri,
                kwargs.get('name'),
                kwargs.get('type'),
                kwargs.get('mime_type'),
                kwargs.get('description'),
                json.dumps(kwargs.get('usage')) if kwargs.get('usage') else None,
                json.dumps(kwargs.get('metadata', {})),
                now,
            ))
            conn.commit()

    def load_mcp_resources(self) -> Dict[str, Dict[str, Any]]:
        """Load all MCP resources."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM mcp_resources")
            resources = {}
            for row in cursor:
                resources[row['id']] = {
                    'id': row['id'],
                    'server_id': row['server_id'],
                    'uri': row['uri'],
                    'name': row['name'],
                    'type': row['type'],
                    'mime_type': row['mime_type'],
                    'description': row['description'],
                    'usage': json.loads(row['usage']) if row['usage'] else None,
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                }
            return resources

    # =========================================================================
    # Tool Mapping Operations
    # =========================================================================

    def save_mapping(self, plugin_name: str, attributes: Dict, identity: Dict,
                     match_type: str = 'all') -> int:
        """Save a tool mapping. Returns the mapping ID."""
        with self._get_connection() as conn:
            now = datetime.utcnow().isoformat()
            cursor = conn.execute("""
                INSERT INTO tool_mappings (plugin_name, attributes, identity, match_type, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (plugin_name, json.dumps(attributes), json.dumps(identity), match_type, now))
            conn.commit()
            return cursor.lastrowid

    def load_mappings(self, plugin_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load tool mappings, optionally filtered by plugin."""
        with self._get_connection() as conn:
            if plugin_name:
                cursor = conn.execute(
                    "SELECT * FROM tool_mappings WHERE plugin_name = ?",
                    (plugin_name,)
                )
            else:
                cursor = conn.execute("SELECT * FROM tool_mappings")

            return [
                {
                    'id': row['id'],
                    'plugin_name': row['plugin_name'],
                    'attributes': json.loads(row['attributes']),
                    'identity': json.loads(row['identity']),
                    'match_type': row['match_type'],
                    'created_at': row['created_at'],
                }
                for row in cursor
            ]

    def delete_mapping(self, mapping_id: int) -> bool:
        """Delete a tool mapping by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM tool_mappings WHERE id = ?",
                (mapping_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    # =========================================================================
    # Discovery Status Operations
    # =========================================================================

    def save_status(self, key: str, value: Any):
        """Save a status value."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO discovery_status (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?
            """, (key, json.dumps(value), json.dumps(value)))
            conn.commit()

    def load_status(self, key: str) -> Optional[Any]:
        """Load a status value."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT value FROM discovery_status WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            return json.loads(row['value']) if row else None

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def export_all(self) -> Dict[str, Any]:
        """Export all data for backup/migration."""
        return {
            'schema_version': SCHEMA_VERSION,
            'exported_at': datetime.utcnow().isoformat(),
            'plugin_instances': self.load_plugin_instances(),
            'tools': {k: {**v, 'models': list(v['models'])} for k, v in self.load_tools().items()},
            'models': {k: {**v, 'tools': list(v['tools'])} for k, v in self.load_models().items()},
            'agents': self.load_agents(),
            'downstream_systems': self.load_downstream_systems(),
            'lineage': self.load_lineage(),
            'mcp_servers': self.load_mcp_servers(),
            'mcp_tools': self.load_mcp_tools(),
            'mcp_resources': self.load_mcp_resources(),
            'mappings': self.load_mappings(),
        }

    def clear_all(self):
        """Clear all data (useful for testing)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM lineage")
            conn.execute("DELETE FROM downstream_systems")
            conn.execute("DELETE FROM agents")
            conn.execute("DELETE FROM mcp_resources")
            conn.execute("DELETE FROM mcp_tools")
            conn.execute("DELETE FROM mcp_servers")
            conn.execute("DELETE FROM models")
            conn.execute("DELETE FROM tools")
            conn.execute("DELETE FROM tool_mappings")
            conn.execute("DELETE FROM discovery_status")
            conn.execute("DELETE FROM plugin_instances")
            conn.commit()
            logger.info("Cleared all persistence data")

    def close(self):
        """Close database connections."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
