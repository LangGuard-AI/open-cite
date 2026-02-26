"""
Open-CITE SQLAlchemy Persistence Layer.

Provides durable storage for discovered assets (tools, models, agents,
downstream systems, lineage, MCP entities, and discovery status).

Plugin instance persistence has moved to ``PluginConfigStore``.
Identity mapping persistence has moved to ``ToolIdentifier``.
"""

import functools
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from open_cite.db import (
    get_session,
    init_db,
    Tool,
    Model,
    Agent,
    DownstreamSystem,
    Lineage,
    McpServer,
    McpTool,
    McpResource,
    DiscoveryStatus,
)

logger = logging.getLogger(__name__)

_CONCURRENT_KEYWORDS = ("ConcurrentAppendException", "DELTA_CONCURRENT_APPEND", "concurrent update")


def _retry_on_concurrent_write(func):
    """Retry a persistence method up to 3 times on Delta concurrent-write conflicts."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(3):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                msg = str(exc)
                if any(kw in msg for kw in _CONCURRENT_KEYWORDS):
                    if attempt < 2:
                        delay = 0.2 * (attempt + 1)
                        logger.debug("Concurrent write conflict, retrying in %.1fs (%s)", delay, func.__name__)
                        time.sleep(delay)
                        continue
                raise
    return wrapper


class PersistenceManager:
    """
    SQLAlchemy-based persistence for Open-CITE discovered assets.

    Thread-safe storage for tools, models, agents, downstream systems,
    lineage, MCP entities, and discovery status.
    """

    def __init__(self):
        """Initialize the persistence manager using the shared SQLAlchemy engine."""
        init_db()
        logger.info("Persistence initialized (SQLAlchemy)")

    # =========================================================================
    # Tool Operations
    # =========================================================================

    @_retry_on_concurrent_write
    def save_tool(self, name: str, models: List[str], trace_count: int,
                  metadata: Optional[Dict] = None):
        """Save or update a discovered tool."""
        session = get_session()
        try:
            now = datetime.utcnow().isoformat()
            existing = session.get(Tool, name)
            if existing:
                existing.models = models
                existing.trace_count = trace_count
                existing.metadata_ = metadata or {}
                existing.last_updated = now
            else:
                session.add(Tool(
                    name=name,
                    models=models,
                    trace_count=trace_count,
                    metadata_=metadata or {},
                    last_updated=now,
                ))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def load_tools(self) -> Dict[str, Dict[str, Any]]:
        """Load all tools."""
        session = get_session()
        try:
            rows = session.query(Tool).all()
            tools = {}
            for row in rows:
                tools[row.name] = {
                    'models': set(row.models or []),
                    'traces': [],
                    'metadata': row.metadata_ or {},
                    'trace_count': row.trace_count,
                }
            return tools
        finally:
            session.close()

    # =========================================================================
    # Model Operations
    # =========================================================================

    @_retry_on_concurrent_write
    def save_model(self, name: str, provider: str, tools: List[str],
                   usage_count: int):
        """Save or update a discovered model."""
        session = get_session()
        try:
            now = datetime.utcnow().isoformat()
            existing = session.get(Model, name)
            if existing:
                existing.provider = provider
                existing.tools = tools
                existing.usage_count = usage_count
                existing.last_updated = now
            else:
                session.add(Model(
                    name=name,
                    provider=provider,
                    tools=tools,
                    usage_count=usage_count,
                    last_updated=now,
                ))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def load_models(self) -> Dict[str, Dict[str, Any]]:
        """Load all models."""
        session = get_session()
        try:
            rows = session.query(Model).all()
            return {
                row.name: {
                    'provider': row.provider,
                    'tools': set(row.tools or []),
                    'usage_count': row.usage_count,
                }
                for row in rows
            }
        finally:
            session.close()

    # =========================================================================
    # Agent Operations
    # =========================================================================

    @_retry_on_concurrent_write
    def save_agent(self, agent_id: str, name: str,
                   tools_used: Optional[List[str]] = None,
                   models_used: Optional[List[str]] = None,
                   first_seen: Optional[str] = None,
                   metadata: Optional[Dict] = None):
        """Save or update a discovered agent."""
        session = get_session()
        try:
            now = datetime.utcnow().isoformat()
            existing = session.get(Agent, agent_id)
            if existing:
                existing.name = name
                existing.tools_used = tools_used or []
                existing.models_used = models_used or []
                existing.last_seen = now
                existing.metadata_ = metadata or {}
            else:
                session.add(Agent(
                    id=agent_id,
                    name=name,
                    tools_used=tools_used or [],
                    models_used=models_used or [],
                    first_seen=first_seen or now,
                    last_seen=now,
                    metadata_=metadata or {},
                ))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def load_agents(self) -> Dict[str, Dict[str, Any]]:
        """Load all agents."""
        session = get_session()
        try:
            rows = session.query(Agent).all()
            agents = {}
            for row in rows:
                agents[row.id] = {
                    'id': row.id,
                    'name': row.name,
                    'tools_used': row.tools_used or [],
                    'models_used': row.models_used or [],
                    'first_seen': row.first_seen,
                    'last_seen': row.last_seen,
                    'metadata': row.metadata_ or {},
                }
            return agents
        finally:
            session.close()

    # =========================================================================
    # Downstream System Operations
    # =========================================================================

    @_retry_on_concurrent_write
    def save_downstream_system(self, system_id: str, name: str,
                               system_type: str = 'unknown',
                               endpoint: Optional[str] = None,
                               tools_connecting: Optional[List[str]] = None,
                               first_seen: Optional[str] = None,
                               metadata: Optional[Dict] = None):
        """Save or update a discovered downstream system."""
        session = get_session()
        try:
            now = datetime.utcnow().isoformat()
            existing = session.get(DownstreamSystem, system_id)
            if existing:
                existing.name = name
                existing.type = system_type
                existing.endpoint = endpoint
                existing.tools_connecting = tools_connecting or []
                existing.last_seen = now
                existing.metadata_ = metadata or {}
            else:
                session.add(DownstreamSystem(
                    id=system_id,
                    name=name,
                    type=system_type,
                    endpoint=endpoint,
                    tools_connecting=tools_connecting or [],
                    first_seen=first_seen or now,
                    last_seen=now,
                    metadata_=metadata or {},
                ))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def load_downstream_systems(self) -> Dict[str, Dict[str, Any]]:
        """Load all downstream systems."""
        session = get_session()
        try:
            rows = session.query(DownstreamSystem).all()
            systems = {}
            for row in rows:
                systems[row.id] = {
                    'id': row.id,
                    'name': row.name,
                    'type': row.type,
                    'endpoint': row.endpoint,
                    'tools_connecting': row.tools_connecting or [],
                    'first_seen': row.first_seen,
                    'last_seen': row.last_seen,
                    'metadata': row.metadata_ or {},
                }
            return systems
        finally:
            session.close()

    # =========================================================================
    # Lineage Operations
    # =========================================================================

    @_retry_on_concurrent_write
    def save_lineage(self, source_id: str, source_type: str,
                     target_id: str, target_type: str,
                     relationship_type: str, weight: int = 1):
        """Save or update a lineage relationship."""
        session = get_session()
        try:
            now = datetime.utcnow().isoformat()
            existing = (
                session.query(Lineage)
                .filter_by(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=relationship_type,
                )
                .first()
            )
            if existing:
                existing.weight = existing.weight + 1
                existing.last_seen = now
            else:
                session.add(Lineage(
                    source_id=source_id,
                    source_type=source_type,
                    target_id=target_id,
                    target_type=target_type,
                    relationship_type=relationship_type,
                    weight=weight,
                    first_seen=now,
                    last_seen=now,
                ))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def load_lineage(self, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load lineage relationships, optionally filtered by source."""
        session = get_session()
        try:
            query = session.query(Lineage)
            if source_id:
                query = query.filter(
                    (Lineage.source_id == source_id) | (Lineage.target_id == source_id)
                )
            rows = query.all()
            return [
                {
                    'source_id': row.source_id,
                    'source_type': row.source_type,
                    'target_id': row.target_id,
                    'target_type': row.target_type,
                    'relationship_type': row.relationship_type,
                    'weight': row.weight,
                    'first_seen': row.first_seen,
                    'last_seen': row.last_seen,
                }
                for row in rows
            ]
        finally:
            session.close()

    # =========================================================================
    # MCP Server Operations
    # =========================================================================

    @_retry_on_concurrent_write
    def save_mcp_server(self, server_id: str, name: str, **kwargs):
        """Save or update an MCP server."""
        session = get_session()
        try:
            now = datetime.utcnow().isoformat()
            existing = session.get(McpServer, server_id)
            if existing:
                existing.name = name
                existing.transport = kwargs.get('transport')
                existing.endpoint = kwargs.get('endpoint')
                existing.command = kwargs.get('command')
                existing.args = kwargs.get('args')
                existing.env = kwargs.get('env')
                existing.source_file = kwargs.get('source_file')
                existing.source_env_var = kwargs.get('source_env_var')
                existing.metadata_ = kwargs.get('metadata', {})
                existing.last_updated = now
            else:
                session.add(McpServer(
                    id=server_id,
                    name=name,
                    transport=kwargs.get('transport'),
                    endpoint=kwargs.get('endpoint'),
                    command=kwargs.get('command'),
                    args=kwargs.get('args'),
                    env=kwargs.get('env'),
                    source_file=kwargs.get('source_file'),
                    source_env_var=kwargs.get('source_env_var'),
                    metadata_=kwargs.get('metadata', {}),
                    last_updated=now,
                ))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def load_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """Load all MCP servers."""
        session = get_session()
        try:
            rows = session.query(McpServer).all()
            servers = {}
            for row in rows:
                servers[row.id] = {
                    'id': row.id,
                    'name': row.name,
                    'transport': row.transport,
                    'endpoint': row.endpoint,
                    'command': row.command,
                    'args': row.args,
                    'env': row.env,
                    'source_file': row.source_file,
                    'source_env_var': row.source_env_var,
                    'metadata': row.metadata_ or {},
                }
            return servers
        finally:
            session.close()

    # =========================================================================
    # MCP Tool Operations
    # =========================================================================

    @_retry_on_concurrent_write
    def save_mcp_tool(self, tool_id: str, server_id: str, name: str, **kwargs):
        """Save or update an MCP tool."""
        session = get_session()
        try:
            now = datetime.utcnow().isoformat()
            existing = session.get(McpTool, tool_id)
            if existing:
                existing.server_id = server_id
                existing.name = name
                existing.description = kwargs.get('description')
                existing.schema_ = kwargs.get('schema')
                existing.usage = kwargs.get('usage')
                existing.metadata_ = kwargs.get('metadata', {})
                existing.last_updated = now
            else:
                session.add(McpTool(
                    id=tool_id,
                    server_id=server_id,
                    name=name,
                    description=kwargs.get('description'),
                    schema_=kwargs.get('schema'),
                    usage=kwargs.get('usage'),
                    metadata_=kwargs.get('metadata', {}),
                    last_updated=now,
                ))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def load_mcp_tools(self) -> Dict[str, Dict[str, Any]]:
        """Load all MCP tools."""
        session = get_session()
        try:
            rows = session.query(McpTool).all()
            tools = {}
            for row in rows:
                tools[row.id] = {
                    'id': row.id,
                    'server_id': row.server_id,
                    'name': row.name,
                    'description': row.description,
                    'schema': row.schema_,
                    'usage': row.usage,
                    'metadata': row.metadata_ or {},
                }
            return tools
        finally:
            session.close()

    # =========================================================================
    # MCP Resource Operations
    # =========================================================================

    @_retry_on_concurrent_write
    def save_mcp_resource(self, resource_id: str, server_id: str, uri: str, **kwargs):
        """Save or update an MCP resource."""
        session = get_session()
        try:
            now = datetime.utcnow().isoformat()
            existing = session.get(McpResource, resource_id)
            if existing:
                existing.server_id = server_id
                existing.uri = uri
                existing.name = kwargs.get('name')
                existing.type = kwargs.get('type')
                existing.mime_type = kwargs.get('mime_type')
                existing.description = kwargs.get('description')
                existing.usage = kwargs.get('usage')
                existing.metadata_ = kwargs.get('metadata', {})
                existing.last_updated = now
            else:
                session.add(McpResource(
                    id=resource_id,
                    server_id=server_id,
                    uri=uri,
                    name=kwargs.get('name'),
                    type=kwargs.get('type'),
                    mime_type=kwargs.get('mime_type'),
                    description=kwargs.get('description'),
                    usage=kwargs.get('usage'),
                    metadata_=kwargs.get('metadata', {}),
                    last_updated=now,
                ))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def load_mcp_resources(self) -> Dict[str, Dict[str, Any]]:
        """Load all MCP resources."""
        session = get_session()
        try:
            rows = session.query(McpResource).all()
            resources = {}
            for row in rows:
                resources[row.id] = {
                    'id': row.id,
                    'server_id': row.server_id,
                    'uri': row.uri,
                    'name': row.name,
                    'type': row.type,
                    'mime_type': row.mime_type,
                    'description': row.description,
                    'usage': row.usage,
                    'metadata': row.metadata_ or {},
                }
            return resources
        finally:
            session.close()

    # =========================================================================
    # Discovery Status Operations
    # =========================================================================

    @_retry_on_concurrent_write
    def save_status(self, key: str, value: Any):
        """Save a status value."""
        session = get_session()
        try:
            existing = session.get(DiscoveryStatus, key)
            if existing:
                existing.value = value
            else:
                session.add(DiscoveryStatus(key=key, value=value))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def load_status(self, key: str) -> Optional[Any]:
        """Load a status value."""
        session = get_session()
        try:
            row = session.get(DiscoveryStatus, key)
            return row.value if row else None
        finally:
            session.close()

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def export_all(self) -> Dict[str, Any]:
        """Export all data for backup/migration."""
        return {
            'exported_at': datetime.utcnow().isoformat(),
            'tools': {k: {**v, 'models': list(v['models'])} for k, v in self.load_tools().items()},
            'models': {k: {**v, 'tools': list(v['tools'])} for k, v in self.load_models().items()},
            'agents': self.load_agents(),
            'downstream_systems': self.load_downstream_systems(),
            'lineage': self.load_lineage(),
            'mcp_servers': self.load_mcp_servers(),
            'mcp_tools': self.load_mcp_tools(),
            'mcp_resources': self.load_mcp_resources(),
        }

    def clear_all(self, max_retries: int = 5):
        """Clear all asset data with retries for concurrent write conflicts."""
        import time as _time

        for attempt in range(1, max_retries + 1):
            session = get_session()
            try:
                session.query(Lineage).delete()
                session.query(DownstreamSystem).delete()
                session.query(Agent).delete()
                session.query(McpResource).delete()
                session.query(McpTool).delete()
                session.query(McpServer).delete()
                session.query(Model).delete()
                session.query(Tool).delete()
                session.query(DiscoveryStatus).delete()
                session.commit()
                logger.info("Cleared all persistence data")
                return
            except Exception as e:
                session.rollback()
                if attempt < max_retries:
                    delay = 0.5 * attempt
                    logger.warning("clear_all attempt %d/%d failed: %s — retrying in %.1fs",
                                   attempt, max_retries, e, delay)
                    _time.sleep(delay)
                else:
                    raise
            finally:
                session.close()

    def close(self):
        """Close database connections (delegates to close_db)."""
        from open_cite.db import close_db
        close_db()
