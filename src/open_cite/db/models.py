"""
SQLAlchemy ORM model definitions for all OpenCITE tables.

Uses SQLAlchemy 2.0 DeclarativeBase. ``sqlalchemy.JSON`` maps to TEXT on
SQLite and JSONB on PostgreSQL — no custom type adapters needed.
"""

from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.types import JSON
from sqlalchemy.orm import DeclarativeBase, relationship


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
    confidence = Column(String, default="medium")
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

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String, nullable=False, index=True)
    source_type = Column(String, nullable=False)
    target_id = Column(String, nullable=False, index=True)
    target_type = Column(String, nullable=False)
    relationship_type = Column(String, nullable=False)
    weight = Column(Integer, default=1)
    first_seen = Column(String, nullable=False)
    last_seen = Column(String, nullable=False)

    __table_args__ = (
        UniqueConstraint("source_id", "target_id", "relationship_type",
                         name="uq_lineage_src_tgt_rel"),
    )


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
    server_id = Column(String, ForeignKey("mcp_servers.id"), nullable=False, index=True)
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
    server_id = Column(String, ForeignKey("mcp_servers.id"), nullable=False, index=True)
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

    id = Column(Integer, primary_key=True, autoincrement=True)
    plugin_name = Column(String, nullable=False, index=True)
    attributes = Column(JSON, nullable=False)
    identity = Column(JSON, nullable=False)
    match_type = Column(String, default="all")
    created_at = Column(String, nullable=False)
