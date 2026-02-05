"""
Schema validation and export utilities for OpenCITE.

This module provides utilities for validating and exporting OpenCITE data
according to the OpenCITE JSON schema.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Schema version
SCHEMA_VERSION = "1.0.0"


class OpenCiteExporter:
    """
    Exports OpenCITE discovery data to JSON format according to the schema.
    """

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize the exporter.

        Args:
            schema_path: Path to the JSON schema file (optional)
        """
        self.schema_path = schema_path
        self.schema = None

        if schema_path:
            self._load_schema()

    def _load_schema(self):
        """Load the JSON schema for validation."""
        try:
            with open(self.schema_path, 'r') as f:
                self.schema = json.load(f)
            logger.info(f"Loaded schema from {self.schema_path}")
        except Exception as e:
            logger.warning(f"Failed to load schema: {e}")

    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate data against the schema.

        Args:
            data: Data to validate

        Returns:
            True if valid, False otherwise
        """
        if not self.schema:
            logger.warning("No schema loaded, skipping validation")
            return True

        try:
            import jsonschema
            jsonschema.validate(instance=data, schema=self.schema)
            return True
        except ImportError:
            logger.warning("jsonschema package not installed, skipping validation")
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def export_discovery(
        self,
        tools: List[Dict[str, Any]],
        models: List[Dict[str, Any]],
        data_assets: List[Dict[str, Any]] = None,
        mcp_servers: List[Dict[str, Any]] = None,
        mcp_tools: List[Dict[str, Any]] = None,
        mcp_resources: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Export discovery data to JSON format.

        Args:
            tools: List of discovered tools
            models: List of discovered models
            data_assets: List of data assets (optional)
            mcp_servers: List of MCP servers (optional)
            mcp_tools: List of MCP tools (optional)
            mcp_resources: List of MCP resources (optional)
            metadata: Additional metadata (optional)

        Returns:
            JSON-serializable dictionary conforming to OpenCITE schema
        """
        # Get timestamp
        export_timestamp = datetime.utcnow().isoformat() + "Z"

        # Extract plugins list from metadata
        plugins = metadata.get("plugins", []) if metadata else []

        # Build the export according to OpenCITE schema
        export_data = {
            "opencite_version": SCHEMA_VERSION,
            "export_timestamp": export_timestamp,
            "tools": tools or [],
            "models": models or [],
            "data_assets": data_assets or [],
            "mcp_servers": mcp_servers or [],
            "mcp_tools": mcp_tools or [],
            "mcp_resources": mcp_resources or [],
            "plugins": plugins,
        }

        return export_data

    def _calculate_statistics(
        self,
        tools: List[Dict[str, Any]],
        models: List[Dict[str, Any]],
        data_assets: Optional[List[Dict[str, Any]]],
        traces: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Calculate summary statistics."""
        stats = {
            "total_tools": len(tools),
            "total_models": len(models),
            "total_data_assets": len(data_assets) if data_assets else 0,
            "total_traces": len(traces) if traces else 0,
        }

        # Count by provider
        by_provider = {}
        for model in models:
            provider = model.get("provider", "unknown")
            by_provider[provider] = by_provider.get(provider, 0) + 1

        stats["by_provider"] = by_provider

        # Count by discovery source
        by_source = {}
        for tool in tools:
            source = tool.get("discovery_source", "unknown")
            by_source[source] = by_source.get(source, 0) + 1

        for model in models:
            source = model.get("discovery_source", "unknown")
            by_source[source] = by_source.get(source, 0) + 1

        stats["by_discovery_source"] = by_source

        return stats

    def save_to_file(self, data: Dict[str, Any], filepath: str, validate: bool = True):
        """
        Save export data to a JSON file.

        Args:
            data: Data to save
            filepath: Path to save to
            validate: Whether to validate before saving
        """
        if validate and not self.validate(data):
            raise ValueError("Data validation failed")

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported data to {filepath}")


class ToolFormatter:
    """Formats tool data according to the schema."""

    @staticmethod
    def format_tool(
        tool_id: str,
        name: str,
        discovery_source: str,
        models_used: List[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Format a tool according to the schema.

        Args:
            tool_id: Unique tool identifier
            name: Tool name
            discovery_source: Source of discovery
            models_used: List of models used by this tool
            **kwargs: Additional fields

        Returns:
            Formatted tool dictionary
        """
        tool = {
            "id": tool_id,
            "name": name,
            "discovery_source": discovery_source,
        }

        # Add optional fields
        if "type" in kwargs:
            tool["type"] = kwargs["type"]
        if "description" in kwargs:
            tool["description"] = kwargs["description"]
        if "provider" in kwargs:
            tool["provider"] = kwargs["provider"]

        if models_used:
            tool["models_used"] = models_used

        if "first_seen" in kwargs:
            tool["first_seen"] = kwargs["first_seen"]
        if "last_seen" in kwargs:
            tool["last_seen"] = kwargs["last_seen"]

        if "metadata" in kwargs:
            tool["metadata"] = kwargs["metadata"]
        if "endpoints" in kwargs:
            tool["endpoints"] = kwargs["endpoints"]
        if "tags" in kwargs:
            tool["tags"] = kwargs["tags"]

        if "tool_source_name" in kwargs:
            tool["tool_source_name"] = kwargs["tool_source_name"]
        if "tool_source_id" in kwargs:
            tool["tool_source_id"] = kwargs["tool_source_id"]

        return tool


class ModelFormatter:
    """Formats model data according to the schema."""

    @staticmethod
    def format_model(
        model_id: str,
        name: str,
        discovery_source: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Format a model according to the schema.

        Args:
            model_id: Unique model identifier
            name: Model name
            discovery_source: Source of discovery
            **kwargs: Additional fields

        Returns:
            Formatted model dictionary
        """
        model = {
            "id": model_id,
            "name": name,
            "discovery_source": discovery_source,
        }

        # Add optional fields
        if "provider" in kwargs:
            model["provider"] = kwargs["provider"]
        if "model_family" in kwargs:
            model["model_family"] = kwargs["model_family"]
        if "model_version" in kwargs:
            model["model_version"] = kwargs["model_version"]
        if "modality" in kwargs:
            model["modality"] = kwargs["modality"]

        if "usage" in kwargs:
            model["usage"] = kwargs["usage"]
        if "metadata" in kwargs:
            model["metadata"] = kwargs["metadata"]
        if "catalog_info" in kwargs:
            model["catalog_info"] = kwargs["catalog_info"]
        if "tags" in kwargs:
            model["tags"] = kwargs["tags"]

        return model


class DataAssetFormatter:
    """Formats data asset information according to the schema."""

    @staticmethod
    def format_data_asset(
        asset_id: str,
        name: str,
        asset_type: str,
        discovery_source: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Format a data asset according to the schema.

        Args:
            asset_id: Unique asset identifier
            name: Asset name
            asset_type: Type of asset (catalog, schema, table, etc.)
            discovery_source: Source of discovery
            **kwargs: Additional fields

        Returns:
            Formatted data asset dictionary
        """
        asset = {
            "id": asset_id,
            "name": name,
            "type": asset_type,
            "discovery_source": discovery_source,
        }

        # Add optional fields
        if "full_name" in kwargs:
            asset["full_name"] = kwargs["full_name"]
        if "hierarchy" in kwargs:
            asset["hierarchy"] = kwargs["hierarchy"]
        if "metadata" in kwargs:
            asset["metadata"] = kwargs["metadata"]
        if "schema_details" in kwargs:
            asset["schema_details"] = kwargs["schema_details"]
        if "function_details" in kwargs:
            asset["function_details"] = kwargs["function_details"]
        if "tags" in kwargs:
            asset["tags"] = kwargs["tags"]

        return asset


class MCPServerFormatter:
    """Formats MCP server data according to the schema."""

    @staticmethod
    def format_mcp_server(
        server_id: str,
        name: str,
        discovery_source: str,
        transport: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Format an MCP server according to the schema.

        Args:
            server_id: Unique server identifier
            name: Server name
            discovery_source: Source of discovery
            transport: Transport protocol (stdio, http, sse, unknown)
            **kwargs: Additional fields

        Returns:
            Formatted MCP server dictionary
        """
        server = {
            "id": server_id,
            "name": name,
            "discovery_source": discovery_source,
            "transport": transport,
        }

        # Add optional fields
        if "endpoint" in kwargs:
            server["endpoint"] = kwargs["endpoint"]
        if "command" in kwargs:
            server["command"] = kwargs["command"]
        if "args" in kwargs:
            server["args"] = kwargs["args"]
        if "env" in kwargs:
            server["env"] = kwargs["env"]

        if "tools_provided" in kwargs:
            server["tools_provided"] = kwargs["tools_provided"]
        if "resources_provided" in kwargs:
            server["resources_provided"] = kwargs["resources_provided"]

        if "tools_count" in kwargs:
            server["tools_count"] = kwargs["tools_count"]
        if "resources_count" in kwargs:
            server["resources_count"] = kwargs["resources_count"]

        if "source_file" in kwargs:
            server["source_file"] = kwargs["source_file"]
        if "source_env_var" in kwargs:
            server["source_env_var"] = kwargs["source_env_var"]

        if "first_seen" in kwargs:
            server["first_seen"] = kwargs["first_seen"]
        if "last_seen" in kwargs:
            server["last_seen"] = kwargs["last_seen"]

        if "metadata" in kwargs:
            server["metadata"] = kwargs["metadata"]
        if "tags" in kwargs:
            server["tags"] = kwargs["tags"]

        return server


class MCPToolFormatter:
    """Formats MCP tool data according to the schema."""

    @staticmethod
    def format_mcp_tool(
        tool_id: str,
        name: str,
        server_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Format an MCP tool according to the schema.

        Args:
            tool_id: Unique tool identifier
            name: Tool name
            server_id: ID of the MCP server providing this tool
            **kwargs: Additional fields

        Returns:
            Formatted MCP tool dictionary
        """
        tool = {
            "id": tool_id,
            "name": name,
            "server_id": server_id,
        }

        # Add optional fields
        if "discovery_source" in kwargs:
            tool["discovery_source"] = kwargs["discovery_source"]
        if "description" in kwargs:
            tool["description"] = kwargs["description"]
        if "schema" in kwargs:
            tool["schema"] = kwargs["schema"]
        if "usage" in kwargs:
            tool["usage"] = kwargs["usage"]
        if "metadata" in kwargs:
            tool["metadata"] = kwargs["metadata"]
        if "tags" in kwargs:
            tool["tags"] = kwargs["tags"]

        return tool


class MCPResourceFormatter:
    """Formats MCP resource data according to the schema."""

    @staticmethod
    def format_mcp_resource(
        resource_id: str,
        uri: str,
        server_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Format an MCP resource according to the schema.

        Args:
            resource_id: Unique resource identifier
            uri: Resource URI
            server_id: ID of the MCP server providing this resource
            **kwargs: Additional fields

        Returns:
            Formatted MCP resource dictionary
        """
        resource = {
            "id": resource_id,
            "uri": uri,
            "server_id": server_id,
        }

        # Add optional fields
        if "name" in kwargs:
            resource["name"] = kwargs["name"]
        if "discovery_source" in kwargs:
            resource["discovery_source"] = kwargs["discovery_source"]
        if "type" in kwargs:
            resource["type"] = kwargs["type"]
        if "mime_type" in kwargs:
            resource["mime_type"] = kwargs["mime_type"]
        if "description" in kwargs:
            resource["description"] = kwargs["description"]
        if "usage" in kwargs:
            resource["usage"] = kwargs["usage"]
        if "metadata" in kwargs:
            resource["metadata"] = kwargs["metadata"]
        if "tags" in kwargs:
            resource["tags"] = kwargs["tags"]

        return resource


def parse_model_id(model_id: str) -> Dict[str, str]:
    """
    Parse a model ID into provider, family, and version.

    Args:
        model_id: Model ID (e.g., "openai/gpt-4", "anthropic/claude-3-opus")

    Returns:
        Dictionary with provider, model_family, and model_version
    """
    parts = model_id.split("/")

    if len(parts) == 2:
        provider = parts[0]
        model_name = parts[1]

        # Try to split model name into family and version
        # Examples: "gpt-4-turbo" -> family: "gpt-4", version: "turbo"
        #           "claude-3-opus" -> family: "claude-3", version: "opus"
        name_parts = model_name.split("-")

        if len(name_parts) >= 2:
            # Heuristic: family is first 2 parts, version is the rest
            family = "-".join(name_parts[:2])
            version = "-".join(name_parts[2:]) if len(name_parts) > 2 else ""
        else:
            family = model_name
            version = ""

        return {
            "provider": provider,
            "model_family": family,
            "model_version": version or None,
        }

    return {
        "provider": "unknown",
        "model_family": model_id,
        "model_version": None,
    }


# Google Cloud Formatters

class GoogleCloudModelFormatter:
    """Formatter for Google Cloud Vertex AI models."""

    @staticmethod
    def format_vertex_model(
        model_id: str,
        name: str,
        resource_name: str,
        discovery_source: str,
        project_id: str,
        location: str,
        model_type: str = "vertex_ai_model",
        version: Optional[str] = None,
        description: Optional[str] = None,
        created_time: Optional[str] = None,
        updated_time: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format a Vertex AI model according to OpenCITE schema.

        Args:
            model_id: Unique model identifier
            name: Display name
            resource_name: Full GCP resource name
            discovery_source: Source of discovery (e.g., "google_cloud_api")
            project_id: GCP project ID
            location: GCP location/region
            model_type: Type of model
            version: Model version
            description: Model description
            created_time: Creation timestamp
            updated_time: Last update timestamp
            metadata: Additional metadata

        Returns:
            Formatted model dictionary
        """
        model = {
            "model_id": model_id,
            "name": name,
            "resource_name": resource_name,
            "discovery_source": discovery_source,
            "type": model_type,
            "provider": "google",
            "project_id": project_id,
            "location": location,
        }

        if version:
            model["version"] = version

        if description:
            model["description"] = description

        if created_time:
            model["created_time"] = created_time

        if updated_time:
            model["updated_time"] = updated_time

        if metadata:
            model["metadata"] = metadata

        return model


class GoogleCloudEndpointFormatter:
    """Formatter for Google Cloud Vertex AI endpoints."""

    @staticmethod
    def format_vertex_endpoint(
        endpoint_id: str,
        name: str,
        resource_name: str,
        discovery_source: str,
        project_id: str,
        location: str,
        deployed_models: Optional[List[Dict[str, Any]]] = None,
        description: Optional[str] = None,
        created_time: Optional[str] = None,
        updated_time: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format a Vertex AI endpoint according to OpenCITE schema.

        Args:
            endpoint_id: Unique endpoint identifier
            name: Display name
            resource_name: Full GCP resource name
            discovery_source: Source of discovery
            project_id: GCP project ID
            location: GCP location/region
            deployed_models: List of deployed models
            description: Endpoint description
            created_time: Creation timestamp
            updated_time: Last update timestamp
            metadata: Additional metadata

        Returns:
            Formatted endpoint dictionary
        """
        endpoint = {
            "endpoint_id": endpoint_id,
            "name": name,
            "resource_name": resource_name,
            "discovery_source": discovery_source,
            "type": "vertex_ai_endpoint",
            "provider": "google",
            "project_id": project_id,
            "location": location,
            "deployed_models": deployed_models or [],
        }

        if description:
            endpoint["description"] = description

        if created_time:
            endpoint["created_time"] = created_time

        if updated_time:
            endpoint["updated_time"] = updated_time

        if metadata:
            endpoint["metadata"] = metadata

        return endpoint


class GoogleCloudDeploymentFormatter:
    """Formatter for Google Cloud model deployments."""

    @staticmethod
    def format_deployment(
        deployment_id: str,
        endpoint_id: str,
        endpoint_name: str,
        model_id: Optional[str],
        deployed_model_id: str,
        deployed_model_name: Optional[str],
        discovery_source: str,
        project_id: str,
        location: str,
        machine_type: Optional[str] = None,
        min_replicas: Optional[int] = None,
        max_replicas: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format a model deployment according to OpenCITE schema.

        Args:
            deployment_id: Unique deployment identifier
            endpoint_id: Parent endpoint ID
            endpoint_name: Parent endpoint name
            model_id: Deployed model ID
            deployed_model_id: Deployment-specific model ID
            deployed_model_name: Deployment display name
            discovery_source: Source of discovery
            project_id: GCP project ID
            location: GCP location/region
            machine_type: Machine type for serving
            min_replicas: Minimum replica count
            max_replicas: Maximum replica count
            metadata: Additional metadata

        Returns:
            Formatted deployment dictionary
        """
        deployment = {
            "deployment_id": deployment_id,
            "endpoint_id": endpoint_id,
            "endpoint_name": endpoint_name,
            "deployed_model_id": deployed_model_id,
            "discovery_source": discovery_source,
            "type": "vertex_ai_deployment",
            "provider": "google",
            "project_id": project_id,
            "location": location,
        }

        if model_id:
            deployment["model_id"] = model_id

        if deployed_model_name:
            deployment["deployed_model_name"] = deployed_model_name

        if machine_type:
            deployment["machine_type"] = machine_type

        if min_replicas is not None:
            deployment["min_replicas"] = min_replicas

        if max_replicas is not None:
            deployment["max_replicas"] = max_replicas

        if metadata:
            deployment["metadata"] = metadata

        return deployment


class GoogleCloudGenerativeModelFormatter:
    """Formatter for Google Cloud generative AI models."""

    @staticmethod
    def format_generative_model(
        model_id: str,
        name: str,
        discovery_source: str,
        provider: str,
        model_family: str,
        project_id: str,
        location: str,
        capabilities: Optional[List[str]] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format a generative AI model according to OpenCITE schema.

        Args:
            model_id: Model identifier (e.g., "gemini-pro")
            name: Display name
            discovery_source: Source of discovery
            provider: Model provider (e.g., "google")
            model_family: Model family (e.g., "gemini", "palm")
            project_id: GCP project ID
            location: GCP location/region
            capabilities: List of model capabilities
            description: Model description
            metadata: Additional metadata

        Returns:
            Formatted generative model dictionary
        """
        model = {
            "model_id": model_id,
            "name": name,
            "discovery_source": discovery_source,
            "type": "generative_ai_model",
            "provider": provider,
            "model_family": model_family,
            "project_id": project_id,
            "location": location,
        }

        if capabilities:
            model["capabilities"] = capabilities

        if description:
            model["description"] = description

        if metadata:
            model["metadata"] = metadata

        return model
