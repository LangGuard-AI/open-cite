"""
AWS Bedrock AgentCore discovery plugin for OpenCITE.

This plugin discovers AI assets deployed via AWS Bedrock AgentCore:
- Agent runtimes (deployed agent instances)
- Memory stores (agent memory resources)
- Gateways (API gateways / MCP tool configurations)

It also correlates discovered runtimes with live OTel traces received
by the OpenTelemetry plugin, so you see both *what's deployed* and
*what those agents are actually doing* (models called, tools used,
token consumption, downstream systems contacted).

Uses the boto3 ``bedrock-agentcore`` control-plane service.
"""

import logging
from typing import Any, Dict, List, Optional, Set
import threading

from open_cite.core import BaseDiscoveryPlugin
from open_cite.plugins.aws.base import AWSClientMixin

logger = logging.getLogger(__name__)


class AWSAgentCorePlugin(AWSClientMixin, BaseDiscoveryPlugin):
    """
    AWS Bedrock AgentCore discovery plugin.

    Discovers:
    - Agent Runtimes: Deployed agent instances running in AgentCore
    - Memory: Memory stores attached to agents
    - Gateways: API gateways and MCP tool configurations

    When the OpenTelemetry plugin is also running, the AgentCore plugin
    correlates incoming OTel traces with discovered runtimes.  Each
    runtime is enriched with:
    - ``otel_correlated``: True if any matching traces were found
    - ``otel_agents``: Agents discovered via OTel that match this runtime
    - ``models_used``: All models called by this agent (from traces)
    - ``tools_used``: All tools invoked by this agent (from traces)
    - ``token_usage``: Aggregated input/output token counts
    - ``last_trace_seen``: Timestamp of the most recent trace

    Falls back gracefully if the ``bedrock-agentcore`` service is not
    available in the user's boto3 version or AWS region.
    """

    plugin_type = "aws_agentcore"

    def __init__(
        self,
        region: Optional[str] = None,
        profile: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        role_arn: Optional[str] = None,
        instance_id: Optional[str] = None,
        display_name: Optional[str] = None,
    ):
        """
        Initialize the AWS AgentCore plugin.

        Args:
            region: AWS region (default: us-east-1)
            profile: AWS profile name
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
            session_token: AWS session token (for temporary credentials)
            role_arn: IAM role ARN to assume
            instance_id: Unique identifier for this plugin instance
            display_name: Human-readable name for this instance
        """
        AWSClientMixin.__init__(
            self,
            region=region,
            profile=profile,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            role_arn=role_arn,
        )
        BaseDiscoveryPlugin.__init__(
            self,
            instance_id=instance_id,
            display_name=display_name,
        )

        # Caches
        self._runtimes_cache: Dict[str, Dict[str, Any]] = {}
        self._memories_cache: Dict[str, Dict[str, Any]] = {}
        self._gateways_cache: Dict[str, Dict[str, Any]] = {}

        # OTel correlation
        self._otel_plugin = None

        self._lock = threading.Lock()

    # ── OTel correlation ────────────────────────────────────────────────

    def set_otel_plugin(self, otel_plugin) -> None:
        """
        Connect to the OpenTelemetry plugin for trace correlation.

        Called automatically by OpenCiteClient.register_plugin() when
        both the AgentCore and OpenTelemetry plugins are registered.

        Args:
            otel_plugin: An OpenTelemetryPlugin instance
        """
        self._otel_plugin = otel_plugin
        logger.info(
            "AgentCore plugin linked to OpenTelemetry plugin — "
            "runtimes will be enriched with live trace data"
        )

    def _correlate_runtime_traces(self, runtime_info: Dict[str, Any]) -> None:
        """
        Enrich a runtime dict with correlated OTel trace data.

        Matching strategy (any match counts):
        1. OTel discovered_agents whose name contains the runtime name
        2. OTel discovered_tools whose service_name contains the runtime name
        3. OTel discovered_agents whose name contains the runtime_id

        When matched, the runtime gets extra keys showing live behaviour.
        """
        if not self._otel_plugin:
            return

        runtime_name = (runtime_info.get("name") or "").lower()
        runtime_id = (runtime_info.get("runtime_id") or "").lower()

        if not runtime_name and not runtime_id:
            return

        # ── Match agents ────────────────────────────────────────────
        matched_agents: List[Dict[str, Any]] = []
        all_models: Set[str] = set()
        all_tools: Set[str] = set()
        latest_seen: Optional[str] = None

        agents = getattr(self._otel_plugin, "discovered_agents", {})
        for agent_id, agent_data in agents.items():
            agent_lower = agent_id.lower()
            if (
                (runtime_name and runtime_name in agent_lower)
                or (runtime_name and agent_lower in runtime_name)
                or (runtime_id and runtime_id in agent_lower)
            ):
                tools = list(agent_data.get("tools_used", set()))
                models = list(agent_data.get("models_used", set()))
                last_seen = agent_data.get("last_seen")

                matched_agents.append({
                    "agent_id": agent_id,
                    "confidence": agent_data.get("confidence", "low"),
                    "tools_used": tools,
                    "models_used": models,
                    "first_seen": agent_data.get("first_seen"),
                    "last_seen": last_seen,
                })
                all_models.update(models)
                all_tools.update(tools)

                if last_seen and (not latest_seen or last_seen > latest_seen):
                    latest_seen = last_seen

        # ── Match tools by service name ─────────────────────────────
        matched_tool_names: List[str] = []
        tools_dict = getattr(self._otel_plugin, "discovered_tools", {})
        for tool_name, tool_data in tools_dict.items():
            svc = (
                tool_data.get("metadata", {}).get("service_name", "")
            ).lower()
            tool_lower = tool_name.lower()
            if (
                (runtime_name and runtime_name in svc)
                or (runtime_name and runtime_name in tool_lower)
                or (runtime_id and runtime_id in svc)
            ):
                matched_tool_names.append(tool_name)
                all_models.update(tool_data.get("models", set()))

        # ── Token usage for matched models ──────────────────────────
        token_usage: Dict[str, Dict[str, int]] = {}
        model_tokens = getattr(self._otel_plugin, "model_token_usage", {})
        for model_name in all_models:
            if model_name in model_tokens:
                token_usage[model_name] = dict(model_tokens[model_name])

        # ── Write enrichment ────────────────────────────────────────
        if matched_agents or matched_tool_names:
            runtime_info["otel_correlated"] = True
            runtime_info["otel_agents"] = matched_agents
            runtime_info["otel_matched_tools"] = matched_tool_names
            runtime_info["models_used"] = sorted(all_models)
            runtime_info["tools_used"] = sorted(all_tools)
            runtime_info["token_usage"] = token_usage
            runtime_info["last_trace_seen"] = latest_seen
        else:
            runtime_info["otel_correlated"] = False

    # ── Plugin interface ────────────────────────────────────────────────

    @property
    def supported_asset_types(self) -> Set[str]:
        return {"agent_runtime", "memory", "gateway"}

    def get_identification_attributes(self) -> List[str]:
        return [
            "aws.agentcore.runtime_id",
            "aws.agentcore.region",
            "aws.account_id",
        ]

    def get_config(self) -> Dict[str, Any]:
        """Return plugin configuration (sensitive values masked)."""
        return {
            "region": self.region,
            "profile": self.profile,
            "access_key_id": "****" if self.access_key_id else None,
            "otel_linked": self._otel_plugin is not None,
        }

    @classmethod
    def plugin_metadata(cls) -> Dict[str, Any]:
        return {
            "name": "AWS Bedrock AgentCore",
            "description": (
                "Discovers agent runtimes, memory stores, and gateways "
                "in AWS Bedrock AgentCore. When the OpenTelemetry receiver "
                "is also running, correlates live traces with deployed runtimes."
            ),
            "required_fields": {
                "region": {
                    "label": "AWS Region",
                    "default": "us-east-1",
                    "required": False,
                },
                "profile": {
                    "label": "AWS Profile",
                    "default": "",
                    "required": False,
                },
                "access_key_id": {
                    "label": "Access Key ID",
                    "default": "",
                    "required": False,
                },
                "secret_access_key": {
                    "label": "Secret Access Key",
                    "default": "",
                    "required": False,
                    "type": "password",
                },
                "role_arn": {
                    "label": "Role ARN (optional)",
                    "default": "",
                    "required": False,
                },
            },
            "env_vars": [
                "AWS_REGION",
                "AWS_PROFILE",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
            ],
        }

    @classmethod
    def from_config(cls, config, instance_id=None, display_name=None, dependencies=None):
        return cls(
            region=config.get("region"),
            profile=config.get("profile"),
            access_key_id=config.get("access_key_id"),
            secret_access_key=config.get("secret_access_key"),
            session_token=config.get("session_token"),
            role_arn=config.get("role_arn"),
            instance_id=instance_id,
            display_name=display_name,
        )

    def verify_connection(self) -> Dict[str, Any]:
        """
        Verify connection to AWS Bedrock AgentCore.

        Returns:
            Dict with connection status and details
        """
        try:
            client = self._get_client("bedrock-agentcore-control")
            # Lightweight call to validate credentials + service availability
            response = client.list_agent_runtimes(maxResults=1)
            account_id = self.get_account_id()

            result = {
                "success": True,
                "region": self.region,
                "account_id": account_id,
                "message": "Successfully connected to AWS Bedrock AgentCore",
                "otel_linked": self._otel_plugin is not None,
            }

            if self._otel_plugin:
                result["otel_receiver_status"] = (
                    "running" if self._otel_plugin.status == "running" else "stopped"
                )

            return result

        except Exception as e:
            return {
                "success": False,
                "region": self.region,
                "error": str(e),
                "message": "Failed to connect to AWS Bedrock AgentCore",
            }

    def list_assets(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        List AWS Bedrock AgentCore assets.

        Supported asset types:
        - "agent_runtime": Deployed agent runtimes (enriched with OTel data)
        - "memory": Memory stores
        - "gateway": API gateways / MCP tool configs

        Args:
            asset_type: Type of asset to list
            **kwargs: Additional filters

        Returns:
            List of assets
        """
        with self._lock:
            if asset_type == "agent_runtime":
                return self._list_agent_runtimes(**kwargs)
            elif asset_type == "memory":
                return self._list_memories(**kwargs)
            elif asset_type == "gateway":
                return self._list_gateways(**kwargs)
            else:
                raise ValueError(
                    f"Unsupported asset type: {asset_type}. "
                    f"Supported types: agent_runtime, memory, gateway"
                )

    # ── Agent Runtimes ──────────────────────────────────────────────────

    def _list_agent_runtimes(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List deployed agent runtimes.

        Calls ``list_agent_runtimes()`` then ``get_agent_runtime()`` for
        each runtime to get full details.  If the OTel plugin is linked,
        each runtime is enriched with correlated trace data.

        Returns:
            List of agent runtime dicts
        """
        try:
            client = self._get_client("bedrock-agentcore-control")

            runtimes: List[Dict[str, Any]] = []
            paginator_args: Dict[str, Any] = {}
            next_token: Optional[str] = None

            while True:
                if next_token:
                    paginator_args["nextToken"] = next_token

                response = client.list_agent_runtimes(**paginator_args)

                for summary in response.get("agentRuntimes", []):
                    runtime_id = summary.get("agentRuntimeId", "")
                    runtime_info = self._build_runtime_summary(summary)

                    # Fetch full details
                    try:
                        detail = client.get_agent_runtime(
                            agentRuntimeId=runtime_id
                        )
                        runtime_info.update(
                            self._enrich_runtime_detail(detail)
                        )
                    except Exception as e:
                        logger.debug(
                            f"Could not get details for runtime {runtime_id}: {e}"
                        )

                    # Correlate with OTel traces
                    self._correlate_runtime_traces(runtime_info)

                    self._runtimes_cache[runtime_id] = runtime_info
                    runtimes.append(runtime_info)

                next_token = response.get("nextToken")
                if not next_token:
                    break

            logger.info(f"Discovered {len(runtimes)} AgentCore agent runtimes")
            return runtimes

        except Exception as e:
            logger.warning(f"Failed to list agent runtimes: {e}")
            return []

    def _build_runtime_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Build a normalised runtime dict from a list-API summary."""
        runtime_id = summary.get("agentRuntimeId", "")
        return {
            "id": summary.get("agentRuntimeArn", runtime_id),
            "name": summary.get("agentRuntimeName", runtime_id),
            "runtime_id": runtime_id,
            "type": "agent_runtime",
            "discovery_source": "aws_agentcore_api",
            "status": summary.get("status"),
            "version": summary.get("agentRuntimeVersion"),
            "description": summary.get("description"),
            "last_modified": (
                summary["lastUpdatedAt"].isoformat()
                if summary.get("lastUpdatedAt")
                else None
            ),
            "region": self.region,
            "metadata": {},
        }

    @staticmethod
    def _enrich_runtime_detail(detail: Dict[str, Any]) -> Dict[str, Any]:
        """Extract extra fields from a get_agent_runtime response."""
        extras: Dict[str, Any] = {}

        if detail.get("description"):
            extras["description"] = detail["description"]
        if detail.get("agentRuntimeVersion"):
            extras["version"] = detail["agentRuntimeVersion"]
        if detail.get("createdAt"):
            extras["creation_time"] = detail["createdAt"].isoformat()
        if detail.get("roleArn"):
            extras.setdefault("metadata", {})["role_arn"] = detail["roleArn"]
        if detail.get("failureReason"):
            extras.setdefault("metadata", {})["failure_reason"] = detail["failureReason"]

        # Network / environment config
        network = detail.get("networkConfiguration")
        if network:
            extras.setdefault("metadata", {})["network_configuration"] = network

        env_vars = detail.get("environmentVariables")
        if env_vars:
            extras.setdefault("metadata", {})["environment_variables"] = dict(env_vars)

        return extras

    # ── Memory Stores ───────────────────────────────────────────────────

    def _list_memories(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List memory stores.

        Returns:
            List of memory resource dicts
        """
        try:
            client = self._get_client("bedrock-agentcore-control")

            memories: List[Dict[str, Any]] = []
            next_token: Optional[str] = None

            while True:
                call_kwargs: Dict[str, Any] = {}
                if next_token:
                    call_kwargs["nextToken"] = next_token

                response = client.list_memories(**call_kwargs)

                for mem in response.get("memories", []):
                    memory_id = mem.get("id", "")
                    memory_info = {
                        "id": mem.get("arn", memory_id),
                        "name": memory_id,
                        "memory_id": memory_id,
                        "type": "memory",
                        "discovery_source": "aws_agentcore_api",
                        "status": mem.get("status"),
                        "creation_time": (
                            mem["createdAt"].isoformat()
                            if mem.get("createdAt")
                            else None
                        ),
                        "last_modified": (
                            mem["updatedAt"].isoformat()
                            if mem.get("updatedAt")
                            else None
                        ),
                        "region": self.region,
                        "metadata": {},
                    }

                    self._memories_cache[memory_id] = memory_info
                    memories.append(memory_info)

                next_token = response.get("nextToken")
                if not next_token:
                    break

            logger.info(f"Discovered {len(memories)} AgentCore memory stores")
            return memories

        except Exception as e:
            logger.warning(f"Failed to list memories: {e}")
            return []

    # ── Gateways ────────────────────────────────────────────────────────

    def _list_gateways(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List gateways and their targets (MCP tool configs).

        Returns:
            List of gateway dicts, each with a ``targets`` sub-list
        """
        try:
            client = self._get_client("bedrock-agentcore-control")

            gateways: List[Dict[str, Any]] = []
            next_token: Optional[str] = None

            while True:
                call_kwargs: Dict[str, Any] = {}
                if next_token:
                    call_kwargs["nextToken"] = next_token

                response = client.list_gateways(**call_kwargs)

                for gw in response.get("items", []):
                    gateway_id = gw.get("gatewayId", "")
                    gateway_info = {
                        "id": gateway_id,
                        "name": gw.get("name", gateway_id),
                        "gateway_id": gateway_id,
                        "type": "gateway",
                        "discovery_source": "aws_agentcore_api",
                        "status": gw.get("status"),
                        "protocol_type": gw.get("protocolType"),
                        "authorizer_type": gw.get("authorizerType"),
                        "creation_time": (
                            gw["createdAt"].isoformat()
                            if gw.get("createdAt")
                            else None
                        ),
                        "last_modified": (
                            gw["updatedAt"].isoformat()
                            if gw.get("updatedAt")
                            else None
                        ),
                        "region": self.region,
                        "metadata": {},
                        "targets": [],
                    }

                    if gw.get("description"):
                        gateway_info["description"] = gw["description"]

                    # Fetch gateway targets (MCP tool / API configs)
                    gateway_info["targets"] = self._list_gateway_targets(
                        client, gateway_id
                    )

                    self._gateways_cache[gateway_id] = gateway_info
                    gateways.append(gateway_info)

                next_token = response.get("nextToken")
                if not next_token:
                    break

            logger.info(f"Discovered {len(gateways)} AgentCore gateways")
            return gateways

        except Exception as e:
            logger.warning(f"Failed to list gateways: {e}")
            return []

    @staticmethod
    def _list_gateway_targets(
        client, gateway_id: str
    ) -> List[Dict[str, Any]]:
        """
        List targets attached to a gateway.

        Args:
            client: boto3 bedrock-agentcore client
            gateway_id: The gateway to query

        Returns:
            List of target dicts
        """
        targets: List[Dict[str, Any]] = []

        try:
            next_token: Optional[str] = None
            while True:
                call_kwargs: Dict[str, Any] = {"gatewayId": gateway_id}
                if next_token:
                    call_kwargs["nextToken"] = next_token

                response = client.list_gateway_targets(**call_kwargs)

                for tgt in response.get("items", []):
                    target_info = {
                        "target_id": tgt.get("targetId"),
                        "name": tgt.get("name", tgt.get("targetId", "")),
                        "status": tgt.get("status"),
                        "creation_time": (
                            tgt["createdAt"].isoformat()
                            if tgt.get("createdAt")
                            else None
                        ),
                        "last_modified": (
                            tgt["updatedAt"].isoformat()
                            if tgt.get("updatedAt")
                            else None
                        ),
                    }
                    if tgt.get("description"):
                        target_info["description"] = tgt["description"]

                    targets.append(target_info)

                next_token = response.get("nextToken")
                if not next_token:
                    break

        except Exception as e:
            logger.debug(
                f"Could not list targets for gateway {gateway_id}: {e}"
            )

        return targets

    # ── Refresh ─────────────────────────────────────────────────────────

    def refresh_discovery(self):
        """Refresh all cached discovery data."""
        with self._lock:
            logger.info("Refreshing AWS AgentCore discovery...")
            self._runtimes_cache.clear()
            self._memories_cache.clear()
            self._gateways_cache.clear()

            self._list_agent_runtimes()
            self._list_memories()
            self._list_gateways()

            logger.info("AWS AgentCore discovery refresh complete")
