"""
Splunk discovery plugin for Open-CITE.

Discovers AI assets by querying Splunk for network requests to known AI/ML
provider endpoints. Uses the Splunk Common Information Model (CIM) to query
both application-level web traffic (Web data model) and network-level traffic
(Network Traffic data model).

Detected AI providers include OpenAI, Anthropic, Google AI, Azure OpenAI,
AWS Bedrock, Cohere, HuggingFace, Mistral, OpenRouter, and others. The
plugin also detects MCP (Model Context Protocol) server traffic.
"""

import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import requests

from open_cite.core import BaseDiscoveryPlugin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known AI provider endpoint patterns
# ---------------------------------------------------------------------------

# Each entry: (compiled_regex, provider_name, service_label)
_AI_ENDPOINT_PATTERNS = [
    # OpenAI
    (re.compile(r"api\.openai\.com"), "openai", "OpenAI API"),
    # Azure OpenAI
    (re.compile(r"[\w-]+\.openai\.azure\.com"), "azure_openai", "Azure OpenAI"),
    # Anthropic
    (re.compile(r"api\.anthropic\.com"), "anthropic", "Anthropic API"),
    # Google AI / Vertex AI
    (re.compile(r"generativelanguage\.googleapis\.com"), "google", "Google AI (Gemini)"),
    (re.compile(r"[\w-]+-aiplatform\.googleapis\.com"), "google", "Vertex AI"),
    # AWS Bedrock
    (re.compile(r"bedrock-runtime\.[\w-]+\.amazonaws\.com"), "aws_bedrock", "AWS Bedrock"),
    (re.compile(r"bedrock\.[\w-]+\.amazonaws\.com"), "aws_bedrock", "AWS Bedrock"),
    # Cohere
    (re.compile(r"api\.cohere\.ai"), "cohere", "Cohere API"),
    (re.compile(r"api\.cohere\.com"), "cohere", "Cohere API"),
    # HuggingFace
    (re.compile(r"api-inference\.huggingface\.co"), "huggingface", "HuggingFace Inference"),
    (re.compile(r"[\w-]+\.endpoints\.huggingface\.cloud"), "huggingface", "HuggingFace Endpoints"),
    # Mistral
    (re.compile(r"api\.mistral\.ai"), "mistral", "Mistral AI API"),
    # OpenRouter
    (re.compile(r"openrouter\.ai"), "openrouter", "OpenRouter"),
    # Replicate
    (re.compile(r"api\.replicate\.com"), "replicate", "Replicate API"),
    # Together AI
    (re.compile(r"api\.together\.xyz"), "together", "Together AI"),
    # Perplexity
    (re.compile(r"api\.perplexity\.ai"), "perplexity", "Perplexity AI"),
    # Groq
    (re.compile(r"api\.groq\.com"), "groq", "Groq API"),
    # Fireworks AI
    (re.compile(r"api\.fireworks\.ai"), "fireworks", "Fireworks AI"),
    # DeepSeek
    (re.compile(r"api\.deepseek\.com"), "deepseek", "DeepSeek API"),
    # Stability AI
    (re.compile(r"api\.stability\.ai"), "stability", "Stability AI"),
]

# URL path patterns that hint at model identity
_MODEL_PATH_PATTERNS = [
    # OpenAI / Azure: /v1/chat/completions, /v1/completions, /v1/embeddings
    re.compile(r"/v1/(chat/completions|completions|embeddings|images|audio|moderations)"),
    # Anthropic: /v1/messages, /v1/complete
    re.compile(r"/v1/(messages|complete)"),
    # Bedrock: /model/<model-id>/invoke
    re.compile(r"/model/([\w.-]+)/invoke"),
    # HuggingFace: /models/<model-name>
    re.compile(r"/models/([\w./-]+)"),
]

# MCP (Model Context Protocol) detection patterns
_MCP_INDICATORS = [
    re.compile(r"jsonrpc", re.IGNORECASE),
    re.compile(r"mcp-session-id", re.IGNORECASE),
    re.compile(r"\"method\"\s*:\s*\"(tools/list|tools/call|resources/list|prompts/list)\""),
]


# ---------------------------------------------------------------------------
# Splunk SPL queries using the CIM
# ---------------------------------------------------------------------------

# Web data model (application firewall / proxy logs)
# CIM fields: url, http_method, status, src, dest, http_user_agent,
#              bytes_in, bytes_out, action, app, user
_SPL_WEB_AI_TRAFFIC = '''
| tstats summariesonly=t count as request_count
    sum(Web.bytes_in) as total_bytes_in
    sum(Web.bytes_out) as total_bytes_out
    dc(Web.src) as unique_sources
    values(Web.http_method) as http_methods
    latest(Web._time) as last_seen
    earliest(Web._time) as first_seen
  from datamodel=Web.Web
  where ({dest_filter})
  by Web.url Web.dest Web.action Web.app Web.user Web.http_user_agent Web.status
| rename Web.* as *
| eval dest_port=if(isnull(dest_port), if(match(url, "^https"), 443, 80), dest_port)
'''

# Network Traffic data model (firewall / IDS / flow logs)
# CIM fields: src, dest, dest_port, transport, bytes, bytes_in, bytes_out,
#              packets, action, app, protocol_version
_SPL_NETWORK_AI_TRAFFIC = '''
| tstats summariesonly=t count as connection_count
    sum(All_Traffic.bytes) as total_bytes
    sum(All_Traffic.bytes_in) as total_bytes_in
    sum(All_Traffic.bytes_out) as total_bytes_out
    dc(All_Traffic.src) as unique_sources
    values(All_Traffic.transport) as transport
    latest(All_Traffic._time) as last_seen
    earliest(All_Traffic._time) as first_seen
  from datamodel=Network_Traffic.All_Traffic
  where ({dest_filter})
  by All_Traffic.dest All_Traffic.dest_port All_Traffic.action All_Traffic.app
| rename All_Traffic.* as *
'''

# Raw search fallback (when CIM accelerated data models are not available)
_SPL_RAW_WEB_SEARCH = '''
index={index} sourcetype IN ({sourcetypes})
  ({dest_filter})
| stats count as request_count
    sum(bytes_in) as total_bytes_in
    sum(bytes_out) as total_bytes_out
    dc(src) as unique_sources
    values(http_method) as http_methods
    latest(_time) as last_seen
    earliest(_time) as first_seen
  by url dest action app user http_user_agent status
'''

# Query to detect MCP traffic in request/response bodies
_SPL_MCP_DETECTION = '''
| tstats summariesonly=t count as request_count
    dc(Web.src) as unique_sources
    latest(Web._time) as last_seen
    earliest(Web._time) as first_seen
  from datamodel=Web.Web
  where (Web.http_content_type="application/json")
  by Web.url Web.dest Web.src Web.http_method Web.user
| rename Web.* as *
| search url="*jsonrpc*" OR url="*mcp*"
'''


def _build_dest_filter(field_prefix: str = "Web") -> str:
    """Build the WHERE clause matching AI provider destinations."""
    clauses = []
    # Use wildcard-friendly patterns for tstats
    patterns = [
        "api.openai.com",
        "*.openai.azure.com",
        "api.anthropic.com",
        "generativelanguage.googleapis.com",
        "*-aiplatform.googleapis.com",
        "bedrock-runtime.*.amazonaws.com",
        "bedrock.*.amazonaws.com",
        "api.cohere.ai",
        "api.cohere.com",
        "api-inference.huggingface.co",
        "*.endpoints.huggingface.cloud",
        "api.mistral.ai",
        "openrouter.ai",
        "api.replicate.com",
        "api.together.xyz",
        "api.perplexity.ai",
        "api.groq.com",
        "api.fireworks.ai",
        "api.deepseek.com",
        "api.stability.ai",
    ]
    for p in patterns:
        clauses.append(f'{field_prefix}.dest="{p}"')
    return " OR ".join(clauses)


class SplunkPlugin(BaseDiscoveryPlugin):
    """
    Splunk discovery plugin for Open-CITE.

    Discovers AI assets by querying Splunk's CIM-based data models for
    network traffic to known AI/ML provider endpoints. Supports:

    - **Web data model**: HTTP-level visibility from web proxies, WAFs,
      and application firewalls. Provides URL paths, user agents, HTTP
      methods, and response codes.
    - **Network Traffic data model**: Connection-level visibility from
      firewalls, flow logs, and IDS/IPS. Provides IP-level traffic
      stats and connection metadata.
    - **MCP detection**: Identifies Model Context Protocol traffic
      patterns in web logs.
    """

    plugin_type = "splunk"

    # -----------------------------------------------------------------------
    # Plugin metadata & factory
    # -----------------------------------------------------------------------

    @classmethod
    def plugin_metadata(cls) -> Dict[str, Any]:
        return {
            "name": "Splunk",
            "description": (
                "Discovers AI assets from network traffic using Splunk CIM "
                "(Web and Network Traffic data models)"
            ),
            "required_fields": {
                "splunk_url": {
                    "label": "Splunk URL",
                    "default": "https://localhost:8089",
                    "required": True,
                    "type": "text",
                },
                "token": {
                    "label": "Bearer Token (or Session Key)",
                    "default": "",
                    "required": False,
                    "type": "password",
                },
                "username": {
                    "label": "Username",
                    "default": "",
                    "required": False,
                },
                "password": {
                    "label": "Password",
                    "default": "",
                    "required": False,
                    "type": "password",
                },
                "verify_ssl": {
                    "label": "Verify SSL",
                    "default": "true",
                    "required": False,
                },
                "time_range": {
                    "label": "Search time range",
                    "default": "-24h",
                    "required": False,
                },
                "use_raw_search": {
                    "label": "Use raw search (if CIM not accelerated)",
                    "default": "false",
                    "required": False,
                },
                "raw_index": {
                    "label": "Index for raw search",
                    "default": "main",
                    "required": False,
                },
                "raw_sourcetypes": {
                    "label": "Source types for raw search (comma-separated)",
                    "default": "proxy, web_proxy, bluecoat, paloalto, zscaler",
                    "required": False,
                },
            },
            "env_vars": [
                "SPLUNK_URL",
                "SPLUNK_TOKEN",
                "SPLUNK_USERNAME",
                "SPLUNK_PASSWORD",
            ],
        }

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        instance_id: Optional[str] = None,
        display_name: Optional[str] = None,
        dependencies: Optional[Dict[str, Any]] = None,
    ) -> "SplunkPlugin":
        dependencies = dependencies or {}
        return cls(
            splunk_url=config.get("splunk_url", "https://localhost:8089"),
            token=config.get("token"),
            username=config.get("username"),
            password=config.get("password"),
            verify_ssl=config.get("verify_ssl", "true"),
            time_range=config.get("time_range", "-24h"),
            use_raw_search=config.get("use_raw_search", "false"),
            raw_index=config.get("raw_index", "main"),
            raw_sourcetypes=config.get("raw_sourcetypes", "proxy, web_proxy"),
            http_client=dependencies.get("http_client"),
            instance_id=instance_id,
            display_name=display_name,
        )

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def __init__(
        self,
        splunk_url: str = "https://localhost:8089",
        token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        verify_ssl: str = "true",
        time_range: str = "-24h",
        use_raw_search: str = "false",
        raw_index: str = "main",
        raw_sourcetypes: str = "proxy, web_proxy",
        http_client: Any = None,
        instance_id: Optional[str] = None,
        display_name: Optional[str] = None,
    ):
        super().__init__(instance_id=instance_id, display_name=display_name)
        self.splunk_url = splunk_url.rstrip("/")
        self.token = token
        self.username = username
        self.password = password
        self.verify_ssl = verify_ssl.lower() in ("true", "1", "yes") if isinstance(verify_ssl, str) else bool(verify_ssl)
        self.time_range = time_range
        self.use_raw_search = use_raw_search.lower() in ("true", "1", "yes") if isinstance(use_raw_search, str) else bool(use_raw_search)
        self.raw_index = raw_index
        self.raw_sourcetypes = raw_sourcetypes
        self.http_client = http_client or requests
        self._session_key: Optional[str] = None

        # Discovered assets
        self.discovered_tools: Dict[str, Dict[str, Any]] = {}
        self.discovered_models: Dict[str, Dict[str, Any]] = {}
        self.discovered_endpoints: Dict[str, Dict[str, Any]] = {}
        self.mcp_servers: Dict[str, Dict[str, Any]] = {}

    # -----------------------------------------------------------------------
    # BaseDiscoveryPlugin interface
    # -----------------------------------------------------------------------

    @property
    def supported_asset_types(self) -> Set[str]:
        return {"tool", "model", "endpoint"}

    def get_identification_attributes(self) -> List[str]:
        return [
            "splunk.dest",
            "splunk.url",
            "splunk.provider",
            "splunk.service",
        ]

    def get_config(self) -> Dict[str, Any]:
        return {
            "splunk_url": self.splunk_url,
            "username": self.username,
            "token": "****" if self.token else None,
            "verify_ssl": self.verify_ssl,
            "time_range": self.time_range,
            "use_raw_search": self.use_raw_search,
        }

    def verify_connection(self) -> Dict[str, Any]:
        try:
            self._authenticate()
            # Try a lightweight server info call
            resp = self._splunk_get("/services/server/info", output_mode="json")
            if resp.get("entry"):
                server_name = resp["entry"][0].get("content", {}).get("serverName", "unknown")
                version = resp["entry"][0].get("content", {}).get("version", "unknown")
                return {
                    "success": True,
                    "server_name": server_name,
                    "version": version,
                    "splunk_url": self.splunk_url,
                }
            return {"success": True, "splunk_url": self.splunk_url}
        except Exception as e:
            logger.error("Splunk connection verification failed: %s", e)
            return {"success": False, "error": str(e)}

    def list_assets(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        if asset_type == "tool":
            return list(self.discovered_tools.values())
        if asset_type == "model":
            return list(self.discovered_models.values())
        if asset_type == "endpoint":
            return list(self.discovered_endpoints.values())
        return []

    def export_assets(self) -> Dict[str, Any]:
        return {
            "tools": list(self.discovered_tools.values()),
            "models": list(self.discovered_models.values()),
        }

    def start(self):
        self._status = "running"
        try:
            self.discover()
        except Exception as e:
            logger.error("Splunk discovery failed on start: %s", e)
            self._status = "error"
            return
        logger.info("Started Splunk plugin %s", self.instance_id)

    def stop(self):
        super().stop()

    # -----------------------------------------------------------------------
    # Authentication
    # -----------------------------------------------------------------------

    def _authenticate(self):
        """Authenticate to the Splunk REST API.

        Supports bearer token auth (preferred) or username/password session
        auth. Sets ``self._session_key`` for subsequent requests.
        """
        if self.token:
            self._session_key = self.token
            return

        if self._session_key:
            return

        if not self.username or not self.password:
            raise ValueError(
                "Splunk credentials required: provide either a token or "
                "username/password."
            )

        url = f"{self.splunk_url}/services/auth/login"
        resp = self.http_client.post(
            url,
            data={"username": self.username, "password": self.password, "output_mode": "json"},
            verify=self.verify_ssl,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._session_key = data.get("sessionKey")
        if not self._session_key:
            raise ValueError("Failed to obtain Splunk session key")
        logger.info("Authenticated to Splunk as %s", self.username)

    def _auth_headers(self) -> Dict[str, str]:
        """Return authorization headers for Splunk REST calls."""
        if not self._session_key:
            self._authenticate()

        # Bearer token (for token-based auth including Splunk Cloud HEC tokens)
        if self.token:
            return {"Authorization": f"Bearer {self._session_key}"}
        # Session key from username/password auth
        return {"Authorization": f"Splunk {self._session_key}"}

    # -----------------------------------------------------------------------
    # Splunk REST helpers
    # -----------------------------------------------------------------------

    def _splunk_get(self, path: str, **params) -> Dict[str, Any]:
        """Issue a GET to the Splunk REST API and return parsed JSON."""
        url = f"{self.splunk_url}{path}"
        params.setdefault("output_mode", "json")
        resp = self.http_client.get(
            url,
            headers=self._auth_headers(),
            params=params,
            verify=self.verify_ssl,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _splunk_post(self, path: str, data: Optional[Dict] = None, **params) -> Dict[str, Any]:
        """Issue a POST to the Splunk REST API and return parsed JSON."""
        url = f"{self.splunk_url}{path}"
        params.setdefault("output_mode", "json")
        resp = self.http_client.post(
            url,
            headers=self._auth_headers(),
            data=data,
            params=params,
            verify=self.verify_ssl,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _run_search(self, spl: str, time_range: Optional[str] = None) -> List[Dict[str, Any]]:
        """Submit a oneshot search job and return results.

        Args:
            spl: The SPL query string.
            time_range: Earliest time modifier (e.g. ``"-24h"``).

        Returns:
            List of result dicts from the search.
        """
        self._authenticate()
        earliest = time_range or self.time_range

        data = {
            "search": f"search {spl}" if not spl.lstrip().startswith("|") else spl,
            "earliest_time": earliest,
            "latest_time": "now",
            "output_mode": "json",
            "exec_mode": "oneshot",
            "count": 10000,
        }

        url = f"{self.splunk_url}/services/search/jobs"
        resp = self.http_client.post(
            url,
            headers=self._auth_headers(),
            data=data,
            verify=self.verify_ssl,
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
        return result.get("results", [])

    # -----------------------------------------------------------------------
    # Discovery orchestration
    # -----------------------------------------------------------------------

    def discover(self):
        """Run all discovery queries and populate asset dictionaries."""
        self._authenticate()

        # 1. Web data model (application-level traffic)
        try:
            web_results = self._discover_web_traffic()
            self._process_web_results(web_results)
        except Exception as e:
            logger.warning("Web data model query failed: %s", e)

        # 2. Network Traffic data model (connection-level)
        try:
            net_results = self._discover_network_traffic()
            self._process_network_results(net_results)
        except Exception as e:
            logger.warning("Network Traffic data model query failed: %s", e)

        # 3. MCP detection
        try:
            mcp_results = self._discover_mcp_traffic()
            self._process_mcp_results(mcp_results)
        except Exception as e:
            logger.warning("MCP detection query failed: %s", e)

        logger.info(
            "Splunk discovery complete: %d tools, %d models, %d endpoints",
            len(self.discovered_tools),
            len(self.discovered_models),
            len(self.discovered_endpoints),
        )

    # -----------------------------------------------------------------------
    # Web data model discovery
    # -----------------------------------------------------------------------

    def _discover_web_traffic(self) -> List[Dict[str, Any]]:
        """Query the CIM Web data model for AI-related HTTP traffic."""
        if self.use_raw_search:
            return self._discover_web_traffic_raw()

        dest_filter = _build_dest_filter("Web")
        spl = _SPL_WEB_AI_TRAFFIC.format(dest_filter=dest_filter)
        return self._run_search(spl)

    def _discover_web_traffic_raw(self) -> List[Dict[str, Any]]:
        """Fallback: raw search when CIM data model is not accelerated."""
        dest_clauses = []
        for pattern in _AI_ENDPOINT_PATTERNS:
            # Convert regex to a simpler Splunk-friendly wildcard
            dest_clauses.append(f'dest="*{pattern[1]}*"')
        dest_filter = " OR ".join(dest_clauses)

        sourcetypes = ", ".join(
            f'"{s.strip()}"' for s in self.raw_sourcetypes.split(",")
        )
        spl = _SPL_RAW_WEB_SEARCH.format(
            index=self.raw_index,
            sourcetypes=sourcetypes,
            dest_filter=dest_filter,
        )
        return self._run_search(spl)

    def _process_web_results(self, results: List[Dict[str, Any]]):
        """Process web traffic search results into discovered assets."""
        for row in results:
            dest = row.get("dest", "")
            url = row.get("url", "")
            user_agent = row.get("http_user_agent", "")
            user = row.get("user", "")
            action = row.get("action", "")
            request_count = _safe_int(row.get("request_count", 0))
            bytes_in = _safe_int(row.get("total_bytes_in", 0))
            bytes_out = _safe_int(row.get("total_bytes_out", 0))
            unique_sources = _safe_int(row.get("unique_sources", 0))
            first_seen = row.get("first_seen", "")
            last_seen = row.get("last_seen", "")
            status = row.get("status", "")

            provider, service = _classify_endpoint(dest, url)
            if not provider:
                continue

            # Determine model from URL path if possible
            model_name = _extract_model_from_url(url, provider)

            # Build endpoint asset
            endpoint_key = f"{provider}:{dest}"
            if endpoint_key not in self.discovered_endpoints:
                self.discovered_endpoints[endpoint_key] = {
                    "id": endpoint_key,
                    "name": service,
                    "type": "ai_api_endpoint",
                    "provider": provider,
                    "dest": dest,
                    "discovery_source": self.instance_id,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "request_count": 0,
                    "total_bytes_in": 0,
                    "total_bytes_out": 0,
                    "unique_sources": 0,
                    "users": set(),
                    "user_agents": set(),
                    "urls": set(),
                    "actions": set(),
                    "tags": ["ai", "network-discovered", "splunk-web-dm"],
                }

            ep = self.discovered_endpoints[endpoint_key]
            ep["request_count"] += request_count
            ep["total_bytes_in"] += bytes_in
            ep["total_bytes_out"] += bytes_out
            ep["unique_sources"] = max(ep["unique_sources"], unique_sources)
            if last_seen:
                ep["last_seen"] = max(ep.get("last_seen") or "", last_seen)
            if first_seen:
                if not ep.get("first_seen"):
                    ep["first_seen"] = first_seen
                else:
                    ep["first_seen"] = min(ep["first_seen"], first_seen)
            if user:
                ep["users"].add(user)
            if user_agent:
                ep["user_agents"].add(user_agent)
            if url:
                ep["urls"].add(url)
            if action:
                ep["actions"].add(action)

            # Build tool asset (the client application making AI calls)
            if user_agent:
                tool_key = f"splunk-web:{user_agent}:{provider}"
                if tool_key not in self.discovered_tools:
                    tool_name = _infer_tool_name(user_agent)
                    self.discovered_tools[tool_key] = {
                        "id": tool_key,
                        "name": tool_name,
                        "type": "ai_client",
                        "provider": provider,
                        "discovery_source": self.instance_id,
                        "first_seen": first_seen,
                        "last_seen": last_seen,
                        "request_count": 0,
                        "metadata": {
                            "user_agent": user_agent,
                            "dest": dest,
                            "service": service,
                            "source_type": "web_data_model",
                        },
                        "tags": ["ai", "network-discovered", "splunk-web-dm"],
                    }
                self.discovered_tools[tool_key]["request_count"] += request_count
                if last_seen:
                    self.discovered_tools[tool_key]["last_seen"] = max(
                        self.discovered_tools[tool_key].get("last_seen") or "", last_seen
                    )

            # Build model asset if we could identify a model
            if model_name:
                model_key = f"{provider}:{model_name}"
                if model_key not in self.discovered_models:
                    self.discovered_models[model_key] = {
                        "id": model_key,
                        "name": model_name,
                        "provider": provider,
                        "discovery_source": self.instance_id,
                        "first_seen": first_seen,
                        "last_seen": last_seen,
                        "request_count": 0,
                        "metadata": {
                            "detected_from": "url_path",
                            "dest": dest,
                            "source_type": "web_data_model",
                        },
                        "tags": ["ai", "network-discovered", "splunk-web-dm"],
                    }
                self.discovered_models[model_key]["request_count"] += request_count

    # -----------------------------------------------------------------------
    # Network Traffic data model discovery
    # -----------------------------------------------------------------------

    def _discover_network_traffic(self) -> List[Dict[str, Any]]:
        """Query the CIM Network Traffic data model for AI-related connections."""
        dest_filter = _build_dest_filter("All_Traffic")
        spl = _SPL_NETWORK_AI_TRAFFIC.format(dest_filter=dest_filter)
        return self._run_search(spl)

    def _process_network_results(self, results: List[Dict[str, Any]]):
        """Process network traffic results into discovered endpoints."""
        for row in results:
            dest = row.get("dest", "")
            dest_port = row.get("dest_port", "")
            action = row.get("action", "")
            app = row.get("app", "")
            connection_count = _safe_int(row.get("connection_count", 0))
            total_bytes = _safe_int(row.get("total_bytes", 0))
            unique_sources = _safe_int(row.get("unique_sources", 0))
            first_seen = row.get("first_seen", "")
            last_seen = row.get("last_seen", "")
            transport = row.get("transport", "")

            provider, service = _classify_endpoint(dest)
            if not provider:
                continue

            endpoint_key = f"{provider}:{dest}"
            if endpoint_key not in self.discovered_endpoints:
                self.discovered_endpoints[endpoint_key] = {
                    "id": endpoint_key,
                    "name": service,
                    "type": "ai_api_endpoint",
                    "provider": provider,
                    "dest": dest,
                    "discovery_source": self.instance_id,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "request_count": 0,
                    "total_bytes_in": 0,
                    "total_bytes_out": 0,
                    "unique_sources": 0,
                    "users": set(),
                    "user_agents": set(),
                    "urls": set(),
                    "actions": set(),
                    "tags": ["ai", "network-discovered", "splunk-network-dm"],
                }

            ep = self.discovered_endpoints[endpoint_key]
            ep["request_count"] += connection_count
            if total_bytes:
                ep["total_bytes_in"] += total_bytes
            ep["unique_sources"] = max(ep["unique_sources"], unique_sources)
            if last_seen:
                ep["last_seen"] = max(ep.get("last_seen") or "", last_seen)
            if first_seen:
                if not ep.get("first_seen"):
                    ep["first_seen"] = first_seen
                else:
                    ep["first_seen"] = min(ep["first_seen"], first_seen)
            if action:
                ep["actions"].add(action)

            # Add network-level metadata
            ep.setdefault("metadata", {})
            if dest_port:
                ep["metadata"]["dest_port"] = dest_port
            if transport:
                if isinstance(transport, list):
                    ep["metadata"]["transport"] = transport
                else:
                    ep["metadata"]["transport"] = [transport]
            if app:
                ep["metadata"]["firewall_app"] = app

    # -----------------------------------------------------------------------
    # MCP detection
    # -----------------------------------------------------------------------

    def _discover_mcp_traffic(self) -> List[Dict[str, Any]]:
        """Search for MCP (Model Context Protocol) traffic patterns."""
        spl = _SPL_MCP_DETECTION
        return self._run_search(spl)

    def _process_mcp_results(self, results: List[Dict[str, Any]]):
        """Process MCP detection results."""
        for row in results:
            dest = row.get("dest", "")
            url = row.get("url", "")
            src = row.get("src", "")
            user = row.get("user", "")
            request_count = _safe_int(row.get("request_count", 0))
            first_seen = row.get("first_seen", "")
            last_seen = row.get("last_seen", "")

            server_key = f"mcp:{dest}:{url}"
            if server_key not in self.mcp_servers:
                self.mcp_servers[server_key] = {
                    "id": server_key,
                    "name": f"MCP Server ({dest})",
                    "type": "mcp_server",
                    "dest": dest,
                    "url": url,
                    "discovery_source": self.instance_id,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "request_count": request_count,
                    "sources": set(),
                    "users": set(),
                    "tags": ["mcp", "network-discovered", "splunk-web-dm"],
                }
            server = self.mcp_servers[server_key]
            if src:
                server["sources"].add(src)
            if user:
                server["users"].add(user)

            # Also register as a tool
            tool_key = f"splunk-mcp:{dest}"
            if tool_key not in self.discovered_tools:
                self.discovered_tools[tool_key] = {
                    "id": tool_key,
                    "name": f"MCP Client ({dest})",
                    "type": "mcp_client",
                    "provider": "mcp",
                    "discovery_source": self.instance_id,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                    "request_count": request_count,
                    "metadata": {
                        "dest": dest,
                        "url": url,
                        "source_type": "mcp_detection",
                    },
                    "tags": ["mcp", "network-discovered", "splunk-web-dm"],
                }

    # -----------------------------------------------------------------------
    # Serialization helpers
    # -----------------------------------------------------------------------

    def _serialize_endpoints(self) -> List[Dict[str, Any]]:
        """Serialize endpoints, converting sets to lists for JSON compat."""
        results = []
        for ep in self.discovered_endpoints.values():
            serialized = dict(ep)
            for key in ("users", "user_agents", "urls", "actions"):
                if isinstance(serialized.get(key), set):
                    serialized[key] = sorted(serialized[key])
            results.append(serialized)
        return results


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _classify_endpoint(dest: str, url: str = "") -> tuple:
    """Match a destination hostname against known AI provider patterns.

    Args:
        dest: Destination hostname or IP.
        url: Full URL (optional, for more precise matching).

    Returns:
        Tuple of ``(provider_name, service_label)`` or ``("", "")`` if
        no match.
    """
    search_target = url or dest
    for pattern, provider, service in _AI_ENDPOINT_PATTERNS:
        if pattern.search(search_target):
            return provider, service
    return "", ""


def _extract_model_from_url(url: str, provider: str) -> Optional[str]:
    """Try to extract a model name from a URL path.

    Applies provider-specific heuristics to identify model names from
    API URL paths.

    Args:
        url: The full URL or path.
        provider: The provider key (e.g. ``"openai"``, ``"aws_bedrock"``).

    Returns:
        Model name string if identified, else None.
    """
    if not url:
        return None

    # Bedrock: /model/<model-id>/invoke
    if provider == "aws_bedrock":
        match = re.search(r"/model/([\w.-]+)/invoke", url)
        if match:
            return match.group(1)

    # Azure OpenAI: /openai/deployments/<deployment>/chat/completions
    if provider == "azure_openai":
        match = re.search(r"/openai/deployments/([\w.-]+)/", url)
        if match:
            return match.group(1)

    # HuggingFace: /models/<model-name>
    if provider == "huggingface":
        match = re.search(r"/models/([\w./-]+)", url)
        if match:
            return match.group(1)

    return None


def _infer_tool_name(user_agent: str) -> str:
    """Derive a human-readable tool name from a User-Agent string.

    Args:
        user_agent: The HTTP User-Agent header value.

    Returns:
        A cleaned-up tool name.
    """
    if not user_agent:
        return "Unknown Client"

    # Extract the first product token (e.g. "python-requests/2.31.0" -> "python-requests")
    parts = user_agent.split("/")
    name = parts[0].strip()

    # Common SDK user-agents
    ua_lower = user_agent.lower()
    if "openai-python" in ua_lower or "openai/" in ua_lower:
        return "OpenAI Python SDK"
    if "openai-node" in ua_lower:
        return "OpenAI Node SDK"
    if "anthropic-python" in ua_lower or "anthropic/" in ua_lower:
        return "Anthropic Python SDK"
    if "anthropic-typescript" in ua_lower:
        return "Anthropic TypeScript SDK"
    if "langchain" in ua_lower:
        return "LangChain"
    if "llamaindex" in ua_lower or "llama-index" in ua_lower or "llama_index" in ua_lower:
        return "LlamaIndex"
    if "python-requests" in ua_lower:
        return "Python Requests"
    if "curl" in ua_lower:
        return "cURL"
    if "axios" in ua_lower:
        return "Axios (Node.js)"
    if "go-http-client" in ua_lower:
        return "Go HTTP Client"

    return name if name else "Unknown Client"


def _safe_int(val: Any) -> int:
    """Safely convert a value to int, returning 0 on failure."""
    if val is None:
        return 0
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0
