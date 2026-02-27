"""
Unit tests for the Splunk discovery plugin.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from open_cite.plugins.splunk import (
    SplunkPlugin,
    _classify_endpoint,
    _extract_model_from_url,
    _infer_tool_name,
    _safe_int,
    _build_dest_filter,
    _AI_ENDPOINT_PATTERNS,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _make_plugin(**overrides) -> SplunkPlugin:
    """Create a SplunkPlugin with mock HTTP client."""
    defaults = {
        "splunk_url": "https://splunk.example.com:8089",
        "token": "test-token-123",
        "instance_id": "splunk-test",
        "display_name": "Test Splunk",
    }
    defaults.update(overrides)
    mock_http = MagicMock()
    defaults["http_client"] = mock_http
    return SplunkPlugin(**defaults)


def _mock_response(json_data=None, status_code=200):
    """Create a mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status.return_value = None
    return resp


# -----------------------------------------------------------------------
# Sample Splunk search results (simulating CIM output)
# -----------------------------------------------------------------------


SAMPLE_WEB_RESULTS = [
    {
        "url": "https://api.openai.com/v1/chat/completions",
        "dest": "api.openai.com",
        "action": "allowed",
        "app": "web_proxy",
        "user": "alice@corp.com",
        "http_user_agent": "openai-python/1.12.0",
        "status": "200",
        "request_count": "150",
        "total_bytes_in": "450000",
        "total_bytes_out": "1200000",
        "unique_sources": "3",
        "http_methods": "POST",
        "first_seen": "2025-01-01T00:00:00",
        "last_seen": "2025-01-15T12:00:00",
    },
    {
        "url": "https://api.anthropic.com/v1/messages",
        "dest": "api.anthropic.com",
        "action": "allowed",
        "app": "web_proxy",
        "user": "bob@corp.com",
        "http_user_agent": "anthropic-python/0.18.0",
        "status": "200",
        "request_count": "75",
        "total_bytes_in": "200000",
        "total_bytes_out": "600000",
        "unique_sources": "2",
        "http_methods": "POST",
        "first_seen": "2025-01-02T00:00:00",
        "last_seen": "2025-01-14T08:00:00",
    },
    {
        "url": "https://myinstance.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02",
        "dest": "myinstance.openai.azure.com",
        "action": "allowed",
        "app": "web_proxy",
        "user": "charlie@corp.com",
        "http_user_agent": "openai-python/1.12.0",
        "status": "200",
        "request_count": "500",
        "total_bytes_in": "1500000",
        "total_bytes_out": "3000000",
        "unique_sources": "10",
        "http_methods": "POST",
        "first_seen": "2025-01-01T00:00:00",
        "last_seen": "2025-01-15T18:00:00",
    },
]

SAMPLE_NETWORK_RESULTS = [
    {
        "dest": "api.openai.com",
        "dest_port": "443",
        "action": "allow",
        "app": "ssl",
        "connection_count": "2500",
        "total_bytes": "15000000",
        "total_bytes_in": "5000000",
        "total_bytes_out": "10000000",
        "unique_sources": "15",
        "transport": "tcp",
        "first_seen": "2024-12-01T00:00:00",
        "last_seen": "2025-01-15T23:59:00",
    },
    {
        "dest": "bedrock-runtime.us-east-1.amazonaws.com",
        "dest_port": "443",
        "action": "allow",
        "app": "ssl",
        "connection_count": "800",
        "total_bytes": "5000000",
        "total_bytes_in": "2000000",
        "total_bytes_out": "3000000",
        "unique_sources": "5",
        "transport": "tcp",
        "first_seen": "2025-01-05T00:00:00",
        "last_seen": "2025-01-15T20:00:00",
    },
]

SAMPLE_MCP_RESULTS = [
    {
        "dest": "mcp.internal.corp.com",
        "url": "https://mcp.internal.corp.com/jsonrpc",
        "src": "10.0.1.50",
        "http_method": "POST",
        "user": "dev-user@corp.com",
        "request_count": "45",
        "unique_sources": "2",
        "first_seen": "2025-01-10T00:00:00",
        "last_seen": "2025-01-15T16:00:00",
    },
]


# -----------------------------------------------------------------------
# Test: plugin metadata and configuration
# -----------------------------------------------------------------------


class TestPluginMetadata:
    def test_plugin_type(self):
        plugin = _make_plugin()
        assert plugin.plugin_type == "splunk"

    def test_metadata(self):
        meta = SplunkPlugin.plugin_metadata()
        assert meta["name"] == "Splunk"
        assert "required_fields" in meta
        assert "splunk_url" in meta["required_fields"]
        assert "token" in meta["required_fields"]
        assert "username" in meta["required_fields"]

    def test_supported_asset_types(self):
        plugin = _make_plugin()
        assert "tool" in plugin.supported_asset_types
        assert "model" in plugin.supported_asset_types
        assert "endpoint" in plugin.supported_asset_types

    def test_get_config_masks_token(self):
        plugin = _make_plugin()
        config = plugin.get_config()
        assert config["token"] == "****"
        assert config["splunk_url"] == "https://splunk.example.com:8089"

    def test_get_config_no_token(self):
        plugin = _make_plugin(token=None)
        config = plugin.get_config()
        assert config["token"] is None

    def test_identification_attributes(self):
        plugin = _make_plugin()
        attrs = plugin.get_identification_attributes()
        assert "splunk.dest" in attrs
        assert "splunk.provider" in attrs


# -----------------------------------------------------------------------
# Test: from_config factory
# -----------------------------------------------------------------------


class TestFromConfig:
    def test_basic_config(self):
        plugin = SplunkPlugin.from_config(
            config={
                "splunk_url": "https://splunk.corp.com:8089",
                "token": "my-token",
                "time_range": "-48h",
            },
            instance_id="splunk-prod",
            display_name="Production Splunk",
        )
        assert plugin.splunk_url == "https://splunk.corp.com:8089"
        assert plugin.token == "my-token"
        assert plugin.time_range == "-48h"
        assert plugin.instance_id == "splunk-prod"
        assert plugin.display_name == "Production Splunk"

    def test_defaults(self):
        plugin = SplunkPlugin.from_config(config={})
        assert plugin.splunk_url == "https://localhost:8089"
        assert plugin.time_range == "-24h"
        assert plugin.verify_ssl is True
        assert plugin.use_raw_search is False

    def test_raw_search_mode(self):
        plugin = SplunkPlugin.from_config(
            config={
                "use_raw_search": "true",
                "raw_index": "proxy_logs",
                "raw_sourcetypes": "bluecoat, paloalto",
            }
        )
        assert plugin.use_raw_search is True
        assert plugin.raw_index == "proxy_logs"
        assert plugin.raw_sourcetypes == "bluecoat, paloalto"


# -----------------------------------------------------------------------
# Test: authentication
# -----------------------------------------------------------------------


class TestAuthentication:
    def test_token_auth(self):
        plugin = _make_plugin(token="bearer-token-123")
        plugin._authenticate()
        headers = plugin._auth_headers()
        assert headers["Authorization"] == "Bearer bearer-token-123"

    def test_session_auth(self):
        plugin = _make_plugin(token=None, username="admin", password="changeme")
        plugin.http_client.post.return_value = _mock_response(
            {"sessionKey": "session-key-456"}
        )
        plugin._authenticate()
        headers = plugin._auth_headers()
        assert headers["Authorization"] == "Splunk session-key-456"

    def test_no_credentials_raises(self):
        plugin = _make_plugin(token=None, username=None, password=None)
        with pytest.raises(ValueError, match="credentials required"):
            plugin._authenticate()


# -----------------------------------------------------------------------
# Test: connection verification
# -----------------------------------------------------------------------


class TestVerifyConnection:
    def test_verify_success(self):
        plugin = _make_plugin()
        plugin.http_client.get.return_value = _mock_response({
            "entry": [{
                "content": {
                    "serverName": "splunk-prod-01",
                    "version": "9.2.1",
                }
            }]
        })
        result = plugin.verify_connection()
        assert result["success"] is True
        assert result["server_name"] == "splunk-prod-01"
        assert result["version"] == "9.2.1"

    def test_verify_failure(self):
        plugin = _make_plugin()
        plugin.http_client.get.side_effect = Exception("Connection refused")
        result = plugin.verify_connection()
        assert result["success"] is False
        assert "Connection refused" in result["error"]


# -----------------------------------------------------------------------
# Test: endpoint classification
# -----------------------------------------------------------------------


class TestClassifyEndpoint:
    def test_openai(self):
        provider, service = _classify_endpoint("api.openai.com")
        assert provider == "openai"
        assert service == "OpenAI API"

    def test_anthropic(self):
        provider, service = _classify_endpoint("api.anthropic.com")
        assert provider == "anthropic"
        assert service == "Anthropic API"

    def test_azure_openai(self):
        provider, service = _classify_endpoint("myinstance.openai.azure.com")
        assert provider == "azure_openai"
        assert service == "Azure OpenAI"

    def test_aws_bedrock_runtime(self):
        provider, service = _classify_endpoint("bedrock-runtime.us-east-1.amazonaws.com")
        assert provider == "aws_bedrock"
        assert service == "AWS Bedrock"

    def test_aws_bedrock_control(self):
        provider, service = _classify_endpoint("bedrock.us-west-2.amazonaws.com")
        assert provider == "aws_bedrock"
        assert service == "AWS Bedrock"

    def test_google_gemini(self):
        provider, service = _classify_endpoint("generativelanguage.googleapis.com")
        assert provider == "google"
        assert service == "Google AI (Gemini)"

    def test_google_vertex(self):
        provider, service = _classify_endpoint("us-central1-aiplatform.googleapis.com")
        assert provider == "google"
        assert service == "Vertex AI"

    def test_cohere(self):
        provider, service = _classify_endpoint("api.cohere.ai")
        assert provider == "cohere"
        assert service == "Cohere API"

    def test_huggingface(self):
        provider, service = _classify_endpoint("api-inference.huggingface.co")
        assert provider == "huggingface"
        assert service == "HuggingFace Inference"

    def test_mistral(self):
        provider, service = _classify_endpoint("api.mistral.ai")
        assert provider == "mistral"
        assert service == "Mistral AI API"

    def test_openrouter(self):
        provider, service = _classify_endpoint("openrouter.ai")
        assert provider == "openrouter"
        assert service == "OpenRouter"

    def test_groq(self):
        provider, service = _classify_endpoint("api.groq.com")
        assert provider == "groq"
        assert service == "Groq API"

    def test_deepseek(self):
        provider, service = _classify_endpoint("api.deepseek.com")
        assert provider == "deepseek"
        assert service == "DeepSeek API"

    def test_unknown_endpoint(self):
        provider, service = _classify_endpoint("api.example.com")
        assert provider == ""
        assert service == ""

    def test_classify_from_url(self):
        provider, service = _classify_endpoint(
            "api.openai.com",
            "https://api.openai.com/v1/chat/completions",
        )
        assert provider == "openai"


# -----------------------------------------------------------------------
# Test: model extraction from URL
# -----------------------------------------------------------------------


class TestModelExtraction:
    def test_bedrock_model(self):
        model = _extract_model_from_url(
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-sonnet/invoke",
            "aws_bedrock",
        )
        assert model == "anthropic.claude-3-sonnet"

    def test_azure_deployment(self):
        model = _extract_model_from_url(
            "https://myinstance.openai.azure.com/openai/deployments/gpt-4o/chat/completions",
            "azure_openai",
        )
        assert model == "gpt-4o"

    def test_huggingface_model(self):
        model = _extract_model_from_url(
            "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B",
            "huggingface",
        )
        assert model == "mistralai/Mixtral-8x7B"

    def test_openai_no_model_in_url(self):
        model = _extract_model_from_url(
            "https://api.openai.com/v1/chat/completions",
            "openai",
        )
        assert model is None

    def test_empty_url(self):
        model = _extract_model_from_url("", "openai")
        assert model is None


# -----------------------------------------------------------------------
# Test: tool name inference from user agent
# -----------------------------------------------------------------------


class TestInferToolName:
    def test_openai_python(self):
        assert _infer_tool_name("openai-python/1.12.0") == "OpenAI Python SDK"

    def test_openai_node(self):
        assert _infer_tool_name("openai-node/4.20.0") == "OpenAI Node SDK"

    def test_anthropic_python(self):
        assert _infer_tool_name("anthropic-python/0.18.0") == "Anthropic Python SDK"

    def test_anthropic_typescript(self):
        assert _infer_tool_name("anthropic-typescript/0.12.0") == "Anthropic TypeScript SDK"

    def test_langchain(self):
        assert _infer_tool_name("LangChain/0.1.0") == "LangChain"

    def test_llamaindex(self):
        assert _infer_tool_name("llama-index/0.10.0") == "LlamaIndex"

    def test_python_requests(self):
        assert _infer_tool_name("python-requests/2.31.0") == "Python Requests"

    def test_curl(self):
        assert _infer_tool_name("curl/8.4.0") == "cURL"

    def test_go_http(self):
        assert _infer_tool_name("Go-http-client/2.0") == "Go HTTP Client"

    def test_unknown(self):
        assert _infer_tool_name("CustomApp/1.0") == "CustomApp"

    def test_empty(self):
        assert _infer_tool_name("") == "Unknown Client"


# -----------------------------------------------------------------------
# Test: safe int conversion
# -----------------------------------------------------------------------


class TestSafeInt:
    def test_string_int(self):
        assert _safe_int("42") == 42

    def test_string_float(self):
        assert _safe_int("3.14") == 3

    def test_int(self):
        assert _safe_int(100) == 100

    def test_none(self):
        assert _safe_int(None) == 0

    def test_bad_string(self):
        assert _safe_int("not_a_number") == 0


# -----------------------------------------------------------------------
# Test: web traffic processing
# -----------------------------------------------------------------------


class TestWebTrafficProcessing:
    def test_process_web_results(self):
        plugin = _make_plugin()
        plugin._process_web_results(SAMPLE_WEB_RESULTS)

        # Should discover endpoints for OpenAI, Anthropic, and Azure OpenAI
        assert len(plugin.discovered_endpoints) == 3

        # Check OpenAI endpoint
        openai_ep = plugin.discovered_endpoints.get("openai:api.openai.com")
        assert openai_ep is not None
        assert openai_ep["provider"] == "openai"
        assert openai_ep["name"] == "OpenAI API"
        assert openai_ep["request_count"] == 150

        # Check Anthropic endpoint
        anthropic_ep = plugin.discovered_endpoints.get("anthropic:api.anthropic.com")
        assert anthropic_ep is not None
        assert anthropic_ep["provider"] == "anthropic"
        assert anthropic_ep["request_count"] == 75

        # Check Azure OpenAI endpoint
        azure_ep = plugin.discovered_endpoints.get("azure_openai:myinstance.openai.azure.com")
        assert azure_ep is not None
        assert azure_ep["provider"] == "azure_openai"

    def test_discovers_tools_from_user_agents(self):
        plugin = _make_plugin()
        plugin._process_web_results(SAMPLE_WEB_RESULTS)

        # Should find tools based on user agents
        assert len(plugin.discovered_tools) > 0

        # OpenAI Python SDK tool
        openai_tool_key = "splunk-web:openai-python/1.12.0:openai"
        assert openai_tool_key in plugin.discovered_tools
        tool = plugin.discovered_tools[openai_tool_key]
        assert tool["name"] == "OpenAI Python SDK"
        assert tool["type"] == "ai_client"

        # Anthropic Python SDK tool
        anthropic_tool_key = "splunk-web:anthropic-python/0.18.0:anthropic"
        assert anthropic_tool_key in plugin.discovered_tools
        tool = plugin.discovered_tools[anthropic_tool_key]
        assert tool["name"] == "Anthropic Python SDK"

    def test_discovers_models_from_azure_url(self):
        plugin = _make_plugin()
        plugin._process_web_results(SAMPLE_WEB_RESULTS)

        # Azure OpenAI should extract gpt-4o from deployment path
        model_key = "azure_openai:gpt-4o"
        assert model_key in plugin.discovered_models
        model = plugin.discovered_models[model_key]
        assert model["name"] == "gpt-4o"
        assert model["provider"] == "azure_openai"
        assert model["request_count"] == 500

    def test_aggregates_request_counts(self):
        # Two results for the same endpoint
        results = [
            {
                "url": "https://api.openai.com/v1/chat/completions",
                "dest": "api.openai.com",
                "action": "allowed",
                "app": "proxy",
                "user": "alice",
                "http_user_agent": "openai-python/1.0",
                "status": "200",
                "request_count": "100",
                "total_bytes_in": "10000",
                "total_bytes_out": "20000",
                "unique_sources": "5",
                "http_methods": "POST",
                "first_seen": "2025-01-01T00:00:00",
                "last_seen": "2025-01-10T00:00:00",
            },
            {
                "url": "https://api.openai.com/v1/embeddings",
                "dest": "api.openai.com",
                "action": "allowed",
                "app": "proxy",
                "user": "bob",
                "http_user_agent": "openai-python/1.0",
                "status": "200",
                "request_count": "200",
                "total_bytes_in": "5000",
                "total_bytes_out": "15000",
                "unique_sources": "3",
                "http_methods": "POST",
                "first_seen": "2025-01-02T00:00:00",
                "last_seen": "2025-01-12T00:00:00",
            },
        ]
        plugin = _make_plugin()
        plugin._process_web_results(results)

        ep = plugin.discovered_endpoints["openai:api.openai.com"]
        assert ep["request_count"] == 300
        assert ep["total_bytes_in"] == 15000
        assert ep["total_bytes_out"] == 35000
        assert "alice" in ep["users"]
        assert "bob" in ep["users"]

    def test_tracks_first_and_last_seen(self):
        results = [
            {
                "dest": "api.openai.com",
                "url": "",
                "action": "",
                "app": "",
                "user": "",
                "http_user_agent": "",
                "status": "",
                "request_count": "10",
                "total_bytes_in": "0",
                "total_bytes_out": "0",
                "unique_sources": "1",
                "http_methods": "",
                "first_seen": "2025-01-05T00:00:00",
                "last_seen": "2025-01-10T00:00:00",
            },
            {
                "dest": "api.openai.com",
                "url": "",
                "action": "",
                "app": "",
                "user": "",
                "http_user_agent": "",
                "status": "",
                "request_count": "20",
                "total_bytes_in": "0",
                "total_bytes_out": "0",
                "unique_sources": "1",
                "http_methods": "",
                "first_seen": "2025-01-01T00:00:00",
                "last_seen": "2025-01-15T00:00:00",
            },
        ]
        plugin = _make_plugin()
        plugin._process_web_results(results)

        ep = plugin.discovered_endpoints["openai:api.openai.com"]
        assert ep["first_seen"] == "2025-01-01T00:00:00"
        assert ep["last_seen"] == "2025-01-15T00:00:00"


# -----------------------------------------------------------------------
# Test: network traffic processing
# -----------------------------------------------------------------------


class TestNetworkTrafficProcessing:
    def test_process_network_results(self):
        plugin = _make_plugin()
        plugin._process_network_results(SAMPLE_NETWORK_RESULTS)

        assert len(plugin.discovered_endpoints) == 2

        # OpenAI endpoint from network traffic
        openai_ep = plugin.discovered_endpoints.get("openai:api.openai.com")
        assert openai_ep is not None
        assert openai_ep["request_count"] == 2500
        assert "splunk-network-dm" in openai_ep["tags"]

        # Bedrock endpoint
        bedrock_ep = plugin.discovered_endpoints.get(
            "aws_bedrock:bedrock-runtime.us-east-1.amazonaws.com"
        )
        assert bedrock_ep is not None
        assert bedrock_ep["provider"] == "aws_bedrock"

    def test_network_metadata(self):
        plugin = _make_plugin()
        plugin._process_network_results(SAMPLE_NETWORK_RESULTS)

        bedrock_ep = plugin.discovered_endpoints[
            "aws_bedrock:bedrock-runtime.us-east-1.amazonaws.com"
        ]
        assert bedrock_ep["metadata"]["dest_port"] == "443"
        assert "tcp" in bedrock_ep["metadata"]["transport"]

    def test_merges_with_web_results(self):
        """Network results should merge into existing endpoints from web."""
        plugin = _make_plugin()
        # First, process web results
        plugin._process_web_results(SAMPLE_WEB_RESULTS)
        web_count = plugin.discovered_endpoints["openai:api.openai.com"]["request_count"]

        # Then process network results (should add to existing)
        plugin._process_network_results(SAMPLE_NETWORK_RESULTS)
        combined_count = plugin.discovered_endpoints["openai:api.openai.com"]["request_count"]

        assert combined_count == web_count + 2500


# -----------------------------------------------------------------------
# Test: MCP detection
# -----------------------------------------------------------------------


class TestMCPDetection:
    def test_process_mcp_results(self):
        plugin = _make_plugin()
        plugin._process_mcp_results(SAMPLE_MCP_RESULTS)

        assert len(plugin.mcp_servers) == 1
        server = list(plugin.mcp_servers.values())[0]
        assert server["type"] == "mcp_server"
        assert server["dest"] == "mcp.internal.corp.com"
        assert "10.0.1.50" in server["sources"]
        assert "dev-user@corp.com" in server["users"]

    def test_mcp_creates_tool(self):
        plugin = _make_plugin()
        plugin._process_mcp_results(SAMPLE_MCP_RESULTS)

        tool_key = "splunk-mcp:mcp.internal.corp.com"
        assert tool_key in plugin.discovered_tools
        tool = plugin.discovered_tools[tool_key]
        assert tool["type"] == "mcp_client"
        assert tool["provider"] == "mcp"


# -----------------------------------------------------------------------
# Test: full discovery orchestration
# -----------------------------------------------------------------------


class TestDiscovery:
    def test_discover_calls_all_sources(self):
        plugin = _make_plugin()

        # Mock each discovery method to return our sample data
        plugin.http_client.post.return_value = _mock_response(
            {"results": SAMPLE_WEB_RESULTS}
        )

        call_count = {"web": 0, "net": 0, "mcp": 0}

        original_web = plugin._discover_web_traffic
        original_net = plugin._discover_network_traffic
        original_mcp = plugin._discover_mcp_traffic

        def mock_web():
            call_count["web"] += 1
            return SAMPLE_WEB_RESULTS

        def mock_net():
            call_count["net"] += 1
            return SAMPLE_NETWORK_RESULTS

        def mock_mcp():
            call_count["mcp"] += 1
            return SAMPLE_MCP_RESULTS

        plugin._discover_web_traffic = mock_web
        plugin._discover_network_traffic = mock_net
        plugin._discover_mcp_traffic = mock_mcp

        plugin.discover()

        assert call_count["web"] == 1
        assert call_count["net"] == 1
        assert call_count["mcp"] == 1

        # Should have endpoints from both web and network
        assert len(plugin.discovered_endpoints) > 0
        assert len(plugin.discovered_tools) > 0

    def test_discover_handles_partial_failure(self):
        """Discovery should continue if one source fails."""
        plugin = _make_plugin()

        def fail_web():
            raise Exception("Web data model not found")

        def ok_net():
            return SAMPLE_NETWORK_RESULTS

        def ok_mcp():
            return []

        plugin._discover_web_traffic = fail_web
        plugin._discover_network_traffic = ok_net
        plugin._discover_mcp_traffic = ok_mcp

        plugin.discover()

        # Network results should still be processed
        assert len(plugin.discovered_endpoints) == 2


# -----------------------------------------------------------------------
# Test: list_assets interface
# -----------------------------------------------------------------------


class TestListAssets:
    def test_list_tools(self):
        plugin = _make_plugin()
        plugin._process_web_results(SAMPLE_WEB_RESULTS)
        tools = plugin.list_assets("tool")
        assert len(tools) > 0
        assert all(t.get("type") in ("ai_client", "mcp_client") for t in tools)

    def test_list_models(self):
        plugin = _make_plugin()
        plugin._process_web_results(SAMPLE_WEB_RESULTS)
        models = plugin.list_assets("model")
        # Azure URL should produce at least one model
        assert any(m["name"] == "gpt-4o" for m in models)

    def test_list_endpoints(self):
        plugin = _make_plugin()
        plugin._process_web_results(SAMPLE_WEB_RESULTS)
        endpoints = plugin.list_assets("endpoint")
        assert len(endpoints) == 3

    def test_list_unknown_type(self):
        plugin = _make_plugin()
        assert plugin.list_assets("unknown_type") == []


# -----------------------------------------------------------------------
# Test: export_assets
# -----------------------------------------------------------------------


class TestExportAssets:
    def test_export(self):
        plugin = _make_plugin()
        plugin._process_web_results(SAMPLE_WEB_RESULTS)
        exported = plugin.export_assets()
        assert "tools" in exported
        assert "models" in exported
        assert len(exported["tools"]) > 0


# -----------------------------------------------------------------------
# Test: dest filter builder
# -----------------------------------------------------------------------


class TestDestFilter:
    def test_builds_filter(self):
        filt = _build_dest_filter("Web")
        assert 'Web.dest="api.openai.com"' in filt
        assert 'Web.dest="api.anthropic.com"' in filt
        assert 'Web.dest="*.openai.azure.com"' in filt
        assert " OR " in filt

    def test_network_prefix(self):
        filt = _build_dest_filter("All_Traffic")
        assert 'All_Traffic.dest="api.openai.com"' in filt


# -----------------------------------------------------------------------
# Test: plugin registration via registry
# -----------------------------------------------------------------------


class TestPluginRegistration:
    def test_discovered_by_registry(self):
        from open_cite.plugins.registry import discover_plugin_classes, reset_cache

        reset_cache()
        classes = discover_plugin_classes()
        assert "splunk" in classes
        assert classes["splunk"] is SplunkPlugin

    def test_create_via_registry(self):
        from open_cite.plugins.registry import create_plugin_instance, reset_cache

        reset_cache()
        plugin = create_plugin_instance(
            "splunk",
            config={
                "splunk_url": "https://splunk.test:8089",
                "token": "test-token",
            },
            instance_id="splunk-registry-test",
        )
        assert isinstance(plugin, SplunkPlugin)
        assert plugin.splunk_url == "https://splunk.test:8089"
        assert plugin.instance_id == "splunk-registry-test"


# -----------------------------------------------------------------------
# Test: endpoint serialization
# -----------------------------------------------------------------------


class TestSerialization:
    def test_serialize_endpoints_converts_sets(self):
        plugin = _make_plugin()
        plugin._process_web_results(SAMPLE_WEB_RESULTS)

        serialized = plugin._serialize_endpoints()
        for ep in serialized:
            for key in ("users", "user_agents", "urls", "actions"):
                if key in ep:
                    assert isinstance(ep[key], list), f"{key} should be a list"

    def test_to_dict(self):
        plugin = _make_plugin()
        d = plugin.to_dict()
        assert d["plugin_type"] == "splunk"
        assert d["instance_id"] == "splunk-test"
        assert "tool" in d["supported_asset_types"]


# -----------------------------------------------------------------------
# Test: edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_results(self):
        plugin = _make_plugin()
        plugin._process_web_results([])
        assert len(plugin.discovered_endpoints) == 0
        assert len(plugin.discovered_tools) == 0

    def test_non_ai_traffic_ignored(self):
        results = [
            {
                "url": "https://www.google.com/search?q=test",
                "dest": "www.google.com",
                "action": "allowed",
                "app": "proxy",
                "user": "",
                "http_user_agent": "Mozilla/5.0",
                "status": "200",
                "request_count": "1000",
                "total_bytes_in": "0",
                "total_bytes_out": "0",
                "unique_sources": "1",
                "http_methods": "GET",
                "first_seen": "",
                "last_seen": "",
            },
        ]
        plugin = _make_plugin()
        plugin._process_web_results(results)
        assert len(plugin.discovered_endpoints) == 0

    def test_missing_fields_handled(self):
        results = [
            {
                "dest": "api.openai.com",
                # All other fields missing
            },
        ]
        plugin = _make_plugin()
        plugin._process_web_results(results)
        ep = plugin.discovered_endpoints.get("openai:api.openai.com")
        assert ep is not None
        assert ep["request_count"] == 0

    def test_verify_ssl_string_false(self):
        plugin = _make_plugin(verify_ssl="false")
        assert plugin.verify_ssl is False

    def test_verify_ssl_string_true(self):
        plugin = _make_plugin(verify_ssl="true")
        assert plugin.verify_ssl is True

    def test_url_trailing_slash_stripped(self):
        plugin = _make_plugin(splunk_url="https://splunk.example.com:8089/")
        assert plugin.splunk_url == "https://splunk.example.com:8089"
