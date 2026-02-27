"""
Unit tests for the Microsoft Fabric discovery plugin.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from open_cite.plugins.microsoft_fabric import (
    FABRIC_API_BASE,
    FABRIC_SCOPE,
    MicrosoftFabricPlugin,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def plugin():
    """Plugin configured with service-principal credentials."""
    return MicrosoftFabricPlugin(
        tenant_id="test-tenant",
        client_id="test-client",
        client_secret="test-secret",
        instance_id="fabric-test",
        display_name="Fabric Test",
    )


@pytest.fixture
def plugin_with_token():
    """Plugin configured with a pre-acquired access token."""
    p = MicrosoftFabricPlugin(
        access_token="pre-acquired-token",
        instance_id="fabric-token",
    )
    # Treat the pre-acquired token as not-yet-expired.
    p._token_expires_at = time.time() + 3600
    return p


@pytest.fixture
def sample_workspaces():
    return [
        {
            "id": "ws-001",
            "displayName": "Sales Analytics",
            "description": "Sales workspace",
            "capacityId": "cap-001",
            "state": "Active",
        },
        {
            "id": "ws-002",
            "displayName": "Data Engineering",
            "description": "",
            "capacityId": "cap-001",
            "state": "Active",
        },
    ]


@pytest.fixture
def sample_lakehouses():
    return [
        {
            "id": "lh-001",
            "displayName": "Bronze Lakehouse",
            "description": "Raw data",
        },
        {
            "id": "lh-002",
            "displayName": "Silver Lakehouse",
            "description": "Curated data",
        },
    ]


@pytest.fixture
def sample_capacities():
    return [
        {
            "id": "cap-001",
            "displayName": "Production F64",
            "sku": "F64",
            "state": "Active",
            "region": "eastus",
        },
    ]


# ---------------------------------------------------------------------------
# Class attributes & metadata
# ---------------------------------------------------------------------------

class TestPluginMetadata:
    def test_plugin_type(self):
        assert MicrosoftFabricPlugin.plugin_type == "microsoft_fabric"

    def test_supported_asset_types(self, plugin):
        expected = {
            "workspace", "lakehouse", "warehouse", "notebook",
            "pipeline", "ml_model", "ml_experiment", "report",
            "semantic_model", "event_stream", "kql_database", "capacity",
        }
        assert plugin.supported_asset_types == expected

    def test_identification_attributes(self, plugin):
        attrs = plugin.get_identification_attributes()
        assert "fabric.tenant_id" in attrs
        assert "fabric.workspace_id" in attrs
        assert "fabric.item_id" in attrs

    def test_plugin_metadata_fields(self):
        meta = MicrosoftFabricPlugin.plugin_metadata()
        assert meta["name"] == "Microsoft Fabric"
        assert "tenant_id" in meta["required_fields"]
        assert "client_id" in meta["required_fields"]
        assert "client_secret" in meta["required_fields"]
        assert meta["required_fields"]["client_secret"]["type"] == "password"

    def test_get_config_masks_secrets(self, plugin):
        cfg = plugin.get_config()
        assert cfg["tenant_id"] == "test-tenant"
        assert cfg["client_secret"] == "****"

    def test_get_config_masks_token(self, plugin_with_token):
        cfg = plugin_with_token.get_config()
        assert cfg["access_token"] == "****"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_from_config_basic(self):
        p = MicrosoftFabricPlugin.from_config(
            {
                "tenant_id": "t1",
                "client_id": "c1",
                "client_secret": "s1",
            },
            instance_id="inst1",
            display_name="My Fabric",
        )
        assert p.tenant_id == "t1"
        assert p.client_id == "c1"
        assert p.instance_id == "inst1"
        assert p.display_name == "My Fabric"

    def test_from_config_workspace_ids_csv(self):
        p = MicrosoftFabricPlugin.from_config(
            {"workspace_ids": " ws-1, ws-2 ,ws-3 "},
        )
        assert p.workspace_ids == ["ws-1", "ws-2", "ws-3"]

    def test_from_config_workspace_ids_list(self):
        p = MicrosoftFabricPlugin.from_config(
            {"workspace_ids": ["ws-1", "ws-2"]},
        )
        assert p.workspace_ids == ["ws-1", "ws-2"]

    def test_from_config_empty_workspace_ids(self):
        p = MicrosoftFabricPlugin.from_config({"workspace_ids": ""})
        assert p.workspace_ids == []


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class TestAuthentication:
    def test_preacquired_token(self, plugin_with_token):
        assert plugin_with_token._get_access_token() == "pre-acquired-token"

    def test_no_credentials_raises(self):
        p = MicrosoftFabricPlugin()
        with pytest.raises(ValueError, match="No valid authentication"):
            p._get_access_token()

    @patch("open_cite.plugins.microsoft_fabric.requests.post")
    def test_client_credentials_flow(self, mock_post, plugin):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "new-token-123",
            "expires_in": 3600,
        }
        mock_post.return_value = mock_resp

        token = plugin._get_access_token()

        assert token == "new-token-123"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["data"]["grant_type"] == "client_credentials"
        assert call_kwargs[1]["data"]["scope"] == FABRIC_SCOPE
        assert call_kwargs[1]["data"]["client_id"] == "test-client"

    @patch("open_cite.plugins.microsoft_fabric.requests.post")
    def test_token_caching(self, mock_post, plugin):
        """Subsequent calls should use the cached token."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "cached-tok",
            "expires_in": 3600,
        }
        mock_post.return_value = mock_resp

        tok1 = plugin._get_access_token()
        tok2 = plugin._get_access_token()

        assert tok1 == tok2 == "cached-tok"
        # Should only call the token endpoint once.
        assert mock_post.call_count == 1


# ---------------------------------------------------------------------------
# Verify connection
# ---------------------------------------------------------------------------

class TestVerifyConnection:
    @patch("open_cite.plugins.microsoft_fabric.requests.get")
    @patch("open_cite.plugins.microsoft_fabric.requests.post")
    def test_verify_success(self, mock_post, mock_get, plugin):
        # Token acquisition
        token_resp = MagicMock()
        token_resp.raise_for_status = MagicMock()
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 3600}
        mock_post.return_value = token_resp

        # Workspaces call
        ws_resp = MagicMock()
        ws_resp.status_code = 200
        ws_resp.json.return_value = {"value": [{"id": "ws-1"}]}
        mock_get.return_value = ws_resp

        result = plugin.verify_connection()
        assert result["success"] is True
        assert "workspace_count_hint" in result

    @patch("open_cite.plugins.microsoft_fabric.requests.get")
    @patch("open_cite.plugins.microsoft_fabric.requests.post")
    def test_verify_failure_http(self, mock_post, mock_get, plugin):
        token_resp = MagicMock()
        token_resp.raise_for_status = MagicMock()
        token_resp.json.return_value = {"access_token": "tok", "expires_in": 3600}
        mock_post.return_value = token_resp

        ws_resp = MagicMock()
        ws_resp.status_code = 401
        ws_resp.text = "Unauthorized"
        mock_get.return_value = ws_resp

        result = plugin.verify_connection()
        assert result["success"] is False
        assert "401" in result["error"]

    def test_verify_failure_no_auth(self):
        p = MicrosoftFabricPlugin()
        result = p.verify_connection()
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Pagination helper
# ---------------------------------------------------------------------------

class TestPaginatedGet:
    @patch("open_cite.plugins.microsoft_fabric.requests.get")
    def test_single_page(self, mock_get, plugin_with_token):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"value": [{"id": "1"}, {"id": "2"}]}
        mock_get.return_value = resp

        items = plugin_with_token._paginated_get(
            "https://example.com/items",
            {"Authorization": "Bearer tok"},
        )
        assert len(items) == 2

    @patch("open_cite.plugins.microsoft_fabric.requests.get")
    def test_multiple_pages(self, mock_get, plugin_with_token):
        page1 = MagicMock()
        page1.status_code = 200
        page1.json.return_value = {
            "value": [{"id": "1"}],
            "continuationToken": "page2tok",
        }

        page2 = MagicMock()
        page2.status_code = 200
        page2.json.return_value = {"value": [{"id": "2"}]}

        mock_get.side_effect = [page1, page2]

        items = plugin_with_token._paginated_get(
            "https://example.com/items",
            {"Authorization": "Bearer tok"},
        )
        assert len(items) == 2
        assert items[0]["id"] == "1"
        assert items[1]["id"] == "2"

    @patch("open_cite.plugins.microsoft_fabric.requests.get")
    def test_max_items_limit(self, mock_get, plugin_with_token):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "value": [{"id": str(i)} for i in range(100)],
            "continuationToken": "more",
        }
        mock_get.return_value = resp

        items = plugin_with_token._paginated_get(
            "https://example.com/items",
            {"Authorization": "Bearer tok"},
            max_items=50,
        )
        assert len(items) == 50

    @patch("open_cite.plugins.microsoft_fabric.requests.get")
    def test_error_response(self, mock_get, plugin_with_token):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "Internal Server Error"
        mock_get.return_value = resp

        items = plugin_with_token._paginated_get(
            "https://example.com/items",
            {"Authorization": "Bearer tok"},
        )
        assert items == []


# ---------------------------------------------------------------------------
# List workspaces
# ---------------------------------------------------------------------------

class TestListWorkspaces:
    @patch("open_cite.plugins.microsoft_fabric.requests.get")
    def test_list_workspaces(self, mock_get, plugin_with_token, sample_workspaces):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"value": sample_workspaces}
        mock_get.return_value = resp

        results = plugin_with_token.list_assets("workspace")
        assert len(results) == 2
        assert results[0]["name"] == "Sales Analytics"
        assert results[0]["type"] == "workspace"
        assert results[0]["capacity_id"] == "cap-001"
        assert results[0]["discovery_source"] == "fabric-token"


# ---------------------------------------------------------------------------
# List workspace items (lakehouses, warehouses, etc.)
# ---------------------------------------------------------------------------

class TestListWorkspaceItems:
    @patch("open_cite.plugins.microsoft_fabric.requests.get")
    def test_list_lakehouses(self, mock_get, plugin_with_token, sample_workspaces, sample_lakehouses):
        ws_resp = MagicMock()
        ws_resp.status_code = 200
        ws_resp.json.return_value = {"value": sample_workspaces}

        lh_resp = MagicMock()
        lh_resp.status_code = 200
        lh_resp.json.return_value = {"value": sample_lakehouses}

        mock_get.side_effect = [ws_resp, lh_resp, lh_resp]

        results = plugin_with_token.list_assets("lakehouse")
        assert len(results) == 4  # 2 lakehouses x 2 workspaces
        assert results[0]["type"] == "lakehouse"
        assert results[0]["name"] == "Bronze Lakehouse"
        assert "fabric.workspace_id" in results[0]["metadata"]

    @patch("open_cite.plugins.microsoft_fabric.requests.get")
    def test_list_with_workspace_filter(self, mock_get, sample_workspaces, sample_lakehouses):
        """When workspace_ids is set, only those workspaces are scanned."""
        p = MicrosoftFabricPlugin(
            access_token="tok",
            workspace_ids=["ws-001"],
            instance_id="filtered",
        )
        p._token_expires_at = time.time() + 3600

        ws_resp = MagicMock()
        ws_resp.status_code = 200
        ws_resp.json.return_value = {"value": sample_workspaces}

        lh_resp = MagicMock()
        lh_resp.status_code = 200
        lh_resp.json.return_value = {"value": sample_lakehouses}

        mock_get.side_effect = [ws_resp, lh_resp]

        results = p.list_assets("lakehouse")
        assert len(results) == 2  # Only ws-001

    def test_unsupported_asset_type(self, plugin_with_token):
        with pytest.raises(ValueError, match="Unsupported asset type"):
            plugin_with_token.list_assets("nonexistent")


# ---------------------------------------------------------------------------
# List capacities
# ---------------------------------------------------------------------------

class TestListCapacities:
    @patch("open_cite.plugins.microsoft_fabric.requests.get")
    def test_list_capacities(self, mock_get, plugin_with_token, sample_capacities):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"value": sample_capacities}
        mock_get.return_value = resp

        results = plugin_with_token.list_assets("capacity")
        assert len(results) == 1
        assert results[0]["sku"] == "F64"
        assert results[0]["region"] == "eastus"


# ---------------------------------------------------------------------------
# Registry auto-discovery
# ---------------------------------------------------------------------------

class TestRegistryDiscovery:
    def test_plugin_discovered_by_registry(self):
        from open_cite.plugins.registry import discover_plugin_classes, reset_cache

        reset_cache()
        classes = discover_plugin_classes()
        assert "microsoft_fabric" in classes
        assert classes["microsoft_fabric"] is MicrosoftFabricPlugin


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExportAssets:
    @patch.object(MicrosoftFabricPlugin, "list_assets", return_value=[])
    def test_export_keys(self, mock_list, plugin_with_token):
        result = plugin_with_token.export_assets()
        expected_keys = {
            "fabric_workspaces",
            "fabric_lakehouses",
            "fabric_warehouses",
            "fabric_notebooks",
            "fabric_pipelines",
            "fabric_ml_models",
            "fabric_ml_experiments",
            "fabric_reports",
            "fabric_semantic_models",
            "fabric_event_streams",
            "fabric_kql_databases",
            "fabric_capacities",
        }
        assert set(result.keys()) == expected_keys
