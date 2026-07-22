"""
Tests for webhook delivery behavior in BaseDiscoveryPlugin.

Covers:
- Debounced delivery: payloads are buffered and merged on flush
- Immediate delivery when flush interval is 0
- 429 cooldown: deliveries pause after a 429 response
- Retry-After header is respected
- Cooldown expiry: deliveries resume after cooldown period
- Other URLs are not affected by one URL's cooldown
"""

import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from open_cite.plugins.opentelemetry import OpenTelemetryPlugin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plugin():
    """Build an OpenTelemetryPlugin without starting a receiver."""
    plugin = OpenTelemetryPlugin(
        host="127.0.0.1",
        port=0,
        instance_id="opentelemetry",
        display_name="OTLP",
        persist_mappings=False,
        embedded_receiver=True,
    )
    plugin.notify_data_changed = MagicMock()
    return plugin


def _make_payload(n_resource_spans=1, span_prefix="span"):
    """Build an OTLP payload with N resourceSpans, each containing 1 span."""
    resource_spans = []
    for i in range(n_resource_spans):
        resource_spans.append({
            "resource": {"attributes": []},
            "scopeSpans": [{
                "spans": [{
                    "traceId": f"trace{i:04d}",
                    "spanId": f"{span_prefix}{i:04d}",
                    "name": f"test-span-{i}",
                    "attributes": [],
                }]
            }],
        })
    return {"resourceSpans": resource_spans}


def _mock_response(status_code=200, headers=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = text
    return resp


# ---------------------------------------------------------------------------
# Debounce buffer tests
# ---------------------------------------------------------------------------

@patch("open_cite.core.validate_webhook_url", return_value=None)
class TestWebhookDebounce:

    @patch("open_cite.core._WEBHOOK_FLUSH_INTERVAL", 0)
    def test_immediate_delivery_when_interval_zero(self, _mock_validate):
        """With flush interval 0, payloads are delivered immediately."""
        plugin = _make_plugin()
        plugin.subscribe_webhook("http://example.com/traces")

        payload = _make_payload(2)
        with patch.object(plugin, "_flush_webhook_payloads") as mock_flush:
            plugin._deliver_to_webhooks(payload)
            mock_flush.assert_called_once()
            args = mock_flush.call_args
            # Should pass the payload directly
            assert len(args[0][0]) == 1
            assert len(args[0][0][0]["resourceSpans"]) == 2

    @patch("open_cite.core._WEBHOOK_FLUSH_INTERVAL", 5)
    def test_payloads_buffered_when_debounce_enabled(self, _mock_validate):
        """With flush interval > 0, payloads are buffered, not delivered immediately."""
        plugin = _make_plugin()
        plugin.subscribe_webhook("http://example.com/traces")

        with patch.object(plugin, "_flush_webhook_payloads") as mock_flush:
            plugin._deliver_to_webhooks(_make_payload(1))
            plugin._deliver_to_webhooks(_make_payload(1))
            # Should not have flushed yet
            mock_flush.assert_not_called()
            # Buffer should have 2 entries
            assert len(plugin._webhook_buffer) == 2

        # Clean up timer
        if plugin._webhook_flush_timer:
            plugin._webhook_flush_timer.cancel()

    @patch("open_cite.core._WEBHOOK_FLUSH_INTERVAL", 5)
    def test_flush_merges_buffered_resource_spans(self, _mock_validate):
        """Flushing merges all buffered resourceSpans into a single delivery."""
        plugin = _make_plugin()
        plugin.subscribe_webhook("http://example.com/traces")

        # Manually buffer payloads (bypass timer)
        plugin._webhook_buffer = [
            {"payload": _make_payload(3, span_prefix="a"), "inbound_headers": None},
            {"payload": _make_payload(2, span_prefix="b"), "inbound_headers": None},
        ]

        with patch.object(plugin, "_flush_webhook_payloads") as mock_flush:
            plugin._flush_webhook_buffer()
            mock_flush.assert_called_once()
            args = mock_flush.call_args
            merged_payload = args[0][0][0]
            # 3 + 2 = 5 resourceSpans merged
            assert len(merged_payload["resourceSpans"]) == 5

        # Buffer should be cleared
        assert len(plugin._webhook_buffer) == 0

    @patch("open_cite.core._WEBHOOK_FLUSH_INTERVAL", 5)
    def test_flush_uses_most_recent_headers_within_scope(self, _mock_validate):
        """Within one tenant scope, the most recent inbound_headers win on merge."""
        plugin = _make_plugin()
        plugin.subscribe_webhook("http://example.com/traces")

        # Same tenant/auth scope, differing only in a non-scope header (OTEL-HOST).
        old_headers = {"x-tenant-id": "t1", "Authorization": "Bearer k1", "OTEL-HOST": "host-a"}
        new_headers = {"x-tenant-id": "t1", "Authorization": "Bearer k1", "OTEL-HOST": "host-b"}
        plugin._webhook_buffer = [
            {"payload": _make_payload(1, span_prefix="a"), "inbound_headers": old_headers},
            {"payload": _make_payload(1, span_prefix="b"), "inbound_headers": new_headers},
        ]

        with patch.object(plugin, "_flush_webhook_payloads") as mock_flush:
            plugin._flush_webhook_buffer()
            # Same scope → single merged delivery with the most recent headers.
            mock_flush.assert_called_once()
            args = mock_flush.call_args
            assert len(args[0][0][0]["resourceSpans"]) == 2
            assert args[0][1] == new_headers

    @patch("open_cite.core._WEBHOOK_FLUSH_INTERVAL", 5)
    def test_flush_does_not_merge_across_tenants(self, _mock_validate):
        """Regression: a debounce window spanning two tenants must NOT merge them.

        Merging would forward one tenant's spans under the other tenant's
        headers (x-tenant-id), writing tenant B's traces into tenant A's DB.
        """
        plugin = _make_plugin()
        plugin.subscribe_webhook("http://example.com/traces")

        headers_a = {"x-tenant-id": "tenant-a", "Authorization": "Bearer a"}
        headers_b = {"x-tenant-id": "tenant-b", "Authorization": "Bearer b"}
        plugin._webhook_buffer = [
            {"payload": _make_payload(2, span_prefix="a"), "inbound_headers": headers_a},
            {"payload": _make_payload(3, span_prefix="b"), "inbound_headers": headers_b},
            {"payload": _make_payload(1, span_prefix="a2"), "inbound_headers": headers_a},
        ]

        with patch.object(plugin, "_flush_webhook_payloads") as mock_flush:
            plugin._flush_webhook_buffer()

            # One delivery per tenant scope — never a single merged batch.
            assert mock_flush.call_count == 2
            by_tenant = {
                call[0][1]["x-tenant-id"]: len(call[0][0][0]["resourceSpans"])
                for call in mock_flush.call_args_list
            }
            # tenant-a: 2 + 1 spans merged within its own scope; tenant-b: 3.
            assert by_tenant == {"tenant-a": 3, "tenant-b": 3}
            # Each delivery carries its OWN headers (no cross-contamination).
            for call in mock_flush.call_args_list:
                sent_headers = call[0][1]
                assert sent_headers in (headers_a, headers_b)

    @patch("open_cite.core._WEBHOOK_FLUSH_INTERVAL", 5)
    def test_flush_scope_key_is_case_insensitive(self, _mock_validate):
        """Header casing must not fragment a single tenant into separate scopes."""
        plugin = _make_plugin()
        plugin.subscribe_webhook("http://example.com/traces")

        plugin._webhook_buffer = [
            {"payload": _make_payload(1, span_prefix="a"), "inbound_headers": {"X-Tenant-Id": "t1"}},
            {"payload": _make_payload(1, span_prefix="b"), "inbound_headers": {"x-tenant-id": "t1"}},
        ]

        with patch.object(plugin, "_flush_webhook_payloads") as mock_flush:
            plugin._flush_webhook_buffer()
            # Same tenant despite differing casing → one merged delivery.
            mock_flush.assert_called_once()
            assert len(mock_flush.call_args[0][0][0]["resourceSpans"]) == 2

    @patch("open_cite.core._WEBHOOK_FLUSH_INTERVAL", 5)
    def test_flush_noop_when_buffer_empty(self, _mock_validate):
        """Flushing an empty buffer does nothing."""
        plugin = _make_plugin()
        plugin.subscribe_webhook("http://example.com/traces")

        with patch.object(plugin, "_flush_webhook_payloads") as mock_flush:
            plugin._flush_webhook_buffer()
            mock_flush.assert_not_called()

    @patch("open_cite.core._WEBHOOK_FLUSH_INTERVAL", 5)
    def test_only_one_timer_scheduled(self, _mock_validate):
        """Multiple _deliver_to_webhooks calls should not spawn multiple timers."""
        plugin = _make_plugin()
        plugin.subscribe_webhook("http://example.com/traces")

        with patch.object(plugin, "_flush_webhook_payloads"):
            plugin._deliver_to_webhooks(_make_payload(1))
            first_timer = plugin._webhook_flush_timer
            plugin._deliver_to_webhooks(_make_payload(1))
            second_timer = plugin._webhook_flush_timer
            # Should be the same timer (not replaced)
            assert first_timer is second_timer

        if plugin._webhook_flush_timer:
            plugin._webhook_flush_timer.cancel()


# ---------------------------------------------------------------------------
# 429 cooldown tests
# ---------------------------------------------------------------------------

@patch("open_cite.core.validate_webhook_url", return_value=None)
class TestWebhook429Cooldown:

    @patch("open_cite.core._WEBHOOK_429_COOLDOWN", 30)
    @patch("open_cite.core._WEBHOOK_MAX_RETRIES", 3)
    def test_429_enters_cooldown(self, _mock_validate):
        """A 429 response should put the URL into cooldown and stop retrying."""
        plugin = _make_plugin()
        url = "http://example.com/traces"
        plugin.subscribe_webhook(url)

        mock_resp = _mock_response(429, text='{"error":"rate limited"}')

        with patch("open_cite.core.requests.post", return_value=mock_resp) as mock_post, \
             patch("open_cite.core.validate_webhook_url", return_value=None):
            plugin._send_webhook(url, _make_payload(1))
            # Should only attempt once (429 returns immediately, no retries)
            assert mock_post.call_count == 1

        # URL should be in cooldown
        assert url in plugin._webhook_cooldowns
        assert plugin._webhook_cooldowns[url] > time.time()

    @patch("open_cite.core._WEBHOOK_429_COOLDOWN", 30)
    def test_429_respects_retry_after_header(self, _mock_validate):
        """The Retry-After header value should be used as cooldown duration."""
        plugin = _make_plugin()
        url = "http://example.com/traces"
        plugin.subscribe_webhook(url)

        mock_resp = _mock_response(429, headers={"Retry-After": "60"})

        before = time.time()
        with patch("open_cite.core.requests.post", return_value=mock_resp), \
             patch("open_cite.core.validate_webhook_url", return_value=None):
            plugin._send_webhook(url, _make_payload(1))

        # Cooldown should be ~60s, not the default 30s
        assert plugin._webhook_cooldowns[url] >= before + 59

    @patch("open_cite.core._WEBHOOK_429_COOLDOWN", 30)
    def test_429_invalid_retry_after_uses_default(self, _mock_validate):
        """An unparseable Retry-After header falls back to default cooldown."""
        plugin = _make_plugin()
        url = "http://example.com/traces"
        plugin.subscribe_webhook(url)

        mock_resp = _mock_response(429, headers={"Retry-After": "not-a-number"})

        before = time.time()
        with patch("open_cite.core.requests.post", return_value=mock_resp), \
             patch("open_cite.core.validate_webhook_url", return_value=None):
            plugin._send_webhook(url, _make_payload(1))

        # Should use default 30s
        assert plugin._webhook_cooldowns[url] >= before + 29
        assert plugin._webhook_cooldowns[url] < before + 35

    def test_cooldown_skips_delivery(self, _mock_validate):
        """URLs in cooldown should be skipped during delivery."""
        plugin = _make_plugin()
        url = "http://example.com/traces"
        plugin.subscribe_webhook(url)

        # Put URL in cooldown (30s from now)
        plugin._webhook_cooldowns[url] = time.time() + 30

        with patch.object(plugin, "_send_webhook") as mock_send:
            plugin._flush_webhook_payloads([_make_payload(1)])
            mock_send.assert_not_called()

    def test_delivery_resumes_after_cooldown(self, _mock_validate):
        """URLs should receive deliveries again after cooldown expires."""
        plugin = _make_plugin()
        url = "http://example.com/traces"
        plugin.subscribe_webhook(url)

        # Set cooldown in the past (already expired)
        plugin._webhook_cooldowns[url] = time.time() - 1

        with patch.object(plugin, "_send_webhook") as mock_send:
            plugin._flush_webhook_payloads([_make_payload(1)])
            assert mock_send.call_count == 1

    def test_cooldown_does_not_affect_other_urls(self, _mock_validate):
        """Only the 429'd URL should be in cooldown, not others."""
        plugin = _make_plugin()
        url_a = "http://a.example.com/traces"
        url_b = "http://b.example.com/traces"
        plugin.subscribe_webhook(url_a)
        plugin.subscribe_webhook(url_b)

        # Only URL A is in cooldown
        plugin._webhook_cooldowns[url_a] = time.time() + 30

        with patch.object(plugin, "_send_webhook") as mock_send:
            plugin._flush_webhook_payloads([_make_payload(1)])
            # Only URL B should have been called
            assert mock_send.call_count == 1
            assert mock_send.call_args[0][0] == url_b

    @patch("open_cite.core._WEBHOOK_MAX_RETRIES", 3)
    def test_non_429_errors_still_retry(self, _mock_validate):
        """Non-429 errors should still use the normal retry logic."""
        plugin = _make_plugin()
        url = "http://example.com/traces"
        plugin.subscribe_webhook(url)

        mock_resp = _mock_response(500, text="Internal Server Error")

        with patch("open_cite.core.requests.post", return_value=mock_resp) as mock_post, \
             patch("open_cite.core.validate_webhook_url", return_value=None), \
             patch("time.sleep"):
            plugin._send_webhook(url, _make_payload(1))
            # Should retry all 3 attempts
            assert mock_post.call_count == 3

        # Should NOT enter cooldown
        assert url not in plugin._webhook_cooldowns
