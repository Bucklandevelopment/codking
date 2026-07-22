"""
Tests for codking's vital-sdk integration (VitalClient + VitalConfig).

The codking SDK follows the same pattern as canela-molida's VitalClient
but with codking-specific defaults (service_name="codking", port=8000)
and codking-specific event types for security/AI analysis.

All external dependencies (httpx, Redis) are mocked so these tests
run without vital-core, Redis, or any other infrastructure.
"""

import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_vital_env_vars(monkeypatch):
    """Ensure VITAL_ env vars do not bleed across tests."""
    for key in list(os.environ):
        if key.startswith("VITAL_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_httpx_response_ok():
    """Create a mock httpx response with status 200."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {"status": "ok", "event_id": "evt-ck-001"}
    response.text = '{"status": "ok"}'
    response.raise_for_status = MagicMock()
    return response


@pytest.fixture
def mock_httpx_response_error():
    """Create a mock httpx response with status 500."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 500
    response.json.return_value = {"error": "internal server error"}
    response.text = '{"error": "internal server error"}'
    response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=response)
    )
    return response


@pytest.fixture
def mock_httpx_client(mock_httpx_response_ok):
    """Patch httpx.AsyncClient to return a fully mocked async client."""
    with patch("integrations.vital_sdk.client.httpx.AsyncClient") as MockClientClass:
        client_instance = AsyncMock()
        client_instance.post = AsyncMock(return_value=mock_httpx_response_ok)
        client_instance.get = AsyncMock(return_value=mock_httpx_response_ok)
        client_instance.aclose = AsyncMock()
        MockClientClass.return_value = client_instance
        yield client_instance


@pytest.fixture
def mock_redis():
    """Patch redis.asyncio.from_url to return a mocked Redis client."""
    with patch("redis.asyncio.from_url") as mock_from_url:
        redis_client = AsyncMock()
        redis_client.ping = AsyncMock()
        redis_client.publish = AsyncMock(return_value=1)
        redis_client.close = AsyncMock()
        redis_client.pubsub.return_value = AsyncMock(
            subscribe=AsyncMock(),
            unsubscribe=AsyncMock(),
            close=AsyncMock(),
            listen=AsyncMock(return_value=iter([])),
        )
        mock_from_url.return_value = redis_client
        yield redis_client


# ---------------------------------------------------------------------------
# Tests: VitalConfig
# ---------------------------------------------------------------------------

class TestVitalConfig:
    """Tests for VitalConfig with codking defaults."""

    def test_codking_defaults(self):
        from integrations.vital_sdk.config import VitalConfig

        config = VitalConfig()
        assert config.core_url == "http://localhost:8888"
        assert config.api_key == ""
        assert config.service_name == "codking"
        assert config.service_port == 8000
        assert config.redis_url == "redis://localhost:6379"

    def test_env_override_core_url(self, monkeypatch):
        monkeypatch.setenv("VITAL_CORE_URL", "http://vital-core:9999")
        from integrations.vital_sdk.config import VitalConfig

        config = VitalConfig()
        assert config.core_url == "http://vital-core:9999"

    def test_env_override_service_name(self, monkeypatch):
        monkeypatch.setenv("VITAL_SERVICE_NAME", "codking-dev")
        from integrations.vital_sdk.config import VitalConfig

        config = VitalConfig()
        assert config.service_name == "codking-dev"

    def test_env_override_api_key(self, monkeypatch):
        monkeypatch.setenv("VITAL_API_KEY", "ck-secret-123")
        from integrations.vital_sdk.config import VitalConfig

        config = VitalConfig()
        assert config.api_key == "ck-secret-123"

    def test_env_override_service_port(self, monkeypatch):
        monkeypatch.setenv("VITAL_SERVICE_PORT", "9090")
        from integrations.vital_sdk.config import VitalConfig

        config = VitalConfig()
        assert config.service_port == 9090

    def test_env_override_redis_url(self, monkeypatch):
        monkeypatch.setenv("VITAL_REDIS_URL", "redis://redis-cluster:6380")
        from integrations.vital_sdk.config import VitalConfig

        config = VitalConfig()
        assert config.redis_url == "redis://redis-cluster:6380"


# ---------------------------------------------------------------------------
# Tests: VitalClient Instantiation
# ---------------------------------------------------------------------------

class TestVitalClientInstantiation:
    """Tests for VitalClient construction."""

    def test_instantiation_with_default_config(self, mock_httpx_client):
        from integrations.vital_sdk.client import VitalClient
        from integrations.vital_sdk.config import VitalConfig

        config = VitalConfig()
        client = VitalClient(config=config)

        assert client.config.service_name == "codking"
        assert client.config.service_port == 8000
        assert client.config.core_url == "http://localhost:8888"

    def test_instantiation_without_config_uses_defaults(self, mock_httpx_client):
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()

        assert client.config is not None
        assert client.config.service_name == "codking"
        assert client.config.service_port == 8000

    def test_initial_state(self, mock_httpx_client):
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()

        assert client._redis is None
        assert client._pubsub is None
        assert client._listener_task is None
        assert client._heartbeat_task is None
        assert client._subscriptions == {}


# ---------------------------------------------------------------------------
# Tests: Lifecycle (connect / disconnect)
# ---------------------------------------------------------------------------

class TestLifecycle:
    """Tests for connect() and disconnect() lifecycle methods."""

    @pytest.mark.asyncio
    async def test_connect_calls_health_check(self, mock_httpx_client):
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        await client.connect()

        mock_httpx_client.get.assert_called_once_with("/api/v1/health")
        assert client._heartbeat_task is not None
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_cancels_heartbeat(self, mock_httpx_client):
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        await client.connect()
        assert client._heartbeat_task is not None

        await client.disconnect()

        assert client._heartbeat_task is None
        mock_httpx_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_closes_redis(self, mock_httpx_client, mock_redis):
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        await client.connect()

        # Simulate Redis being connected
        client._redis = mock_redis
        client._pubsub = AsyncMock(
            unsubscribe=AsyncMock(), close=AsyncMock()
        )

        await client.disconnect()

        mock_redis.close.assert_awaited_once()
        assert client._redis is None
        assert client._pubsub is None

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_httpx_client):
        from integrations.vital_sdk.client import VitalClient

        async with VitalClient() as client:
            assert client._heartbeat_task is not None

        mock_httpx_client.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests: register()
# ---------------------------------------------------------------------------

class TestRegister:
    """Tests for register() method with codking-specific capabilities."""

    @pytest.mark.asyncio
    async def test_register_sends_codking_payload(self, mock_httpx_client):
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        await client.connect()

        mock_httpx_client.post.reset_mock()
        result = await client.register()

        assert result is not None
        call_args = mock_httpx_client.post.call_args
        body = call_args[1]["json"]

        assert body["category"] == "system"
        assert body["action"] == "service.registered"
        assert body["source"] == "codking"
        assert body["event_type"] == "system"
        assert body["payload"]["service_name"] == "codking"
        assert body["payload"]["service_port"] == 8000
        assert body["subcategory"] == "sync"
        # Codking-specific capabilities
        assert "threat-detection" in body["payload"]["capabilities"]
        assert "batch-analysis" in body["payload"]["capabilities"]
        assert "text-classification" in body["payload"]["capabilities"]
        assert "log-analysis" in body["payload"]["capabilities"]
        assert "cybersecurity-ai" in body["payload"]["capabilities"]

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_register_returns_none_on_failure(self, mock_httpx_client):
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        await client.connect()

        mock_httpx_client.post.side_effect = httpx.ConnectError("Connection refused")
        result = await client.register()

        assert result is None
        await client.disconnect()


# ---------------------------------------------------------------------------
# Tests: publish_event
# ---------------------------------------------------------------------------

class TestPublishEvent:
    """Tests for publish_event() with codking-specific security events."""

    @pytest.mark.asyncio
    async def test_publish_security_event(self, mock_httpx_client):
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        await client.connect()
        mock_httpx_client.post.reset_mock()

        result = await client.publish_event(
            category="security",
            action="analyze",
            payload={"threat_level": "high", "target": "malware.exe"},
            event_type="threat.detected",
            metadata={"model": "hrm-v1"},
            tags=["malware", "critical"],
        )

        assert result is not None
        body = mock_httpx_client.post.call_args[1]["json"]

        assert body["category"] == "security"
        assert body["source"] == "codking"
        assert body["action"] == "analyze"
        assert body["event_type"] == "threat.detected"
        assert body["payload"]["threat_level"] == "high"
        assert body["metadata"] == {"model": "hrm-v1"}
        assert body["tags"] == ["malware", "critical"]

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_publish_ai_classification_event(self, mock_httpx_client):
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        await client.connect()
        mock_httpx_client.post.reset_mock()

        result = await client.publish_event(
            category="ai",
            action="classify",
            payload={"input": "log entry", "prediction": "anomaly", "confidence": 0.95},
            event_type="classification.completed",
        )

        assert result is not None
        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["category"] == "ai"
        assert body["event_type"] == "classification.completed"
        assert body["payload"]["confidence"] == 0.95

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_publish_event_with_defaults(self, mock_httpx_client):
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        await client.connect()
        mock_httpx_client.post.reset_mock()

        result = await client.publish_event(
            category="security",
            action="scan.completed",
        )

        assert result is not None
        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["event_type"] == "sdk"
        assert body["payload"] == {}
        assert body["metadata"] == {}
        assert body["tags"] == []
        assert body["subcategory"] is None

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_publish_event_returns_none_on_exception(self, mock_httpx_client):
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        await client.connect()
        mock_httpx_client.post.side_effect = httpx.ConnectError("down")

        result = await client.publish_event(
            category="security",
            action="analyze",
        )

        assert result is None
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_publish_event_returns_none_on_http_error(self, mock_httpx_client, mock_httpx_response_error):
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        await client.connect()
        mock_httpx_client.post.return_value = mock_httpx_response_error

        result = await client.publish_event(
            category="security",
            action="analyze",
        )

        assert result is None
        await client.disconnect()


# ---------------------------------------------------------------------------
# Tests: heartbeat
# ---------------------------------------------------------------------------

class TestHeartbeat:
    """Tests for heartbeat() method."""

    @pytest.mark.asyncio
    async def test_heartbeat_sends_correct_payload(self, mock_httpx_client):
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        await client.connect()
        mock_httpx_client.post.reset_mock()

        result = await client.heartbeat()

        assert result is not None
        body = mock_httpx_client.post.call_args[1]["json"]
        assert body["category"] == "system"
        assert body["action"] == "service.heartbeat"
        assert body["payload"]["service_name"] == "codking"
        assert body["event_type"] == "system"
        assert body["subcategory"] == "sync"

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_heartbeat_returns_none_on_failure(self, mock_httpx_client):
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        await client.connect()
        mock_httpx_client.post.side_effect = httpx.ConnectError("timeout")

        result = await client.heartbeat()

        assert result is None
        await client.disconnect()


# ---------------------------------------------------------------------------
# Tests: health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    """Tests for health_check() method."""

    @pytest.mark.asyncio
    async def test_health_check_returns_response(self, mock_httpx_client):
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        client._http = mock_httpx_client

        result = await client.health_check()

        mock_httpx_client.get.assert_called_with("/api/v1/health")
        assert result == {"status": "ok", "event_id": "evt-ck-001"}

    @pytest.mark.asyncio
    async def test_health_check_raises_on_error(self, mock_httpx_client, mock_httpx_response_error):
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        client._http = mock_httpx_client
        mock_httpx_client.get.return_value = mock_httpx_response_error

        with pytest.raises(httpx.HTTPStatusError):
            await client.health_check()


# ---------------------------------------------------------------------------
# Tests: create_event helper
# ---------------------------------------------------------------------------

class TestCreateEvent:
    """Tests for the create_event() helper function."""

    def test_create_event_with_all_params(self):
        from integrations.vital_sdk.events import create_event

        event = create_event(
            category="security",
            action="threat.detected",
            source="codking",
            event_type="threat.detected",
            payload={"severity": "critical"},
            metadata={"model_version": "1.0"},
            tags=["malware", "critical"],
            subcategory="threat-intel",
        )

        assert event["category"] == "security"
        assert event["action"] == "threat.detected"
        assert event["source"] == "codking"
        assert event["event_type"] == "threat.detected"
        assert event["payload"] == {"severity": "critical"}
        assert event["metadata"] == {"model_version": "1.0"}
        assert event["tags"] == ["malware", "critical"]
        assert event["subcategory"] == "threat-intel"

    def test_create_event_defaults(self):
        from integrations.vital_sdk.events import create_event

        event = create_event(
            category="ai",
            action="classify",
            source="codking",
        )

        assert event["event_type"] == "sdk"
        assert event["payload"] == {}
        assert event["metadata"] == {}
        assert event["tags"] == []
        assert event["subcategory"] is None

    def test_create_event_returns_dict(self):
        from integrations.vital_sdk.events import create_event

        event = create_event(
            category="security",
            action="analyze",
            source="codking",
        )
        assert isinstance(event, dict)

    def test_create_event_with_codking_event_types(self):
        """Test that codking-specific event types work correctly."""
        from integrations.vital_sdk.events import create_event, CODKING_EVENT_TYPES

        for event_type, expected_category in CODKING_EVENT_TYPES.items():
            event = create_event(
                category=expected_category,
                action="process",
                source="codking",
                event_type=event_type,
                payload={"test": True},
            )
            assert event["event_type"] == event_type
            assert event["category"] == expected_category


# ---------------------------------------------------------------------------
# Tests: EventCategory and EventBusChannel
# ---------------------------------------------------------------------------

class TestEventConstants:
    """Tests for EventCategory, EventBusChannel, and codking event types."""

    def test_event_category_constants(self):
        from integrations.vital_sdk.events import EventCategory

        assert EventCategory.HEALTH == "health"
        assert EventCategory.EDUCATION == "education"
        assert EventCategory.IDENTITY == "identity"
        assert EventCategory.SECURITY == "security"
        assert EventCategory.AI == "ai"
        assert EventCategory.SYSTEM == "system"

    def test_event_bus_channel_constants(self):
        from integrations.vital_sdk.events import EventBusChannel

        assert EventBusChannel.HEALTH == "vital.health"
        assert EventBusChannel.EDUCATION == "vital.education"
        assert EventBusChannel.IDENTITY == "vital.identity"
        assert EventBusChannel.SECURITY == "vital.security"
        assert EventBusChannel.SYSTEM == "vital.system"
        assert EventBusChannel.AI == "vital.ai"
        assert EventBusChannel.ENERGY == "vital.energy"

    def test_event_bus_channel_for_category(self):
        from integrations.vital_sdk.events import EventBusChannel

        assert EventBusChannel.for_category("security") == "vital.security"
        assert EventBusChannel.for_category("ai") == "vital.ai"
        assert EventBusChannel.for_category("system") == "vital.system"
        assert EventBusChannel.for_category("custom") == "vital.custom"

    def test_codking_event_types_mapping(self):
        from integrations.vital_sdk.events import CODKING_EVENT_TYPES, EventCategory

        assert CODKING_EVENT_TYPES["threat.detected"] == EventCategory.SECURITY
        assert CODKING_EVENT_TYPES["threat.batch_analyzed"] == EventCategory.SECURITY
        assert CODKING_EVENT_TYPES["classification.completed"] == EventCategory.AI
        assert CODKING_EVENT_TYPES["log_file.analyzed"] == EventCategory.SECURITY
        assert CODKING_EVENT_TYPES["model.loaded"] == EventCategory.AI


# ---------------------------------------------------------------------------
# Tests: Graceful Degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """Tests ensuring the SDK degrades gracefully when vital-core is unreachable."""

    @pytest.mark.asyncio
    async def test_connect_survives_health_check_failure(self, mock_httpx_client):
        """connect() should not raise if vital-core is unreachable."""
        mock_httpx_client.get.side_effect = httpx.ConnectError("Connection refused")

        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        await client.connect()

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_publish_event_survives_connection_error(self, mock_httpx_client):
        """publish_event should return None instead of raising."""
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        await client.connect()

        mock_httpx_client.post.side_effect = httpx.ConnectError("Connection refused")
        result = await client.publish_event(
            category="security",
            action="analyze",
        )

        assert result is None
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_register_survives_connection_error(self, mock_httpx_client):
        """register() should return None instead of raising."""
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        await client.connect()

        mock_httpx_client.post.side_effect = httpx.ConnectError("Connection refused")
        result = await client.register()

        assert result is None
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_heartbeat_survives_connection_error(self, mock_httpx_client):
        """heartbeat() should return None instead of raising."""
        from integrations.vital_sdk.client import VitalClient

        client = VitalClient()
        await client.connect()

        mock_httpx_client.post.side_effect = httpx.ConnectError("Connection refused")
        result = await client.heartbeat()

        assert result is None
        await client.disconnect()


# ---------------------------------------------------------------------------
# Tests: Module Exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Tests for the __init__.py public API."""

    def test_public_exports(self):
        from integrations.vital_sdk import (
            VitalClient,
            VitalConfig,
            EventCategory,
            EventBusChannel,
            create_event,
        )

        assert VitalClient is not None
        assert VitalConfig is not None
        assert EventCategory is not None
        assert EventBusChannel is not None
        assert create_event is not None
