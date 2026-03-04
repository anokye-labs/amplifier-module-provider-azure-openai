# pyright: reportMissingImports=false
"""Error translation tests for Azure OpenAI provider.

Verifies that the Azure OpenAI provider inherits Cloudflare 403 detection
from the OpenAI provider. The Azure provider does NOT override _do_complete()
or the error handler — it inherits them via dynamic subclassing of OpenAIProvider.

These tests confirm that inheritance works correctly: a 403 with body=None
(CDN/proxy challenge) produces ProviderUnavailableError, while a 403 with
a dict body (real API denial) produces AccessDeniedError.
"""

import asyncio
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_azure_openai import _create_azure_provider
from amplifier_module_provider_openai import OpenAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_azure_provider(**config_overrides):
    """Create an Azure OpenAI provider with retries disabled."""
    config = {"max_retries": 0, **config_overrides}
    return _create_azure_provider(
        OpenAIProvider,
        base_url="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
        config=config,
    )


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _mock_httpx_response(
    status_code: int = 403, headers: dict | None = None
) -> httpx.Response:
    """Build a minimal httpx.Response for OpenAI SDK error constructors."""
    return httpx.Response(
        status_code=status_code,
        headers=headers or {},
        request=httpx.Request(
            "POST", "https://example.openai.azure.com/openai/v1/responses"
        ),
    )


# ---------------------------------------------------------------------------
# Cloudflare 403 detection (inherited from OpenAI provider)
# ---------------------------------------------------------------------------


def test_403_with_none_body_raises_provider_unavailable():
    """403 with body=None + text/html (CDN challenge) -> ProviderUnavailableError (retryable).

    This verifies that the Azure OpenAI provider inherits the Cloudflare 403
    detection from the OpenAI provider without needing its own error handler.
    """
    provider = _make_azure_provider()
    native = openai.APIStatusError(
        "Forbidden",
        response=_mock_httpx_response(403, headers={"content-type": "text/html"}),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ProviderUnavailableError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert err.provider == "azure-openai"
    assert err.status_code == 403
    assert err.retryable is True
    assert err.__cause__ is native


def test_403_with_dict_body_raises_access_denied():
    """403 with body=dict (real API denial) -> AccessDeniedError (not retryable)."""
    provider = _make_azure_provider()
    native = openai.APIStatusError(
        "Forbidden",
        response=_mock_httpx_response(403),
        body={"error": {"type": "permissions_error", "message": "Access denied"}},
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.AccessDeniedError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert err.provider == "azure-openai"
    assert err.status_code == 403
    assert err.retryable is False
    # Verify it's NOT a ProviderUnavailableError
    assert not isinstance(err, kernel_errors.ProviderUnavailableError)
