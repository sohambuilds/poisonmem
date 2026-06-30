from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import ValidationError

from .models import ACTION_OUTPUT_SCHEMA, AgentOutput, ExperimentConfig, ModelConfig


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass
class CompletionResult:
    output: AgentOutput | None
    valid_json: bool
    attempts: int
    final_error: str | None
    response_id: str | None
    request_id: str | None
    provider: str | None
    usage: dict[str, Any]
    cost: float | None
    raw_content: str | None


class OpenRouterClient:
    def __init__(self, config: ExperimentConfig, api_key: str) -> None:
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.openrouter.base_url.rstrip("/") + "/",
            timeout=config.openrouter.timeout_seconds,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": config.openrouter.app_url,
                "X-Title": config.openrouter.app_name,
                "Content-Type": "application/json",
            },
        )

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.client.aclose()

    def _payload(self, model: ModelConfig, messages: list[dict[str, str]]) -> dict[str, Any]:
        generation = self.config.generation
        response_mode = self.config.openrouter.response_mode
        response_format: dict[str, Any]
        if response_mode == "json_schema":
            response_format = {"type": "json_schema", "json_schema": ACTION_OUTPUT_SCHEMA}
        else:
            response_format = {"type": "json_object"}
        return {
            "model": model.id,
            "messages": messages,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "max_tokens": generation.max_tokens,
            "response_format": response_format,
            "provider": {
                "only": [model.provider],
                "order": [model.provider],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }

    async def validate_provider(self, model: ModelConfig) -> dict[str, Any]:
        author, slug = model.id.split("/", maxsplit=1)
        response = await self.client.get(f"models/{author}/{slug}/endpoints")
        response.raise_for_status()
        data = response.json().get("data", response.json())
        endpoints = data.get("endpoints", [])

        def endpoint_slug(endpoint: dict[str, Any]) -> str:
            return str(
                endpoint.get("provider_slug")
                or endpoint.get("tag")
                or endpoint.get("provider_name")
                or endpoint.get("name")
                or ""
            ).lower()

        provider = model.provider.lower()
        matched = [endpoint for endpoint in endpoints if endpoint_slug(endpoint) == provider]
        if not matched:
            available = sorted(filter(None, (endpoint_slug(endpoint) for endpoint in endpoints)))
            raise ValueError(
                f"Pinned provider {model.provider!r} not found for {model.id}; available={available}"
            )
        supported = matched[0].get("supported_parameters", [])
        return {
            "model": model.id,
            "pinned_provider": model.provider,
            "matching_endpoints": len(matched),
            "supported_parameters": supported,
            "checked_at": utc_now(),
        }

    async def _generation_stats(self, generation_id: str | None) -> dict[str, Any]:
        if not generation_id or not self.config.openrouter.fetch_generation_stats:
            return {}
        for delay in (0.0, 0.25, 0.75):
            if delay:
                await asyncio.sleep(delay)
            try:
                response = await self.client.get("generation", params={"id": generation_id})
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                return response.json().get("data", {})
            except httpx.HTTPError:
                return {}
        return {}

    async def complete(
        self,
        model: ModelConfig,
        messages: list[dict[str, str]],
        call_id: str,
    ) -> tuple[CompletionResult, list[dict[str, Any]]]:
        payload = self._payload(model, messages)
        prompt_hash = canonical_hash(messages)
        usage_rows: list[dict[str, Any]] = []
        malformed_retries = self.config.generation.malformed_retries
        last_error: str | None = None
        last_content: str | None = None
        last_metadata: dict[str, Any] = {}

        for retry_index in range(malformed_retries + 1):
            started_at = utc_now()
            try:
                response = await self.client.post("chat/completions", json=payload)
                response.raise_for_status()
                body = response.json()
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                detail = getattr(getattr(exc, "response", None), "text", None)
                last_error = f"{type(exc).__name__}: {exc}"
                usage_rows.append(
                    {
                        "call_id": call_id,
                        "attempt": retry_index + 1,
                        "model": model.id,
                        "pinned_provider": model.provider,
                        "provider": None,
                        "request_id": None,
                        "response_id": None,
                        "usage": {},
                        "cost": None,
                        "timestamp": started_at,
                        "prompt_hash": prompt_hash,
                        "valid_json": False,
                        "error": last_error,
                        "error_detail": detail[:1000] if detail else None,
                    }
                )
                break

            generation_id = body.get("id") or response.headers.get("x-generation-id")
            stats = await self._generation_stats(generation_id)
            usage = body.get("usage") or {}
            request_id = (
                response.headers.get("x-request-id")
                or stats.get("request_id")
                or body.get("request_id")
            )
            provider = body.get("provider") or stats.get("provider_name")
            cost_value = usage.get("cost")
            if cost_value is None:
                cost_value = stats.get("total_cost")
            try:
                content = body["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    content = json.dumps(content)
            except (KeyError, IndexError, TypeError) as exc:
                content = ""
                last_error = f"Response shape error: {exc}"
            last_content = content
            valid = False
            parsed: AgentOutput | None = None
            try:
                parsed = AgentOutput.model_validate(json.loads(content))
                valid = True
                last_error = None
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = f"Malformed model output: {exc}"

            usage_row = {
                "call_id": call_id,
                "attempt": retry_index + 1,
                "model": model.id,
                "pinned_provider": model.provider,
                "provider": provider,
                "request_id": request_id,
                "response_id": generation_id,
                "usage": usage,
                "cost": float(cost_value) if cost_value is not None else None,
                "timestamp": started_at,
                "prompt_hash": prompt_hash,
                "valid_json": valid,
                "error": last_error,
            }
            usage_rows.append(usage_row)
            last_metadata = usage_row
            if valid:
                return (
                    CompletionResult(
                        output=parsed,
                        valid_json=True,
                        attempts=retry_index + 1,
                        final_error=None,
                        response_id=generation_id,
                        request_id=request_id,
                        provider=provider,
                        usage=usage,
                        cost=usage_row["cost"],
                        raw_content=content,
                    ),
                    usage_rows,
                )

        return (
            CompletionResult(
                output=None,
                valid_json=False,
                attempts=len(usage_rows),
                final_error=last_error,
                response_id=last_metadata.get("response_id"),
                request_id=last_metadata.get("request_id"),
                provider=last_metadata.get("provider"),
                usage=last_metadata.get("usage", {}),
                cost=last_metadata.get("cost"),
                raw_content=last_content,
            ),
            usage_rows,
        )
