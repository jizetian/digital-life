"""OpenAI-compatible chat client for Ollama and hosted APIs.

The local default is Ollama's OpenAI-compatible endpoint
(``http://127.0.0.1:11434/v1``). Any server that accepts
``POST /chat/completions`` can be plugged in through ``base_url``.
"""

from __future__ import annotations

import json
import os
from typing import Protocol

import httpx

from common.config import distillation_config


class LLMError(Exception):
    """The model endpoint did not return assistant text."""


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str) -> str:
        """Return the assistant message text."""


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120,
        temperature: float = 0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url or not str(base_url).strip():
            raise LLMError("LLM base_url is empty")
        if not model or not str(model).strip():
            raise LLMError("LLM model is empty")
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.api_key = api_key or None
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self._transport = transport

    def complete(self, *, system: str, user: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                response = client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMError("LLM request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise LLMError(f"LLM HTTP {response.status_code}")
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise LLMError("LLM response was not JSON") from exc
        return _message_text(body)


def build_llm_client(
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
    temperature: float | None = None,
    transport: httpx.BaseTransport | None = None,
) -> OpenAICompatibleClient:
    cfg = distillation_config()
    api_key = os.environ.get("CHATTWIN_LLM_API_KEY") or None
    if api_key == "":
        api_key = None
    return OpenAICompatibleClient(
        base_url=base_url or str(cfg.get("base_url") or ""),
        model=model or str(cfg.get("model") or ""),
        api_key=api_key,
        timeout=float(cfg.get("timeout_seconds", 120) if timeout is None else timeout),
        temperature=float(cfg.get("temperature", 0) if temperature is None else temperature),
        transport=transport,
    )


def _message_text(body: dict) -> str:
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("LLM response is missing choices[0].message.content") from exc
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        joined = "".join(parts)
        if joined.strip():
            return joined
    raise LLMError("LLM message content is empty")
