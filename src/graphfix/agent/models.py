"""Provider-neutral text generation and an Ollama-compatible local adapter."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol, cast

AGENT_ACTION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "INSPECT_FILE",
                "INSPECT_FUNCTION",
                "EXPAND_CALLERS",
                "EXPAND_CALLEES",
                "EXPAND_IMPORTS",
                "EXPAND_INHERITANCE",
                "STOP",
            ],
        },
        "node_id": {"type": ["string", "null"]},
    },
    "required": ["action", "node_id"],
    "additionalProperties": False,
}


class TextGenerator(Protocol):
    """Minimal provider-neutral interface required by PromptedController."""

    def __call__(self, prompt: str) -> str:
        """Return generated text for one prompt."""
        ...


@dataclass(frozen=True)
class OllamaConfig:
    """Externally configurable deterministic Ollama generation settings."""

    model: str
    endpoint: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 120.0
    temperature: float = 0.0
    seed: int = 0


class OllamaTextGenerator:
    """Small synchronous adapter for Ollama's local ``/api/generate`` endpoint."""

    provider = "ollama"

    def __init__(self, config: OllamaConfig) -> None:
        """Validate and retain the local Ollama connection settings."""
        if not config.model.strip():
            raise ValueError("Ollama model name must be non-empty")
        if config.timeout_seconds <= 0:
            raise ValueError("Ollama timeout must be positive")
        self.config = config
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        """Send one schema-constrained prompt and return Ollama's response text."""
        payload = json.dumps(
            {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "format": AGENT_ACTION_SCHEMA,
                "options": {
                    "temperature": self.config.temperature,
                    "seed": self.config.seed,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.endpoint.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        self.calls += 1
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                body = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise RuntimeError(
                "Unable to reach Ollama. Start Ollama, ensure the configured model is "
                f"available, and verify endpoint {self.config.endpoint!r}: {error}"
            ) from error
        try:
            decoded = cast(dict[str, object], json.loads(body))
            text = decoded["response"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError("Ollama returned an invalid generation response") from error
        if not isinstance(text, str):
            raise RuntimeError("Ollama generation response is not text")
        return text
