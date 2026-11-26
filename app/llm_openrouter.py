"""Hosted chat + embeddings through OpenRouter (OpenAI-compatible API).

Same contract as app/llm.py (docs/contracts.md):
    chat(messages, schema=None, max_tokens=512, temperature=0.2) -> str
    chat_stream(messages, max_tokens=512, temperature=0.4) -> Iterator[str]
    embed(texts) -> np.ndarray            # L2-normalised float32, shape (n, dim)
    unload() -> None                      # no-op (nothing is held in memory)

Env: OPENROUTER_API_KEY (required), OPENROUTER_CHAT_MODEL, OPENROUTER_EMBED_MODEL.
The key is only ever placed in the Authorization header; it is never logged.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Iterator

import httpx
import numpy as np

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CHAT_MODEL = "qwen/qwen3.5-9b"
DEFAULT_EMBED_MODEL = "baai/bge-base-en-v1.5"
TIMEOUT = httpx.Timeout(60.0, connect=10.0)

_client: httpx.Client | None = None
_client_lock = threading.Lock()
_last_cost = threading.local()


class ProviderError(RuntimeError):
    """The hosted provider failed (network, HTTP error, or unusable output)."""


def chat_model_name() -> str:
    return os.environ.get("OPENROUTER_CHAT_MODEL") or DEFAULT_CHAT_MODEL


def embed_model_name() -> str:
    return os.environ.get("OPENROUTER_EMBED_MODEL") or DEFAULT_EMBED_MODEL


def _headers() -> dict:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise ProviderError("OPENROUTER_API_KEY is not set")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-Title": "Fitting Room",
    }


def client() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)
        return _client


def post(path: str, payload: dict, timeout: float | None = None) -> dict:
    """POST JSON and return the decoded body; raise ProviderError on any failure."""
    try:
        r = client().post(path, json=payload, headers=_headers(),
                          timeout=timeout if timeout is not None else TIMEOUT)
    except httpx.HTTPError as e:
        raise ProviderError(f"OpenRouter request failed: {type(e).__name__}") from None
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code >= 400 or "error" in body:
        msg = (body.get("error") or {}).get("message") if isinstance(body.get("error"), dict) else None
        raise ProviderError(f"OpenRouter HTTP {r.status_code}: {msg or 'error'}")
    return body


def last_cost() -> float:
    """Cost in USD of the last call on this thread (0.0 if unknown)."""
    return getattr(_last_cost, "value", 0.0)


# ------------------------------------------------------------------ JSON checks
def _type_ok(value, t) -> bool:
    types = t if isinstance(t, list) else [t]
    for x in types:
        if x == "null" and value is None:
            return True
        if x == "string" and isinstance(value, str):
            return True
        if x == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if x == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if x == "boolean" and isinstance(value, bool):
            return True
        if x == "array" and isinstance(value, list):
            return True
        if x == "object" and isinstance(value, dict):
            return True
    return False


def validate(value, schema: dict, path: str = "$") -> None:
    """Check the subset of JSON Schema this app uses; raise ValueError on mismatch."""
    if "type" in schema and not _type_ok(value, schema["type"]):
        raise ValueError(f"{path}: expected {schema['type']}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: {value!r} not in enum")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                raise ValueError(f"{path}: missing {req}")
        for k, v in value.items():
            if k in props:
                validate(v, props[k], f"{path}.{k}")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            validate(v, schema["items"], f"{path}[{i}]")


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


# ------------------------------------------------------------------ chat
# Providers that ignore "reasoning: disabled" for the default model and spend the whole
# token budget thinking (content comes back empty). Tested 2026-11-26 on qwen/qwen3.5-9b.
DEFAULT_IGNORE = "PrimeIntellect"


def _provider_prefs(**extra) -> dict:
    ignore = [p.strip() for p in os.environ.get("OPENROUTER_IGNORE_PROVIDERS", DEFAULT_IGNORE).split(",") if p.strip()]
    prefs = {"ignore": ignore} if ignore else {}
    prefs.update(extra)
    return prefs


def _payload(messages, max_tokens, temperature, **extra) -> dict:
    payload = {
        "model": chat_model_name(),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Direct answers, no "thinking" text.
        "reasoning": {"enabled": False},
        "usage": {"include": True},
        "provider": _provider_prefs(),
    }
    payload.update(extra)
    return payload


def _strip_empty(value):
    """Drop null and [] values (recursively), matching app/llm.py in local mode."""
    if isinstance(value, dict):
        return {k: _strip_empty(v) for k, v in value.items() if v is not None and v != []}
    if isinstance(value, list):
        return [_strip_empty(v) for v in value]
    return value


def chat(messages: list[dict], schema: dict | None = None, max_tokens: int = 512,
         temperature: float = 0.2) -> str:
    """Single completion. With `schema`, returns a JSON string that parses and matches it
    (then, like app/llm.py, with null and [] values removed)."""
    extra = {}
    if schema is not None:
        extra["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema.get("title", "output"), "strict": True, "schema": schema},
        }
        # Only route to providers that honour response_format.
        extra["provider"] = _provider_prefs(require_parameters=True)
    last_err: Exception | None = None
    for _attempt in range(2):  # retry once on unusable output
        body = post("/chat/completions", _payload(messages, max_tokens, temperature, **extra))
        _last_cost.value = float((body.get("usage") or {}).get("cost") or 0.0)
        try:
            text = (body["choices"][0]["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError):
            last_err = ProviderError("OpenRouter returned no choices")
            continue
        if schema is None:
            if text:
                return text
            last_err = ProviderError("empty completion")
            continue
        try:
            text = _strip_fences(text)
            value = json.loads(text)
            validate(value, schema)
            return json.dumps(_strip_empty(value))
        except ValueError as e:
            last_err = e
    raise ProviderError(f"model output unusable after retry: {last_err}")


def chat_stream(messages: list[dict], max_tokens: int = 512,
                temperature: float = 0.4) -> Iterator[str]:
    """Yield text deltas from a streamed completion (SSE). Retries once (before any text was
    yielded) if the provider returned no content, e.g. it spent the budget on reasoning."""
    for attempt in range(2):
        got = False
        for delta in _stream_once(messages, max_tokens, temperature):
            got = True
            yield delta
        if got:
            return
    raise ProviderError("empty streamed completion")


def _stream_once(messages, max_tokens, temperature) -> Iterator[str]:
    payload = _payload(messages, max_tokens, temperature, stream=True)
    try:
        with client().stream("POST", "/chat/completions", json=payload, headers=_headers()) as r:
            if r.status_code >= 400:
                r.read()
                raise ProviderError(f"OpenRouter HTTP {r.status_code}")
            for line in r.iter_lines():
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if chunk.get("error"):
                    raise ProviderError("OpenRouter stream error")
                if chunk.get("usage"):
                    _last_cost.value = float(chunk["usage"].get("cost") or 0.0)
                for ch in chunk.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content")
                    if delta:
                        yield delta
    except httpx.HTTPError as e:
        raise ProviderError(f"OpenRouter stream failed: {type(e).__name__}") from None


# ------------------------------------------------------------------ embeddings
def embed(texts: list[str]) -> np.ndarray:
    """Embed texts; rows are L2-normalised float32."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    body = post("/embeddings", {"model": embed_model_name(), "input": list(texts)})
    rows = sorted(body.get("data") or [], key=lambda d: d.get("index", 0))
    if len(rows) != len(texts):
        raise ProviderError("embedding count mismatch")
    arr = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def unload() -> None:
    """Nothing to free for a hosted provider."""
    return None
