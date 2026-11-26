"""Single entry point for the chat/embedding provider. The rest of the app imports this.

    LLM_PROVIDER = local (default) | openrouter
    Local is the default even when OPENROUTER_API_KEY is set; hosted only when asked for.

Re-exports chat, chat_stream, embed, unload with the app/llm.py contract, plus:
    provider() -> "openrouter" | "local"
    chat_model_name(), embed_model_name()
    search_embed() -> embed callable for catalog.search, or None when the stored
                      catalog vectors were made by a different embedding model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMBED_META = ROOT / "data" / "catalog" / "embeddings_meta.json"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines; existing environment variables win."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()


def provider() -> str:
    p = os.environ.get("LLM_PROVIDER", "").strip().lower()
    return "openrouter" if p == "openrouter" else "local"


def _impl():
    if provider() == "openrouter":
        from app import llm_openrouter as impl
    else:
        from app import llm as impl
    return impl


def chat(messages, schema=None, max_tokens=512, temperature=0.2):
    return _impl().chat(messages, schema=schema, max_tokens=max_tokens, temperature=temperature)


def chat_stream(messages, max_tokens=512, temperature=0.4):
    return _impl().chat_stream(messages, max_tokens=max_tokens, temperature=temperature)


def embed(texts):
    return _impl().embed(texts)


def unload():
    return _impl().unload()


def chat_model_name() -> str:
    try:
        return _impl().chat_model_name()
    except Exception:
        return ""


def embed_model_name() -> str:
    """Provider-qualified id of the active embedding model, e.g. 'openrouter:baai/bge-base-en-v1.5'."""
    if provider() == "openrouter":
        from app import llm_openrouter
        return "openrouter:" + llm_openrouter.embed_model_name()
    from app import llm
    try:
        return "local:" + llm._find("LLM_EMBED_MODEL", llm.EMBED_DIR).stem
    except Exception:
        return "local:unknown"


def search_embed():
    """The embed callable for catalog.search, only if the stored vectors match the active model.

    Vectors from different models must never be compared; on a mismatch (or no
    meta file) catalog search falls back to keyword ranking.
    """
    try:
        meta = json.loads(EMBED_META.read_text())
    except (OSError, ValueError):
        return None
    if meta.get("model") != embed_model_name():
        return None
    return embed
