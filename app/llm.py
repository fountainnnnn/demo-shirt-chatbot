"""Local chat + embedding models, in-process via llama-cpp-python (Metal when available).

Contract (docs/contracts.md):
    chat(messages, schema=None, max_tokens=512, temperature=0.2) -> str
    chat_stream(messages, max_tokens=512, temperature=0.4) -> Iterator[str]
    embed(texts) -> np.ndarray            # L2-normalised float32, shape (n, dim)
    unload() -> None                      # frees the chat model

Models are GGUF files fetched by scripts/fetch_models.py into models/chat and
models/embed. Both load lazily on first use. Override the paths with
LLM_CHAT_MODEL / LLM_EMBED_MODEL, and force CPU with LLM_CPU=1.

Import this module before anything else imports llama_cpp, so the Metal/CPU
choice (see _choose_backend) can take effect.
"""

from __future__ import annotations

import atexit
import gc
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHAT_DIR = ROOT / "models" / "chat"
EMBED_DIR = ROOT / "models" / "embed"
N_CTX = 4096

_lock = threading.RLock()
_chat = None          # llama_cpp.Llama
_template = None      # compiled jinja2 chat template
_embedder = None      # llama_cpp.Llama (embedding=True)


def _find(env: str, folder: Path) -> Path:
    if os.environ.get(env):
        return Path(os.environ[env])
    files = sorted(folder.glob("*.gguf"))
    if not files:
        raise FileNotFoundError(f"No .gguf in {folder}; run: .venv/bin/python scripts/fetch_models.py")
    return files[0]


_PROBE = """
import sys
from llama_cpp import Llama
Llama(model_path=sys.argv[1], n_gpu_layers=-1, embedding=True, n_ctx=64, verbose=False).close()
"""


def _choose_backend() -> None:
    """Decide GPU vs CPU before llama_cpp is first imported.

    llama.cpp builds a Metal context even with n_gpu_layers=0, so when Metal is
    present but unusable (sandbox, headless) every load fails. The only clean
    switch is GGML_METAL_DEVICES=0, read once at import time. So: probe Metal in
    a subprocess with the tiny embedding model, and disable it if the probe fails.
    """
    if "llama_cpp" in sys.modules or "GGML_METAL_DEVICES" in os.environ or sys.platform != "darwin":
        return
    if os.environ.get("LLM_CPU") == "1":
        os.environ["GGML_METAL_DEVICES"] = "0"
        return
    try:
        probe_model = _find("LLM_EMBED_MODEL", EMBED_DIR)
        ok = subprocess.run([sys.executable, "-c", _PROBE, str(probe_model)],
                            capture_output=True, timeout=120).returncode == 0
    except Exception:
        ok = False
    if not ok:
        os.environ["GGML_METAL_DEVICES"] = "0"


def _llama_cpp():
    """The only way this module imports llama_cpp: backend choice must come first."""
    _choose_backend()
    import llama_cpp
    return llama_cpp


def device() -> str:
    """'metal' or 'cpu' (probes Metal once if no model is loaded yet)."""
    return "metal" if _llama_cpp().llama_supports_gpu_offload() else "cpu"


def _load(path: Path, **kw):
    """Load with every layer on the GPU when Metal works, else on the CPU."""
    llama_cpp = _llama_cpp()
    layers = -1 if llama_cpp.llama_supports_gpu_offload() else 0
    return llama_cpp.Llama(model_path=str(path), n_gpu_layers=layers, verbose=False, **kw)


def chat_model_name() -> str:
    """File stem of the chat model (for /api/status); does not load it."""
    try:
        return _find("LLM_CHAT_MODEL", CHAT_DIR).stem
    except FileNotFoundError:
        return ""


def _get_chat():
    global _chat, _template
    if _chat is None:
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        llm = _load(_find("LLM_CHAT_MODEL", CHAT_DIR), n_ctx=N_CTX, n_batch=512)
        src = llm.metadata.get("tokenizer.chat_template")
        if not src:
            raise RuntimeError("chat model has no embedded chat template")

        def raise_exception(msg):
            raise jinja2.TemplateError(msg)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            loader=jinja2.BaseLoader())
        env.globals["raise_exception"] = raise_exception
        env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False, **k)
        _template = env.from_string(src)
        _chat = llm
    return _chat


def _prompt(messages: list[dict]) -> str:
    """Render the model's own chat template.

    Qwen3-4B-Instruct-2507 has no thinking mode. enable_thinking=False is passed
    anyway so a hybrid-thinking model dropped in via LLM_CHAT_MODEL (Qwen3/3.5)
    answers directly too; their templates honour it.
    """
    return _template.render(messages=messages, add_generation_prompt=True,
                            enable_thinking=False, bos_token="", eos_token="")


_STOP = ["<|im_end|>", "<|endoftext|>", "<end_of_turn>"]


def _sampling(temperature: float) -> dict:
    # Qwen's suggested non-thinking settings, with the caller's temperature.
    return dict(temperature=temperature, top_p=0.8, top_k=20, min_p=0.0,
                repeat_penalty=1.0, stop=_STOP)


def _nullable_optionals(schema: dict) -> tuple[dict, list[str]]:
    """Rewrite a top-level object schema so every optional property is required but nullable.

    llama.cpp's schema grammar accepts object properties only in schema order,
    so a model that writes a later optional key first can no longer emit the
    earlier ones and silently drops them. With every key present (null = "not
    stated") the order is fixed and nothing gets skipped. The nulls are removed
    again before returning, so the caller still gets JSON valid for its schema.
    Returns (grammar_schema, optional_keys).
    """
    props = schema.get("properties") if schema.get("type") == "object" else None
    if not props:
        return schema, []
    required = set(schema.get("required") or [])
    optional = [k for k in props if k not in required]
    if not optional:
        return schema, []
    new_props = {k: (v if k in required else {"anyOf": [v, {"type": "null"}]}) for k, v in props.items()}
    return {**schema, "properties": new_props, "required": list(props)}, optional


def _with_null_hint(messages: list[dict], optional: list[str]) -> list[dict]:
    note = ("Reply with one JSON object. Use null for any of these keys the conversation "
            "does not state or clearly imply (null, not [] or 0); do not guess: "
            + ", ".join(optional) + ".")
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = f"{msgs[0]['content']}\n\n{note}"
    else:
        msgs.insert(0, {"role": "system", "content": note})
    return msgs


def chat(messages: list[dict], schema: dict | None = None, max_tokens: int = 512,
         temperature: float = 0.2) -> str:
    """Single completion. With `schema`, output is grammar-constrained JSON matching it."""
    LlamaGrammar = _llama_cpp().LlamaGrammar

    optional: list[str] = []
    with _lock:
        llm = _get_chat()
        kw = _sampling(temperature)
        if schema is not None:
            grammar_schema, optional = _nullable_optionals(schema)
            kw["grammar"] = LlamaGrammar.from_json_schema(json.dumps(grammar_schema), verbose=False)
            if optional:
                messages = _with_null_hint(messages, optional)
        out = llm.create_completion(_prompt(messages), max_tokens=max_tokens, **kw)
        text = out["choices"][0]["text"].strip()
    if schema is None:
        return text
    data = json.loads(text)  # raises on truncated output (max_tokens too small)
    if optional:  # null / [] on an optional key means "not stated": leave the key out
        data = {k: v for k, v in data.items() if not (k in optional and v in (None, []))}
    return json.dumps(data, ensure_ascii=False)


def chat_stream(messages: list[dict], max_tokens: int = 512,
                temperature: float = 0.4) -> Iterator[str]:
    """Yield text deltas as they are generated."""
    with _lock:
        llm = _get_chat()
        for chunk in llm.create_completion(_prompt(messages), max_tokens=max_tokens,
                                           stream=True, **_sampling(temperature)):
            delta = chunk["choices"][0]["text"]
            if delta:
                yield delta


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = _load(_find("LLM_EMBED_MODEL", EMBED_DIR), embedding=True,
                          n_ctx=512, n_batch=2048, n_ubatch=2048)
    return _embedder


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts; rows are L2-normalised float32."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    with _lock:
        vecs = _get_embedder().embed(list(texts), normalize=False, truncate=True)
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim == 3:  # model without pooling: mean over tokens
        arr = arr.mean(axis=1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _close(model) -> None:
    if model is not None:
        try:
            model.close()
        except Exception:
            pass


def unload() -> None:
    """Free the chat model (the try-on model needs the memory). Embedder stays (~70 MB)."""
    global _chat, _template
    with _lock:
        _close(_chat)
        _chat = None
        _template = None
        gc.collect()


@atexit.register
def _shutdown() -> None:
    # Free models before ggml-metal's static destructor runs; otherwise it
    # aborts at interpreter exit (GGML_ASSERT rsets count == 0, exit code 134).
    global _embedder
    unload()
    _close(_embedder)
    _embedder = None
    gc.collect()
