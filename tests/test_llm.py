"""Tests for app/llm.py against the real local models. Slow: run with RUN_SLOW=1."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1",
                                reason="loads local models; set RUN_SLOW=1")

from app import llm  # noqa: E402

WANTS_SCHEMA = {
    "type": "object",
    "properties": {
        "sleeve": {"type": "string", "enum": ["short", "long"]},
        "avoid_colours": {"type": "array", "items": {"type": "string"}},
        "max_price": {"type": "integer"},
    },
    "additionalProperties": False,
}

EXTRACT_SYSTEM = (
    "Extract the shopper's shirt preferences from their message as JSON. "
    "Fields: sleeve (short|long), avoid_colours (lower-case colour names they do not want), "
    "max_price (whole USD). Include only fields the message states."
)


def test_schema_extraction():
    out = llm.chat(
        [{"role": "system", "content": EXTRACT_SYSTEM},
         {"role": "user", "content": "Short sleeves. No white, and under $60"}],
        schema=WANTS_SCHEMA, temperature=0.0,
    )
    print("extraction:", out)
    assert json.loads(out) == {"sleeve": "short", "avoid_colours": ["white"], "max_price": 60}


def test_embed_shape_and_similarity():
    texts = ["blue striped formal shirt", "navy pinstripe office shirt", "red football jersey"]
    v = llm.embed(texts)
    assert v.dtype == np.float32 and v.shape[0] == 3 and v.ndim == 2
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    near, far = float(v[0] @ v[1]), float(v[0] @ v[2])
    print(f"cos(query, navy pinstripe)={near:.3f}  cos(query, red jersey)={far:.3f}")
    assert near > far


def test_stream_yields_chunks():
    chunks = list(llm.chat_stream(
        [{"role": "user", "content": "In one sentence, why is linen good in summer?"}],
        max_tokens=60))
    print("stream:", len(chunks), "chunks:", repr("".join(chunks)))
    assert len(chunks) > 1
    assert "<think>" not in "".join(chunks)


def test_unload_then_reload():
    llm.unload()
    assert llm._chat is None
    assert llm.chat([{"role": "user", "content": "Reply with the word ok."}], max_tokens=5)
