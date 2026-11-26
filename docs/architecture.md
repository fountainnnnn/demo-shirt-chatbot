# Fitting Room: architecture

A shirt-finding chatbot that runs entirely on one laptop: no cloud calls at
runtime. Three jobs: find shirts by conversation, recommend a size from height
and weight, and show the shirt on the user's own photo.

Design mock: `design/index.html`. Sample catalog: `scripts/fetch_catalog.py`.

## Shape

```
Browser UI (static)  <--HTTP + SSE-->  FastAPI app (Python, one process)
                                         |- agent loop        -> local LLM (Ollama)
                                         |- catalog search    -> SQLite + embeddings in memory
                                         |- size engine       -> plain code, no model
                                         |- try-on worker     -> FASHN VTON v1.5 (PyTorch)
```

Python backend because the try-on model is PyTorch. One process, one job queue.

## Models

| Job | Model | Size | Notes |
| --- | --- | --- | --- |
| Chat and tool calls | 4B-class instruct model, 4-bit, via Ollama | about 2.5 to 3 GB | Must do reliable JSON/tool output. Shortlist Qwen 4B and Gemma 4B class; pick by testing 20 scripted conversations at build time. |
| Text embeddings | Small embedding model (bge-small or nomic-embed class) | under 300 MB | Catalog embedded once at ingest. |
| Catalog enrichment | Zero-shot image classifier (SigLIP/CLIP class), ingest only | about 400 MB | Fills sleeve length, pattern, collar. Not loaded at runtime. |
| Try-on | FASHN VTON v1.5 | about 2 GB + 244 MB parser | Apache-2.0. Takes person photo + garment photo + category `tops`. |

On 8 GB machines the LLM and the try-on model do not stay loaded together: the
LLM is unloaded while a try-on runs.

## Catalog (the "R" in RAG)

Source: 100 shirts and T-shirts with 900x1200 photos. Dataset fields: gender,
type, colour, season, usage, product name.

Missing from the source, added at ingest:

- sleeve length, pattern, collar: image classifier, spot-checked by hand
- fabric, price: synthetic, marked as such in the data
- size chart per item (chest, length, shoulder per size): synthetic, three
  templates (slim, regular, relaxed) with small per-brand offsets

Each item gets one description sentence built from its fields; that sentence is
what gets embedded. With 100 items, search is a SQL filter plus brute-force
cosine in numpy. No vector database.

## Agent

One model, four tools. Code owns the control flow; the model fills in language.

State per conversation:

- `wants`: occasion, sleeve, colours wanted/avoided, budget, formality, gender
- `profile`: height, weight, chest (measured or estimated), fit preference
- `results`, `selected`, `shortlist`

Each user turn:

1. **Extract.** LLM returns a JSON patch to `wants`/`profile` (schema-constrained).
2. **Decide (code).** If a slot that changes the result set is missing (sleeve,
   budget, colour to avoid), ask one question. Cap: two clarifying questions per
   search, then search anyway and say what was assumed.
3. **Search.** `search_catalog(filters, query)`: hard filters in SQL, then
   semantic rank, top 8.
4. **Pick.** LLM chooses 3 of the 8 and writes one reason each, using only the
   fields it was given.
5. **Size.** `recommend_size(item_id, profile)` when height and weight are known.
6. **Try on.** `start_tryon(item_id, photo_id)` queues a job; UI polls.

Tools: `search_catalog`, `get_item`, `recommend_size`, `start_tryon`.

Guard rails:

- The UI renders product cards and size numbers from tool results, never from
  model text. The model cannot invent a shirt, a price or a measurement.
- Any item id in a reply that was not returned by a tool this turn is dropped.

## Size engine (no LLM)

1. Estimate chest and shoulder from height and weight with a regression fitted
   on public anthropometric data (ANSUR II). A measured chest overrides it.
2. Find the size whose chart range contains the chest.
3. Apply fit preference: slim stays or goes down if in the lower third, loose
   goes one up.
4. Return size, room at chest, length note, and a confidence that is lower when
   chest is estimated or sits near a boundary.

The LLM only phrases the result.

## Try-on

- Input checks before running: one person, upper body visible (pose model that
  ships with FASHN).
- Garment image is the catalog photo. FASHN accepts on-model photos.
- Output is cached per (photo, item).
- The image shows the look, not the size: size choice does not change the
  picture. The UI says so.

## API

- `POST /chat` (SSE): text deltas plus typed events `results`, `size`, `question`
- `GET /items/{id}`, `GET /images/{id}`
- `POST /profile`, `POST /photo`
- `POST /tryon` -> job id, `GET /tryon/{id}`

## Open risks

1. **Try-on speed off NVIDIA.** FASHN's README targets CUDA and falls back to
   float32 on CPU. Apple Silicon (MPS) is unverified. "Runs on most laptops"
   holds for chat; try-on may take minutes on CPU. First build task is a timing
   test on this Mac.
2. **Dataset licence.** The photos come from a re-hosted Kaggle set. Images are
   gitignored and fetched by script; check terms before the repo goes public.
3. **Synthetic sizing data.** Size charts and prices are made up for the demo.
4. **Small-model reliability.** Mitigated by schema-constrained output and by
   keeping decisions in code.

## Build order

1. Try-on timing test on this Mac (decides whether try-on is live or queued)
2. Catalog ingest and enrichment, search endpoint
3. Size engine with tests
4. Agent loop with the scripted-conversation test set, model pick
5. Front end from the mock, wired to the API
6. Try-on tab
