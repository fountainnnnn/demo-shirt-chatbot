# Interface contracts

Source of truth for module boundaries. If you need to change one, say so in
your final report; do not change it silently.

## Environment

- Python 3.12 venv at `.venv/` (use `.venv/bin/python`, `.venv/bin/pytest`).
- Install packages with: `UV_CACHE_DIR="$PWD/.uv/cache" uv pip install -p .venv <pkg>`
  and add them to `requirements.txt`.
- All model weights live under `models/` (gitignored). Set `HF_HOME=models/hf`
  for anything that uses the Hugging Face cache. Never write outside the repo.
- No cloud calls at runtime. Downloads happen only in `scripts/`.

## Catalog item (`data/catalog/catalog.json`, list of objects)

```json
{
  "id": 30802,
  "name": "Blue cotton shirt",
  "brand": "Fabindia",
  "gender": "men",                // men | women | boys | girls | unisex
  "type": "shirt",                // shirt | tshirt
  "colour": "blue",               // lower-case base colour from the dataset
  "sleeve": "short",              // short | long
  "pattern": "solid",             // solid | striped | checked | printed
  "usage": "casual",              // casual | formal | sports | ethnic
  "season": "summer",
  "fabric": "light cotton",       // synthetic
  "price": 42,                    // synthetic, whole USD
  "image": "images/30802.jpg",    // relative to data/catalog/
  "description": "one sentence built from the fields above; this is what gets embedded",
  "size_chart": {
    "fit": "regular",             // slim | regular | relaxed
    "sizes": [
      {"size": "S", "body_chest": [86, 94], "garment_chest": 102, "length": 72, "shoulder": 44}
    ]
  },
  "source_name": "Fabindia Men Blue Shirt",
  "synthetic": ["price", "fabric", "size_chart"]
}
```

Sizes are always S, M, L, XL (XS and XXL optional), ordered small to large,
`body_chest` ranges contiguous, all lengths in cm. The raw download is kept as
`data/catalog/raw.json`.

## `app/catalog.py`

```python
load() -> list[dict]
get(item_id: int) -> dict | None
search(filters: dict, query: str = "", k: int = 8, embed=None) -> list[dict]
```

`filters` keys, all optional: `gender`, `type`, `sleeve`, `pattern`, `usage`,
`colours` (list, include any), `avoid_colours` (list), `max_price`, `min_price`.
`embed` is a callable `list[str] -> np.ndarray` (rows L2-normalised). With
`embed=None` or no `data/catalog/embeddings.npy`, rank by keyword overlap.
Each result is the item dict plus `"score": float`.
If hard filters leave fewer than 3 items, relax in this order: pattern, usage,
colours, price (+25%), and return `"relaxed": [names of relaxed filters]` on
each result.

## `app/llm.py`

```python
chat(messages: list[dict], schema: dict | None = None, max_tokens: int = 512, temperature: float = 0.2) -> str
chat_stream(messages, max_tokens=512, temperature=0.4) -> Iterator[str]
embed(texts: list[str]) -> np.ndarray      # L2-normalised float32
unload() -> None                           # free the chat model's memory
```

With `schema` (a JSON Schema), the return value is a JSON string guaranteed to
parse and match it (grammar-constrained).

## `app/sizing.py`

```python
estimate_body(height_cm: float, weight_kg: float, gender: str = "men") -> dict
#  {"chest_cm": float, "shoulder_cm": float}

recommend_size(item: dict, profile: dict) -> dict
```

`profile`: `{"height_cm", "weight_kg", "gender"?, "chest_cm"? (measured), "fit": "slim"|"regular"|"loose"}`.

Returns:

```json
{
  "size": "L", "chart_size": "M", "moved": 1,
  "chest_cm": 100.0, "chest_estimated": true,
  "room_cm": 10.0,
  "garment": {"chest": 110, "length": 76, "shoulder": 48},
  "shoulder_diff_cm": 1.0,
  "ranges": [{"size": "S", "lo": 86, "hi": 94}],
  "confidence": 0.86,
  "notes": ["one_up_for_loose_fit"]
}
```

## `app/tryon.py`

```python
check_photo(path: str) -> dict                     # {"ok": bool, "issues": [str]}
run_tryon(person_path: str, garment_path: str, out_path: str, quality: str = "balanced") -> dict
#  {"seconds": float, "device": "mps"|"cuda"|"cpu"}
```

Lazy-loads the model on first call; `unload()` frees it.

## HTTP API (`app/main.py`, FastAPI, served on 127.0.0.1:8000)

Static front end from `web/` at `/`. Catalog photos at `/images/{id}.jpg`.

- `POST /api/chat` body `{"conversation_id": str | null, "message": str}`.
  Response is `text/event-stream`. Events (`event:` name, `data:` JSON):
  - `conversation` `{"id", "title"}` (first event)
  - `question` `{"text", "options": [str]}` clarifying question
  - `text` `{"delta": str}` streamed assistant words
  - `results` `{"title": str, "filters": [{"key","label"}], "items": [item + "reason": str + "best": bool]}`
  - `size` `{"item_id": int, ...recommend_size output}`
  - `state` `{"wants": {...}, "profile": {...}}`
  - `done` `{}`
- `GET /api/conversations` -> `[{"id","title","updated"}]`, newest first
- `GET /api/conversations/{id}` -> `{"id","title","messages":[...], "results", "size", "state"}`
- `GET /api/items/{id}` -> item
- `POST /api/size` body `{"item_id", "profile"}` -> recommend_size output
- `GET /api/shortlist`, `POST /api/shortlist` `{"item_id"}`, `DELETE /api/shortlist/{item_id}`
  -> `[{item + "size": recommend_size output | null}]`
- `POST /api/photo` (multipart `file`) -> `{"photo_id", "url", "check": {"ok","issues"}}`
- `POST /api/tryon` body `{"photo_id", "item_id", "quality"?}` -> `{"job_id"}`
- `GET /api/tryon/{job_id}` -> `{"status": "queued"|"running"|"done"|"error", "stage": str, "seconds": float | null, "url": str | null, "error": str | null}`
- `GET /api/tryons` -> past results `[{"job_id","item_id","url","seconds"}]`
- `GET /api/status` -> `{"chat_model": str, "tryon_device": str, "ready": bool}`

State is kept in `data/state.sqlite` (gitignored).
