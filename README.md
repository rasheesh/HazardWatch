# HazardWatch — Capstone API + Prototype

Product-review **hazard detection API**: a fine-tuned **BERT + BiGRU** classifier scores
reviews for safety hazards (fire, choking, injury, …), a **MiniLM sentence-transformer +
FAISS** index retrieves the most similar previously-flagged hazards, and product-level
aggregation ranks the riskiest products. Ships with a browser prototype.

## Project layout

```
Capstone/
├── app/
│   ├── ml.py            # BertBiGRU classifier + MiniLM retrieval + HazardEngine
│   └── main.py          # FastAPI app (endpoints below)
├── models/
│   ├── bert_bigru_stage1.pt         # stage-1 classifier weights (~421 MB — NOT in git, download separately)
│   ├── flagged_reviews.parquet      # 1,343 flagged hazard reviews (live store)
│   ├── retrieval.index              # FAISS index of MiniLM embeddings (live store)
│   ├── retrieval_embeddings.npy     # (N, 384) float32, row-aligned (live store)
│   ├── hazard.index                 # frozen stage-1 original (BERT [CLS], unused)
│   ├── flagged_embeddings.npy       # frozen stage-1 original (unused)
│   └── backups/                     # original/ snapshot + timestamped saves
├── static/index.html    # prototype UI (served at /) — Analyze + Product dashboard tabs
├── scripts/
│   ├── sanity_check.py              # pre-deploy checks
│   └── build_retrieval_index.py     # (re)build the MiniLM retrieval store
├── run_server.bat       # one-click server start (logs to server.log)
└── requirements.txt
```

## Setup (once) — for collaborators

Requires **Python 3.10–3.12** (torch 2.3.1 has no 3.13/3.14 wheel).

**1. Clone and get the model weights.** The trained classifier weights
`models/bert_bigru_stage1.pt` (~421 MB) are **not in git** (they exceed GitHub's
100 MB file limit). Download the file and drop it into `models/`:

> 📦 **Download `bert_bigru_stage1.pt`:** _&lt;paste the Google Drive / OneDrive link here&gt;_

```powershell
git clone <repo-url>
cd Capstone
# ...then place the downloaded bert_bigru_stage1.pt into the models\ folder
```

**2. Create the environment and install dependencies:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**3. Build the retrieval index, then sanity-check** (first run downloads
`bert-base-uncased` + `all-MiniLM-L6-v2` from Hugging Face, ~450 MB, cached after):

```powershell
.\.venv\Scripts\python.exe scripts\build_retrieval_index.py   # rebuilds retrieval.index + embeddings from the parquet
.\.venv\Scripts\python.exe scripts\sanity_check.py            # verifies everything is aligned and serving-ready
```

## Run the API

Double-click `run_server.bat`, or:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The port opens immediately; the models load in the background (~15 s warm, ~2 min cold).
`GET /health` reports `"status": "loading"` until ready; other endpoints return 503 meanwhile.

- Prototype UI: <http://127.0.0.1:8000/>
- Swagger docs: <http://127.0.0.1:8000/docs>

## Endpoints

| Method | Path | Auth | What it does |
|--------|------|------|--------------|
| GET  | `/health` | – | status, index size, unsaved-changes flag |
| POST | `/predict` | – | `{text}` → hazard probability + verdict (threshold **0.3**) |
| POST | `/similar-reviews` | – | `{text, k}` → k most similar flagged hazards (cosine) |
| POST | `/analyze` | – | predict + similar in one call (what the UI uses) |
| GET  | `/products` | – | products ranked by risk (reports × mean hazard prob) |
| GET  | `/products/{asin}` | – | one product: stats, per-year timeline, all reviews |
| POST | `/ingest` | 🔑 | classify; if hazard **and not a near-duplicate (≥0.99 sim)**, add to index |
| POST | `/ingest-batch` | 🔑 | bulk ingest |
| POST | `/save` | 🔑 | persist stores (previous version auto-backed-up to `models/backups/`) |

🔑 = requires header `X-API-Key`. Key comes from the `HAZARDWATCH_API_KEY` env var
(dev default: `dev-key-hazardwatch`, which the prototype UI uses automatically).

```powershell
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" `
     -d '{"text": "The charger caught fire and started smoking."}'
```

## Architecture notes

- **Two models, two jobs.** The stage-1 BertBiGRU classifies (threshold 0.3, tuned for
  recall — don't change without re-evaluating). Retrieval uses `all-MiniLM-L6-v2`
  sentence embeddings, which give far better semantic similarity than the original
  frozen-BERT `[CLS]` vectors; the stage-1 index files are kept frozen for reference.
- **Row alignment invariant:** parquet row i ↔ FAISS vector i ↔ embeddings row i.
  Ingest/save are lock-guarded; the engine asserts alignment at load and before save.
- **Data safety:** every `/save` first copies the current on-disk stores to
  `models/backups/<timestamp>/` (~9 MB). The as-shipped dataset lives in
  `models/backups/original/`. Unsaved ingests are auto-saved on clean shutdown.
- **Windows gotcha:** `KMP_DUPLICATE_LIB_OK=TRUE` is set in `app/ml.py` before
  torch/faiss imports (OpenMP runtime clash).
- If you edit the parquet manually, rebuild retrieval with
  `scripts\build_retrieval_index.py`.

Full stage-1 artifact documentation: [MODEL_ARTIFACTS_SETUP.md](MODEL_ARTIFACTS_SETUP.md).
