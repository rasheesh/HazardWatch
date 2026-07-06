# HazardWatch — Model Artifacts Setup Guide

This document explains the **four files in the `models/` folder** and exactly how
to load and use them in any system. It is written for a developer integrating
the trained model into an application — no prior context required.

> **TL;DR:** The `.pt` file is **weights only** (no code). You must define the
> `BertBiGRU` architecture in your code, load the weights into it, and load the
> three companion files (FAISS index + parquet + embeddings) that must stay
> **row-aligned**. Inference is CPU-friendly. Use classification **threshold 0.3**
> and tokenizer **`bert-base-uncased`, max_length 256**.

---

## 1. The four files

| File | Size | What it is |
|------|------|------------|
| `bert_bigru_stage1.pt` | ~440 MB | PyTorch **state dict** (weights only) for the BertBiGRU classifier |
| `hazard.index` | ~4 MB | **FAISS** index (`IndexFlatIP`) of L2-normalised 768-dim `[CLS]` embeddings |
| `flagged_reviews.parquet` | ~285 KB | The flagged hazard reviews (text + metadata); **row i ↔ FAISS vector i** |
| `flagged_embeddings.npy` | ~4 MB | The `(N, 768)` float32 embeddings behind the index; **row i ↔ FAISS vector i** |

All three data files describe the **same N flagged hazards in the same order**.
As shipped, **N = 1343** (verify at load time — see §7).

```
models/
├── bert_bigru_stage1.pt        # classifier weights
├── hazard.index                # FAISS similarity index
├── flagged_reviews.parquet     # columns: parent_asin, text, hazard_prob, timestamp, rating
└── flagged_embeddings.npy      # shape (N, 768), float32
```

---

## 2. Dependencies

Tested, working versions (CPU-only):

```
torch==2.3.1            # CPU wheel: --extra-index-url https://download.pytorch.org/whl/cpu
transformers==4.42.3    # provides bert-base-uncased tokenizer + AutoModel
faiss-cpu==1.8.0
pandas==2.2.2
pyarrow==16.1.0         # parquet engine
numpy==1.26.4
```

> ⚠️ **Python version:** use **3.10–3.12**. `torch==2.3.1` has **no wheel for
> Python 3.13**. (If you must run 3.9, it works but Pydantic/SQLAlchemy-style
> `X | None` annotations need the `eval_type_backport` package.)

The first load downloads `bert-base-uncased` (~440 MB) from Hugging Face to
build the architecture, then overwrites its weights with the `.pt` file.

---

## 3. Define the architecture (required — the `.pt` is weights only)

The state dict cannot be used without the matching `nn.Module`. Define it
**exactly** as below — the layer names/shapes must match the checkpoint:

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class BertBiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")  # frozen during training
        self.gru = nn.GRU(768, 128, batch_first=True, bidirectional=True)
        self.fc  = nn.Linear(128 * 2, 2)   # 2 classes: 0 = non-hazard, 1 = hazard

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        _, h = self.gru(out.last_hidden_state)
        return self.fc(torch.cat([h[0], h[1]], dim=1))   # logits, shape (B, 2)
```

---

## 4. Load everything (once, at startup)

```python
import faiss, numpy as np, pandas as pd
from transformers import AutoTokenizer

MODEL_PATH  = "models/bert_bigru_stage1.pt"
INDEX_PATH  = "models/hazard.index"
PARQUET_PATH= "models/flagged_reviews.parquet"
EMB_PATH    = "models/flagged_embeddings.npy"
DEVICE      = "cpu"
MAX_LEN     = 256
THRESHOLD   = 0.3          # tuned for recall; do not change without re-evaluating

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = BertBiGRU().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

index   = faiss.read_index(INDEX_PATH)          # IndexFlatIP
flagged = pd.read_parquet(PARQUET_PATH)         # row i ↔ index vector i
embeds  = np.load(EMB_PATH).astype("float32")   # row i ↔ index vector i

assert index.ntotal == len(flagged) == embeds.shape[0], "artifacts are misaligned!"
```

---

## 5. The two core operations

### 5a. Classify a review (hazard probability)

```python
def predict_prob(text: str) -> float:
    enc = tokenizer(text, truncation=True, padding="max_length",
                    max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        prob = torch.softmax(logits, dim=1)[0, 1].item()   # P(hazard)
    return prob

# A review is a hazard when prob >= THRESHOLD (0.3).
```

### 5b. Find similar flagged hazards (FAISS)

The index stores **L2-normalised `[CLS]` embeddings** searched with inner
product (= cosine similarity). Embed the query the **same way** or results are
meaningless:

```python
def embed(text: str) -> np.ndarray:
    enc = tokenizer(text, truncation=True, padding="max_length",
                    max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        seq = model.bert(enc["input_ids"], enc["attention_mask"]).last_hidden_state
        vec = seq[:, 0, :].cpu().numpy().astype("float32")   # [CLS] token = position 0
    faiss.normalize_L2(vec)                                  # unit length
    return vec                                               # shape (1, 768)

def search_similar(text: str, k: int = 5):
    scores, idxs = index.search(embed(text), k)
    results = []
    for i, s in zip(idxs[0], scores[0]):
        if i < 0:                       # FAISS returns -1 for empty slots
            continue
        row = flagged.iloc[int(i)]      # row-aligned lookup for text/metadata
        results.append({"row": int(i), "similarity": float(s),
                        "text": row["text"], "parent_asin": row["parent_asin"]})
    return results
```

---

## 6. Growing the index at runtime (optional)

To add newly-flagged hazards and keep them searchable, **append to all three
stores in the same position** — this is the invariant that must never break:

```python
def add_to_index(text: str, parent_asin: str | None):
    prob = predict_prob(text)
    if prob < THRESHOLD:
        return {"flagged": False, "index_size": index.ntotal}

    vec = embed(text)                                    # (1, 768), normalised
    global flagged, embeds
    new_row = {"parent_asin": parent_asin, "text": text, "hazard_prob": round(prob, 4),
               "timestamp": int(__import__("time").time() * 1000), "rating": None}
    flagged = pd.concat([flagged, pd.DataFrame([new_row])], ignore_index=True)  # parquet row
    embeds  = np.vstack([embeds, vec])                                          # embeddings row
    index.add(vec)                                                             # FAISS vector
    return {"flagged": True, "index_size": index.ntotal, "row": index.ntotal - 1}

def save_index():
    assert index.ntotal == len(flagged) == embeds.shape[0], "refuse to save misaligned data"
    faiss.write_index(index, INDEX_PATH)
    flagged.to_parquet(PARQUET_PATH, index=False)
    np.save(EMB_PATH, embeds)
```

Notes:
- **Don't `save_index()` on every add** — it rewrites all three files. Save
  periodically, on shutdown, or on demand.
- **Concurrency:** if multiple threads/requests can add at once, guard
  `add_to_index` / `save_index` with a lock so the three appends stay aligned.
- `save_index()` **overwrites** the files in `models/`. Back them up first if you
  want to keep the pristine set.

---

## 7. Sanity check before you deploy

```python
assert index.ntotal == len(flagged) == embeds.shape[0]

# hazard should score high, benign low
assert predict_prob("The charger caught fire and started smoking.") > 0.5
assert predict_prob("Great product, works well and arrived on time.") < 0.3

# query should retrieve semantically similar hazards
print(search_similar("battery exploded while charging", k=3))
```

`flagged_reviews.parquet` columns: `parent_asin`, `text`, `hazard_prob`,
`timestamp` (epoch ms), `rating`.

---

## 8. Gotchas (learned the hard way)

| Symptom | Cause / Fix |
|---------|-------------|
| `OMP: Error #15 ... libiomp5md.dll already initialized` (Windows) | torch + faiss each link an OpenMP runtime. Set `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` **before importing torch/faiss**. |
| `torch==2.3.1` won't install | You're on Python 3.13. Use Python 3.10–3.12 (CPU wheel: `--extra-index-url https://download.pytorch.org/whl/cpu`). |
| Similarity results look random | Query wasn't embedded identically — must be `[CLS]` token, L2-normalised, `max_length=256`. |
| `load_state_dict` key mismatch | Your `BertBiGRU` class doesn't match §3 (layer names/sizes). |
| Concurrent first requests load the model twice / fail | Load the model **once at startup**, single-threaded, before serving. |
| Index and parquet lengths differ | Never edit one store without the others. Re-check the §7 assert. |

---

## 9. How these map to API endpoints (reference)

If you follow the pattern in this repo, the operations map to:

| Function | Endpoint |
|----------|----------|
| `predict_prob` | `POST /predict` |
| `search_similar` | `POST /similar-reviews` |
| `add_to_index` | `POST /ingest` (and `POST /ingest-batch` for bulk) |
| `save_index` | `POST /save` |

Model training details and the design rationale are in
`HazardWatch_Model_Handoff.txt`. Trained-model export steps are in
`docs/colab_export_guide.md`.
