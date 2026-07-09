"""HazardWatch ML engine.

Two models, two jobs:
  - BertBiGRU (stage-1 checkpoint)          -> hazard classification (threshold 0.3)
  - all-MiniLM-L6-v2 sentence-transformer   -> semantic retrieval over flagged hazards

Live, row-aligned stores (row i of each refers to the same flagged review):
  - models/flagged_reviews.parquet          text + metadata
  - models/retrieval.index                  FAISS IndexFlatIP, L2-normalised 384-dim
  - models/retrieval_embeddings.npy         (N, 384) float32

Frozen stage-1 originals (kept as shipped, no longer queried):
  - models/hazard.index, models/flagged_embeddings.npy
  Build/refresh the retrieval store with scripts/build_retrieval_index.py.
"""

import os

# torch + faiss each link their own OpenMP runtime on Windows; without this the
# second import aborts the process (OMP Error #15). Must be set before either import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
BACKUPS_DIR = MODELS_DIR / "backups"

MODEL_PATH = MODELS_DIR / "bert_bigru_stage1.pt"
PARQUET_PATH = MODELS_DIR / "flagged_reviews.parquet"
RETR_INDEX_PATH = MODELS_DIR / "retrieval.index"
RETR_EMB_PATH = MODELS_DIR / "retrieval_embeddings.npy"

DEVICE = "cpu"
MAX_LEN = 256
THRESHOLD = 0.3  # tuned for recall; do not change without re-evaluating
ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DUP_SIM_THRESHOLD = 0.99  # ingest rejects near-duplicates at/above this similarity


class BertBiGRU(nn.Module):
    """Architecture must match the checkpoint exactly (layer names and shapes)."""

    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.gru = nn.GRU(768, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, 2)  # 2 classes: 0 = non-hazard, 1 = hazard

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        _, h = self.gru(out.last_hidden_state)
        return self.fc(torch.cat([h[0], h[1]], dim=1))  # logits, shape (B, 2)


class HazardEngine:
    """Owns both models + the three row-aligned stores. Thread-safe for adds/saves."""

    def __init__(self):
        self.lock = threading.Lock()
        self.dirty = False  # True when in-memory stores have unsaved additions

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertBiGRU().to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()

        self.embedder = SentenceTransformer(ST_MODEL_NAME, device=DEVICE)

        if not RETR_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"{RETR_INDEX_PATH} not found — run scripts/build_retrieval_index.py first"
            )
        self.index = faiss.read_index(str(RETR_INDEX_PATH))
        self.flagged = pd.read_parquet(PARQUET_PATH)
        self.embeds = np.load(RETR_EMB_PATH).astype("float32")

        self._assert_aligned()

    def _assert_aligned(self):
        assert self.index.ntotal == len(self.flagged) == self.embeds.shape[0], (
            f"artifacts are misaligned! index={self.index.ntotal} "
            f"parquet={len(self.flagged)} embeddings={self.embeds.shape[0]}"
        )

    def predict_prob(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(enc["input_ids"], enc["attention_mask"])
            prob = torch.softmax(logits, dim=1)[0, 1].item()  # P(hazard)
        return prob

    def embed(self, text: str) -> np.ndarray:
        """Sentence embedding, L2-normalised — must match how the retrieval index was built."""
        vec = self.embedder.encode([text], normalize_embeddings=True).astype("float32")
        return vec  # shape (1, 384)

    def search_similar(self, text: str, k: int = 5) -> list[dict]:
        return self._search(self.embed(text), k)

    def _search(self, vec: np.ndarray, k: int) -> list[dict]:
        scores, idxs = self.index.search(vec, k)
        results = []
        for i, s in zip(idxs[0], scores[0]):
            if i < 0:  # FAISS returns -1 for empty slots
                continue
            row = self.flagged.iloc[int(i)]
            results.append(
                {
                    "row": int(i),
                    "similarity": round(float(s), 4),
                    "text": row["text"],
                    "parent_asin": row["parent_asin"],
                    "hazard_prob": float(row["hazard_prob"]) if pd.notna(row["hazard_prob"]) else None,
                    "rating": float(row["rating"]) if pd.notna(row["rating"]) else None,
                }
            )
        return results

    def add_to_index(self, text: str, parent_asin: str | None = None) -> dict:
        """Classify; if hazard (and not a near-duplicate), append to all three stores."""
        prob = self.predict_prob(text)
        if prob < THRESHOLD:
            return {
                "flagged": False,
                "duplicate": False,
                "hazard_prob": round(prob, 4),
                "index_size": self.index.ntotal,
            }

        vec = self.embed(text)
        with self.lock:  # the three appends must stay aligned under concurrency
            top = self._search(vec, 1)
            if top and top[0]["similarity"] >= DUP_SIM_THRESHOLD:
                return {
                    "flagged": False,
                    "duplicate": True,
                    "duplicate_of": top[0]["row"],
                    "similarity": top[0]["similarity"],
                    "hazard_prob": round(prob, 4),
                    "index_size": self.index.ntotal,
                }
            new_row = {
                "parent_asin": parent_asin,
                "text": text,
                "hazard_prob": round(prob, 4),
                "timestamp": int(time.time() * 1000),
                "rating": None,
            }
            self.flagged = pd.concat([self.flagged, pd.DataFrame([new_row])], ignore_index=True)
            self.embeds = np.vstack([self.embeds, vec])
            self.index.add(vec)
            self.dirty = True
            row_id = self.index.ntotal - 1
        return {
            "flagged": True,
            "duplicate": False,
            "hazard_prob": round(prob, 4),
            "index_size": self.index.ntotal,
            "row": row_id,
        }

    def save(self) -> dict:
        """Back up the current on-disk stores, then persist the in-memory ones."""
        with self.lock:
            self._assert_aligned()
            backup_dir = self._backup_current()
            faiss.write_index(self.index, str(RETR_INDEX_PATH))
            self.flagged.to_parquet(PARQUET_PATH, index=False)
            np.save(RETR_EMB_PATH, self.embeds)
            self.dirty = False
            return {
                "saved": True,
                "index_size": self.index.ntotal,
                "backup": str(backup_dir.relative_to(BASE_DIR)) if backup_dir else None,
            }

    def reload(self) -> dict:
        """Discard unsaved in-memory additions by re-reading the last-saved on-disk stores."""
        with self.lock:
            if not RETR_INDEX_PATH.exists():
                raise FileNotFoundError(f"{RETR_INDEX_PATH} not found — nothing to reload")
            self.index = faiss.read_index(str(RETR_INDEX_PATH))
            self.flagged = pd.read_parquet(PARQUET_PATH)
            self.embeds = np.load(RETR_EMB_PATH).astype("float32")
            self._assert_aligned()
            self.dirty = False
            return {
                "reverted": True,
                "index_size": self.index.ntotal,
                "unsaved_changes": self.dirty,
            }

    def _backup_current(self) -> Path | None:
        """Copy the on-disk store trio into models/backups/<utc timestamp>/ (~5 MB each)."""
        files = [PARQUET_PATH, RETR_INDEX_PATH, RETR_EMB_PATH]
        if not all(f.exists() for f in files):
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_dir = BACKUPS_DIR / stamp
        backup_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, backup_dir / f.name)
        return backup_dir

    def product_summary(self, limit: int = 20) -> list[dict]:
        """Per-product hazard aggregates, ranked by risk (report count x mean probability)."""
        df = self.flagged.dropna(subset=["parent_asin"])
        if df.empty:
            return []
        g = df.groupby("parent_asin").agg(
            reports=("text", "size"),
            mean_prob=("hazard_prob", "mean"),
            max_prob=("hazard_prob", "max"),
            latest_ts=("timestamp", "max"),
        )
        g["risk_score"] = g["reports"] * g["mean_prob"]
        g = g.sort_values("risk_score", ascending=False).head(limit)
        return [
            {
                "parent_asin": asin,
                "reports": int(r.reports),
                "mean_prob": round(float(r.mean_prob), 4),
                "max_prob": round(float(r.max_prob), 4),
                "latest_ts": int(r.latest_ts) if pd.notna(r.latest_ts) else None,
                "risk_score": round(float(r.risk_score), 2),
            }
            for asin, r in g.iterrows()
        ]

    def product_detail(self, asin: str) -> dict | None:
        df = self.flagged[self.flagged["parent_asin"] == asin]
        if df.empty:
            return None
        df = df.sort_values("timestamp")
        years = (
            pd.to_datetime(df["timestamp"], unit="ms").dt.year.value_counts().sort_index()
        )
        return {
            "parent_asin": asin,
            "reports": len(df),
            "mean_prob": round(float(df["hazard_prob"].mean()), 4),
            "max_prob": round(float(df["hazard_prob"].max()), 4),
            "by_year": {int(y): int(c) for y, c in years.items()},
            "reviews": [
                {
                    "row": int(i),
                    "text": r["text"],
                    "hazard_prob": float(r["hazard_prob"]) if pd.notna(r["hazard_prob"]) else None,
                    "timestamp": int(r["timestamp"]) if pd.notna(r["timestamp"]) else None,
                    "rating": float(r["rating"]) if pd.notna(r["rating"]) else None,
                }
                for i, r in df.iterrows()
            ],
        }

    def stats(self) -> dict:
        return {
            "index_size": self.index.ntotal,
            "threshold": THRESHOLD,
            "max_length": MAX_LEN,
            "device": DEVICE,
            "embedding_dim": int(self.embeds.shape[1]),
            "retrieval_model": ST_MODEL_NAME,
            "unsaved_changes": self.dirty,
        }
