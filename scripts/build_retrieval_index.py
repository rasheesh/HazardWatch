"""Build the sentence-transformer retrieval index from flagged_reviews.parquet.

Re-embeds all flagged reviews with all-MiniLM-L6-v2 (384-dim, much better semantic
similarity than frozen-BERT [CLS]) and writes:
  models/retrieval.index            FAISS IndexFlatIP (cosine via L2-normalised vectors)
  models/retrieval_embeddings.npy   (N, 384) float32, row i <-> parquet row i

The original stage-1 artifacts (hazard.index, flagged_embeddings.npy) are left
untouched as frozen originals.

Run from the project root:  .venv\\Scripts\\python.exe scripts\\build_retrieval_index.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.ml import PARQUET_PATH, RETR_EMB_PATH, RETR_INDEX_PATH, ST_MODEL_NAME


def main():
    flagged = pd.read_parquet(PARQUET_PATH)
    texts = flagged["text"].fillna("").astype(str).tolist()
    print(f"Embedding {len(texts)} flagged reviews with {ST_MODEL_NAME}...")

    model = SentenceTransformer(ST_MODEL_NAME, device="cpu")
    t0 = time.time()
    embs = model.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    ).astype("float32")
    print(f"Encoded in {time.time() - t0:.1f}s, shape={embs.shape}")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, str(RETR_INDEX_PATH))
    np.save(RETR_EMB_PATH, embs)
    print(f"Wrote {RETR_INDEX_PATH.name} ({index.ntotal} vectors) and {RETR_EMB_PATH.name}")

    # quick quality probe: results should be topically related now
    for query in ["battery exploded while charging", "my kid almost choked on a broken piece"]:
        q = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = index.search(q, 3)
        print(f"\n  '{query}':")
        for i, s in zip(idxs[0], scores[0]):
            print(f"    {s:.3f}  #{i}  {texts[int(i)][:80]}")


if __name__ == "__main__":
    main()
