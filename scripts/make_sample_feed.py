"""Build static/sample_feed.json — a bundled demo feed for the UI's Live Feed tab.

The browser can't parse parquet, so we pre-render a ~1,000-review JSON sample from
the raw Amazon-review parquet. Keyword matching alone over-includes false positives
(e.g. "Amazon Fire tablet"), so by default we *score keyword candidates with the real
BERT+BiGRU classifier* and seed the feed with a target number of genuine hazards
(prob >= threshold). The rest is a random benign draw. The result is real review text
that visibly lights up during a demo. Fields: text, parent_asin, rating, timestamp.

Run from the project root:
  .venv\\Scripts\\python.exe scripts\\make_sample_feed.py                 # scored (default)
  .venv\\Scripts\\python.exe scripts\\make_sample_feed.py --no-score      # fast keyword-only
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd

OUT_PATH = BASE_DIR / "static" / "sample_feed.json"
DEFAULT_SOURCE = Path.home() / "Downloads" / "test_sample_100k.parquet"

HAZARD_RX = re.compile(
    r"fire|burn|shock|explod|smoke|melt|chok|hazard|spark|overheat|injur|"
    r"electrocut|flame|scald|toxic|caught fire|blew up|blister",
    re.I,
)
BR_RX = re.compile(r"<br\s*/?>", re.I)
WS_RX = re.compile(r"\s+")


def clean(s: str) -> str:
    return WS_RX.sub(" ", BR_RX.sub(" ", str(s))).strip()


def confirm_hazards(candidates: pd.DataFrame, target: int, cap: int, seed: int) -> pd.DataFrame:
    """Score keyword candidates with the real classifier; keep genuine hazards (prob >= threshold)."""
    import numpy as np
    import torch
    from app.ml import MAX_LEN, THRESHOLD, BertBiGRU
    from transformers import AutoTokenizer

    pool = candidates.sample(frac=1, random_state=seed).head(cap).reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertBiGRU()
    model.load_state_dict(torch.load(BASE_DIR / "models" / "bert_bigru_stage1.pt", map_location="cpu"))
    model.eval()

    texts = pool["text"].tolist()
    probs = np.empty(len(texts), dtype="float32")
    B = 32
    print(f"Scoring {len(texts)} keyword candidates (batch {B}, threshold {THRESHOLD})…")
    with torch.no_grad():
        for s in range(0, len(texts), B):
            enc = tok(texts[s:s + B], truncation=True, padding="max_length",
                      max_length=MAX_LEN, return_tensors="pt")
            logits = model(enc["input_ids"], enc["attention_mask"])
            probs[s:s + B] = torch.softmax(logits, dim=1)[:, 1].numpy()
            hits = int((probs[:s + B] >= THRESHOLD).sum())
            print(f"  scored {min(s + B, len(texts))}/{len(texts)} · {hits} hazards", end="\r")
            if hits >= target:
                probs = probs[:s + B]
                pool = pool.head(s + B)
                break
    print()
    pool = pool.assign(prob=probs)
    return pool[pool["prob"] >= THRESHOLD].head(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--n", type=int, default=1000, help="total reviews in the feed")
    ap.add_argument("--hazards", type=int, default=150, help="target confirmed hazards to seed")
    ap.add_argument("--score-cap", type=int, default=700, help="max candidates to score (bounds runtime)")
    ap.add_argument("--no-score", action="store_true", help="skip classifier; keyword-only (faster, noisier)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_parquet(args.source)
    df = df.assign(text=df["text"].map(clean))
    df = df[df["text"].str.len().between(3, 1500)].reset_index(drop=True)
    candidates = df[df["text"].str.contains(HAZARD_RX)]

    if args.no_score:
        hazard = candidates.sample(n=min(args.hazards, len(candidates)), random_state=args.seed)
        confirmed = len(hazard)
    else:
        hazard = confirm_hazards(candidates, args.hazards, args.score_cap, args.seed)
        confirmed = len(hazard)

    n_rand = args.n - len(hazard)
    rand = df.drop(hazard.index).sample(n=min(n_rand, len(df) - len(hazard)), random_state=args.seed)

    feed = (
        pd.concat([rand, hazard])
        .sample(frac=1, random_state=args.seed)  # shuffle so hazards interleave with benign
        .reset_index(drop=True)
    )

    records = [
        {
            "text": r.text,
            "parent_asin": None if pd.isna(r.parent_asin) else str(r.parent_asin),
            "rating": None if pd.isna(r.rating) else float(r.rating),
            "timestamp": None if pd.isna(r.timestamp) else int(r.timestamp),
        }
        for r in feed.itertuples(index=False)
    ]
    OUT_PATH.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {len(records)} reviews ({kb:.0f} KB) -> {OUT_PATH.relative_to(BASE_DIR)}")
    print(f"  confirmed hazards seeded: {confirmed}  |  random benign: {len(rand)}"
          f"  ({confirmed / len(records) * 100:.0f}% hazard density)")


if __name__ == "__main__":
    main()
