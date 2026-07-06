"""Pre-deploy sanity check — MODEL_ARTIFACTS_SETUP.md §7, extended for v1.1.

Requires the retrieval store (run scripts/build_retrieval_index.py first).
Run from the project root:  .venv\\Scripts\\python.exe scripts\\sanity_check.py
Downloads bert-base-uncased + all-MiniLM-L6-v2 on first run (cached afterwards).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml import THRESHOLD, HazardEngine  # noqa: E402


def main():
    t0 = time.time()
    print("Loading engine (model + index + parquet + embeddings)...")
    engine = HazardEngine()
    print(f"Loaded in {time.time() - t0:.1f}s")

    # 1. Alignment
    n = engine.index.ntotal
    assert n == len(engine.flagged) == engine.embeds.shape[0]
    print(f"[OK] alignment: index={n} parquet={len(engine.flagged)} embeds={engine.embeds.shape}")

    # 2. Parquet schema / content overview
    print(f"[OK] parquet columns: {list(engine.flagged.columns)}")
    print(engine.flagged.head(3).to_string(max_colwidth=60))
    print(f"     hazard_prob range: {engine.flagged['hazard_prob'].min():.3f}"
          f" – {engine.flagged['hazard_prob'].max():.3f}")

    # 3. Classifier direction: hazard scores high, benign low
    hazard_text = "The charger caught fire and started smoking."
    benign_text = "Great product, works well and arrived on time."
    p_hazard = engine.predict_prob(hazard_text)
    p_benign = engine.predict_prob(benign_text)
    print(f"[..] P(hazard | '{hazard_text}') = {p_hazard:.4f}")
    print(f"[..] P(hazard | '{benign_text}') = {p_benign:.4f}")
    assert p_hazard > 0.5, "hazard text should score > 0.5"
    assert p_benign < THRESHOLD, "benign text should score < threshold"
    print("[OK] classifier sanity passed")

    # 4. Similarity search returns semantically related hazards
    query = "battery exploded while charging"
    t1 = time.time()
    results = engine.search_similar(query, k=3)
    print(f"[OK] search_similar('{query}') in {time.time() - t1:.2f}s:")
    for r in results:
        print(f"     {r['similarity']:.3f}  #{r['row']}  {r['text'][:80]}...")

    print("\nALL CHECKS PASSED — artifacts are ready to serve.")


if __name__ == "__main__":
    main()
