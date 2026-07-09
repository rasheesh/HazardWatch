"""Diagnostic probe for the BERT+BiGRU hazard classifier.

Runs a small *labeled* test set spanning clear hazards, factory-defect hazards,
benign defects (DOA/cosmetic), borderline cases, negations, and benign reviews.
Captures raw logits so we can inspect calibration (not just the thresholded verdict):

  - probability distribution / bimodality (are scores stuck near 0 and 1?)
  - decision margin |logit_hazard - logit_safe| (how saturated the softmax is)
  - recall on hazards overall AND on the factory-defect subset specifically
  - false negatives (missed hazards) and false positives
  - a temperature-scaling "what-if": how much the scores de-saturate at T>1

Run from the project root:
  .venv\\Scripts\\python.exe scripts\\probe_classifier.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from transformers import AutoTokenizer

from app.ml import MAX_LEN, THRESHOLD, BertBiGRU

# label: 1 = hazard, 0 = not a hazard, None = genuinely borderline (report only)
TESTS = [
    # --- unambiguous hazards (explicit danger vocabulary) ---
    ("hazard_clear", 1, "The charger caught fire and melted the wall outlet."),
    ("hazard_clear", 1, "The battery exploded while charging overnight and burned my desk."),
    ("hazard_clear", 1, "My 2-year-old choked on a small piece that snapped off the toy."),
    ("hazard_clear", 1, "The blender blade shattered mid-use and a shard cut my hand."),
    ("hazard_clear", 1, "I got a strong electric shock the moment I plugged it in."),
    ("hazard_clear", 1, "The space heater started smoking and throwing sparks."),

    # --- factory-defect hazards (danger described in NEUTRAL, non-scary words) ---
    ("hazard_defect", 1, "Out of the box the wiring was exposed — I could see bare copper on the live wire."),
    ("hazard_defect", 1, "Brand new unit and the plug is wired backwards, live and neutral are reversed."),
    ("hazard_defect", 1, "The casing arrived cracked and the battery underneath was visibly swollen and bulging."),
    ("hazard_defect", 1, "There was a sharp metal burr on the edge from manufacturing that sliced my finger open."),
    ("hazard_defect", 1, "Manufacturing defect: the blade guard was missing entirely, blade fully exposed."),
    ("hazard_defect", 1, "A defective seal let the battery fluid leak out and it left a chemical burn on my skin."),
    ("hazard_defect", 1, "The ground pin was never attached at the factory, so the metal body is electrified."),
    ("hazard_defect", 1, "Straight from the package the insulation was cut through and the internal contacts were touching."),

    # --- benign defects (broken but NOT dangerous) ---
    ("defect_benign", 0, "Dead on arrival, it would not power on at all."),
    ("defect_benign", 0, "Factory defect — there is one dead pixel in the corner of the screen."),
    ("defect_benign", 0, "Arrived with a small cosmetic scratch on the back cover."),
    ("defect_benign", 0, "The Bluetooth refuses to pair, seems like a defective unit."),
    ("defect_benign", 0, "Wrong item shipped — I ordered the black one and got white."),
    ("defect_benign", 0, "Stopped working after two days, now the screen just stays black."),

    # --- borderline (report only, no hard label) ---
    ("borderline", None, "It gets a little warm during long use but nothing alarming."),
    ("borderline", None, "The power cord feels flimsy and I worry it might become a fire risk down the line."),
    ("borderline", None, "A tiny plastic tab broke off but it didn't cause any problem."),
    ("borderline", None, "Runs hotter than I expected, though probably still within normal range."),

    # --- negation / tricky phrasing (should be LOW) ---
    ("negation", 0, "Thankfully it did NOT catch fire — works perfectly and I'm very happy."),
    ("negation", 0, "No safety issues whatsoever: no shock, no smoke, no overheating. Great buy."),
    ("negation", 0, "I was worried it might explode but it has been completely safe and reliable."),

    # --- clearly benign ---
    ("benign", 0, "Great product, works well and arrived on time. Very happy with it."),
    ("benign", 0, "Love it — exactly as described, fast shipping, would buy again."),
    ("benign", 0, "Comfortable, stylish, and great value for the price."),
    ("benign", 0, "Battery life is excellent and it charges quickly."),
]


def main():
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertBiGRU()
    model.load_state_dict(torch.load(BASE_DIR / "models" / "bert_bigru_stage1.pt", map_location="cpu"))
    model.eval()

    texts = [t[2] for t in TESTS]
    enc = tok(texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"]).numpy()  # (N, 2)
    probs = _softmax(logits)[:, 1]
    margins = logits[:, 1] - logits[:, 0]  # >0 leans hazard

    print(f"\nThreshold = {THRESHOLD}   (verdict = hazard if prob >= threshold)\n")
    print(f"{'category':14} {'exp':>3} {'prob':>7} {'margin':>7}  verdict   text")
    print("-" * 108)
    for (cat, label, text), p, m in zip(TESTS, probs, margins):
        verdict = "HAZARD" if p >= THRESHOLD else "safe"
        exp = "-" if label is None else ("HAZ" if label == 1 else "ok")
        flag = ""
        if label == 1 and p < THRESHOLD:
            flag = "  <== MISS (false negative)"
        elif label == 0 and p >= THRESHOLD:
            flag = "  <== false positive"
        print(f"{cat:14} {exp:>3} {p:7.4f} {m:7.2f}  {verdict:8}  {text[:52]}{flag}")

    _calibration(probs, margins, logits)
    _recall(probs)


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _calibration(probs, margins, logits):
    print("\n" + "=" * 60)
    print("CALIBRATION")
    bins = [(0, .05), (.05, .30), (.30, .70), (.70, .95), (.95, 1.01)]
    print("  probability distribution (are scores stuck at the extremes?):")
    for lo, hi in bins:
        c = int(((probs >= lo) & (probs < hi)).sum())
        bar = "#" * c
        print(f"    {lo:.2f}-{hi:<4.2f} | {c:2d} {bar}")
    mid = int(((probs >= .30) & (probs <= .70)).sum())
    print(f"  in the uncertain band 0.30-0.70: {mid}/{len(probs)} "
          f"({mid / len(probs) * 100:.0f}%)")
    print(f"  mean |margin| (logit gap): {np.abs(margins).mean():.2f} "
          f"(large => saturated/overconfident softmax)")

    # temperature scaling what-if: how many land in the uncertain band as T grows
    print("  temperature-scaling what-if (softmax(logits / T)):")
    for T in (1, 2, 3, 4, 6):
        pT = _softmax(logits / T)[:, 1]
        midT = int(((pT >= .30) & (pT <= .70)).sum())
        print(f"    T={T}: {midT:2d}/{len(pT)} scores in 0.30-0.70 band")


def _recall(probs):
    print("\n" + "=" * 60)
    print("ACCURACY BY GROUP")
    groups = {}
    for (cat, label, _), p in zip(TESTS, probs):
        if label is None:
            continue
        pred = 1 if p >= THRESHOLD else 0
        g = groups.setdefault(cat, [0, 0])
        g[1] += 1
        if pred == label:
            g[0] += 1
    for cat, (ok, n) in groups.items():
        print(f"  {cat:14} {ok}/{n} correct")

    haz = [(cat, p) for (cat, label, _), p in zip(TESTS, probs) if label == 1]
    haz_ok = sum(1 for _, p in haz if p >= THRESHOLD)
    defect = [(cat, p) for (cat, label, _), p in zip(TESTS, probs) if label == 1 and cat == "hazard_defect"]
    defect_ok = sum(1 for _, p in defect if p >= THRESHOLD)
    print(f"\n  hazard recall (all):            {haz_ok}/{len(haz)} "
          f"({haz_ok / len(haz) * 100:.0f}%)")
    print(f"  hazard recall (factory-defect): {defect_ok}/{len(defect)} "
          f"({defect_ok / len(defect) * 100:.0f}%)  <-- the gap you noticed")


if __name__ == "__main__":
    main()
