"""Simulate a live e-commerce review feed against a running HazardWatch server.

Reads a review file (parquet / csv / jsonl), optionally samples N rows, and POSTs
them to the API exactly as a storefront webhook would — either as a real-time
*stream* (one /ingest call per review, with Poisson-timed arrivals) or as periodic
*bulk syncs* (/ingest-batch, up to 500 reviews per call). The server classifies each
review and adds genuine, non-duplicate hazards to the live FAISS + parquet stores.

Nothing is persisted to disk unless you pass --save (ingested rows otherwise live in
server memory and are auto-saved only on a clean shutdown).

Prereqs: the server must be running (run_server.bat) and the models loaded — this
script polls /health and waits for that automatically.

Examples (run from the project root):
  # stream 1000 randomly-sampled reviews at ~10/s, don't persist (safe demo)
  .venv\\Scripts\\python.exe scripts\\simulate_live_reviews.py

  # bulk-sync a specific CSV, 1000 rows, and persist at the end
  .venv\\Scripts\\python.exe scripts\\simulate_live_reviews.py ^
      --file "C:\\path\\to\\reviews.csv" --mode batch --limit 1000 --save

  # firehose: stream everything as fast as possible
  .venv\\Scripts\\python.exe scripts\\simulate_live_reviews.py --limit 0 --delay 0
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# Windows consoles default to cp1252 and choke on the ⚠/→/… glyphs below.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_FILE = Path.home() / "Downloads" / "test_sample_100k.parquet"
DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_KEY = os.environ.get("HAZARDWATCH_API_KEY", "dev-key-hazardwatch")

# where to find the review text / product id if the columns aren't already named
TEXT_ALIASES = ["text", "review", "review_text", "reviewText", "body", "content"]
ASIN_ALIASES = ["parent_asin", "asin", "product_id", "productId", "product"]
MAX_TEXT = 10_000  # matches ReviewIn max_length in app/main.py
BATCH_CAP = 500  # matches IngestBatchIn max_length


def pick_column(df: pd.DataFrame, explicit: str | None, aliases: list[str]) -> str | None:
    if explicit:
        if explicit not in df.columns:
            sys.exit(f"column '{explicit}' not in file (have: {list(df.columns)})")
        return explicit
    for name in aliases:
        if name in df.columns:
            return name
    return None


def load_reviews(path: Path, text_col: str | None, asin_col: str | None) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix in (".csv", ".tsv"):
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    else:
        sys.exit(f"unsupported file type '{suffix}' (use parquet/csv/tsv/json/jsonl)")

    tcol = pick_column(df, text_col, TEXT_ALIASES)
    if tcol is None:
        sys.exit(f"no review-text column found. Columns are {list(df.columns)}; "
                 f"pass --text-col to name it.")
    acol = pick_column(df, asin_col, ASIN_ALIASES)

    out = pd.DataFrame({"text": df[tcol].astype("string")})
    out["parent_asin"] = df[acol].astype("string") if acol else pd.NA
    out = out[out["text"].str.strip().str.len() > 0].reset_index(drop=True)
    out["text"] = out["text"].str.slice(0, MAX_TEXT)
    print(f"Loaded {len(out):,} usable reviews from {path.name} "
          f"(text='{tcol}', asin='{acol or '—'}')")
    return out


def sample(df: pd.DataFrame, limit: int, how: str, seed: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(df):
        return df
    if how == "head":
        return df.head(limit).reset_index(drop=True)
    return df.sample(n=limit, random_state=seed).reset_index(drop=True)


def row_payload(row) -> dict:
    asin = row.parent_asin
    return {
        "text": row.text,
        "parent_asin": None if (asin is None or pd.isna(asin)) else str(asin),
    }


def wait_for_ready(session: requests.Session, url: str, timeout: float = 180) -> dict:
    """Poll /health until the models are loaded (or give up)."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = session.get(f"{url}/health", timeout=5)
            data = r.json()
            if data.get("model_loaded"):
                print(f"Server ready — index currently holds {data.get('index_size')} hazards.")
                return data
            last = data.get("status")
            print(f"  waiting for models… (status: {last})")
        except requests.RequestException:
            print("  waiting for server… (is run_server.bat running?)")
        time.sleep(3)
    sys.exit(f"server not ready after {timeout:.0f}s (last status: {last})")


def stream_feed(session, url, key, reviews, delay):
    """One /ingest per review, with Poisson-distributed inter-arrival gaps."""
    headers = {"X-API-Key": key}
    flagged = dups = errors = 0
    t0 = time.time()
    for n, row in enumerate(reviews.itertuples(index=False), 1):
        try:
            r = session.post(f"{url}/ingest", json=row_payload(row), headers=headers, timeout=30)
            if r.status_code == 401:
                sys.exit("401 Unauthorized — wrong --api-key (or HAZARDWATCH_API_KEY).")
            r.raise_for_status()
            res = r.json()
        except requests.RequestException as e:
            errors += 1
            print(f"  [{n}] request failed: {e}")
            continue

        if res.get("flagged"):
            flagged += 1
            preview = row.text[:70].replace("\n", " ")
            print(f"  ⚠ HAZARD  #{res['row']:>4}  p={res['hazard_prob']:.3f}  "
                  f"asin={row_payload(row)['parent_asin']}  \"{preview}…\"  "
                  f"(index now {res['index_size']})")
        elif res.get("duplicate"):
            dups += 1
        if n % 100 == 0:
            rate = n / (time.time() - t0)
            print(f"  … {n:,} sent | {flagged} flagged | {dups} dup | {rate:.1f}/s")
        if delay > 0:
            time.sleep(random.expovariate(1 / delay))
    return {"processed": len(reviews), "flagged": flagged, "duplicates": dups, "errors": errors}


def batch_feed(session, url, key, reviews, batch_size):
    """Chunked /ingest-batch calls — simulates a periodic storefront bulk sync."""
    headers = {"X-API-Key": key}
    batch_size = min(batch_size, BATCH_CAP)
    flagged = dups = processed = 0
    index_size = None
    for start in range(0, len(reviews), batch_size):
        chunk = reviews.iloc[start:start + batch_size]
        items = [row_payload(row) for row in chunk.itertuples(index=False)]
        try:
            r = session.post(f"{url}/ingest-batch", json={"items": items}, headers=headers, timeout=300)
            if r.status_code == 401:
                sys.exit("401 Unauthorized — wrong --api-key (or HAZARDWATCH_API_KEY).")
            r.raise_for_status()
            res = r.json()
        except requests.RequestException as e:
            sys.exit(f"batch starting at {start} failed: {e}")
        processed += res["processed"]
        flagged += res["flagged"]
        dups += res["duplicates"]
        index_size = res["index_size"]
        print(f"  synced {processed:,}/{len(reviews):,} | "
              f"+{res['flagged']} flagged | +{res['duplicates']} dup | index {index_size}")
    return {"processed": processed, "flagged": flagged, "duplicates": dups, "errors": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, default=DEFAULT_FILE, help=f"review file (default: {DEFAULT_FILE})")
    ap.add_argument("--url", default=DEFAULT_URL, help="server base URL")
    ap.add_argument("--api-key", default=DEFAULT_KEY, help="X-API-Key (default: env or dev key)")
    ap.add_argument("--mode", choices=["stream", "batch"], default="stream",
                    help="stream = one /ingest per review (live feel); batch = /ingest-batch bulk sync")
    ap.add_argument("--limit", type=int, default=1000, help="how many reviews to send (0 = all)")
    ap.add_argument("--sample", choices=["random", "head"], default="random", help="how to pick --limit rows")
    ap.add_argument("--seed", type=int, default=42, help="random-sample seed (reproducible runs)")
    ap.add_argument("--delay", type=float, default=0.1, help="stream mode: mean seconds between arrivals (0 = firehose)")
    ap.add_argument("--batch-size", type=int, default=200, help=f"batch mode: reviews per call (max {BATCH_CAP})")
    ap.add_argument("--text-col", help="override review-text column name")
    ap.add_argument("--asin-col", help="override product-id column name")
    ap.add_argument("--save", action="store_true", help="persist the stores server-side when done")
    args = ap.parse_args()

    reviews = sample(load_reviews(args.file, args.text_col, args.asin_col), args.limit, args.sample, args.seed)
    print(f"Feeding {len(reviews):,} reviews to {args.url} in {args.mode.upper()} mode.\n")

    session = requests.Session()
    start_health = wait_for_ready(session, args.url)
    start_index = start_health.get("index_size")

    t0 = time.time()
    if args.mode == "stream":
        summary = stream_feed(session, args.url, args.api_key, reviews, args.delay)
    else:
        summary = batch_feed(session, args.url, args.api_key, reviews, args.batch_size)
    elapsed = time.time() - t0

    end_index = session.get(f"{args.url}/health", timeout=5).json().get("index_size")
    print("\n" + "=" * 60)
    print(f"Done in {elapsed:.1f}s ({summary['processed'] / max(elapsed, 1e-9):.1f} reviews/s)")
    print(f"  processed : {summary['processed']:,}")
    print(f"  flagged   : {summary['flagged']:,} new hazards")
    print(f"  duplicates: {summary['duplicates']:,} (skipped, already indexed)")
    if summary["errors"]:
        print(f"  errors    : {summary['errors']:,}")
    print(f"  index size: {start_index} → {end_index}  (+{(end_index or 0) - (start_index or 0)})")

    if args.save:
        r = session.post(f"{args.url}/save", headers={"X-API-Key": args.api_key}, timeout=120)
        if r.ok:
            print(f"  saved     : {r.json().get('backup')} backed up, stores persisted")
        else:
            print(f"  save FAILED: {r.status_code} {r.text}")
    else:
        print("  (not saved — pass --save to persist; otherwise auto-saved on clean shutdown)")


if __name__ == "__main__":
    main()
