# -*- coding: utf-8 -*-
"""
HazardWatch — API-level automation (requests).

Same plain-script style as the team's automation example: test_* functions,
assert, print PASS / FAIL, a manual runner at the bottom, run with:

    pip install requests pandas pyarrow
    python test_hazardwatch_api.py

Selenium drives the browser UI (see test_hazardwatch_ui.py); the browser cannot
read raw HTTP status codes or measure latency, so those are automated here.

Data safety: the mutating tests (ingest / batch / save) restore the index with
POST /revert (and a file snapshot around /save), so running this suite leaves the
stores exactly as they were.

Covers API-testable cases: TC-06..TC-30, TC-37, TC-42..TC-50, TC-54..TC-57.
"""
import io
import logging
import os
import shutil
import statistics
import sys
import tempfile
import time

import pandas as pd
import requests

BASE = os.environ.get("HW_URL", "http://127.0.0.1:8000").rstrip("/")
KEY = os.environ.get("HAZARDWATCH_API_KEY", "dev-key-hazardwatch")
AUTH = {"X-API-Key": KEY}
MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
STORE_FILES = ["flagged_reviews.parquet", "retrieval.index", "retrieval_embeddings.npy"]

# Full detail (with timestamps) always goes to the .log file. The console shows
# clean colour-coded PASS/FAIL lines; set HW_VERBOSE=1 to ALSO stream the detail
# (latencies, etc.) to the console live, without the PASS/FAIL duplication.
_handlers = [logging.FileHandler("hazardwatch_api_tests.log")]
if os.environ.get("HW_VERBOSE", "") not in ("", "0", "false", "False"):
    _con = logging.StreamHandler()
    _con.setFormatter(logging.Formatter("      %(message)s"))
    _con.addFilter(lambda r: not r.getMessage().startswith(("PASS -", "FAIL -", "ERROR -", "DONE")))
    _handlers.append(_con)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    handlers=_handlers)
log = logging.getLogger("hw-api")
PASS = FAIL = 0


# ----------------------------- pretty console results -----------------------------
if os.name == "nt":
    os.system("")  # enable ANSI colour codes in Windows terminals


def _clr(t, c):
    return f"\033[{c}m{t}\033[0m"


def _pass(name):
    print(f"  {_clr(' PASS ', '1;30;42')}  {name}")


def _fail(kind, name, err):
    print(f"  {_clr(f' {kind} ', '1;97;41')}  {name}")
    print(f"             {_clr('->', '1;31')} {err}")


def _banner(passed, failed):
    total = passed + failed
    bar = "=" * 62
    if failed == 0:
        tag = _clr(f"  ALL {total} TESTS PASSED  ", "1;30;42")
    else:
        tag = _clr(f"  {failed} FAILED / {total}  —  {passed} passed  ", "1;97;41")
    print(f"\n{bar}\n  {tag}\n{bar}")


def run(name, fn):
    """Run one test function; show a colour-coded PASS/FAIL line (detail to .log)."""
    global PASS, FAIL
    try:
        fn()
        PASS += 1; _pass(name); log.info("PASS - %s", name)
    except AssertionError as e:
        FAIL += 1; _fail("FAIL", name, e); log.error("FAIL - %s -> %s", name, e)
    except Exception as e:  # noqa
        FAIL += 1; _fail("ERR ", name, f"{type(e).__name__}: {e}"); log.error("ERROR - %s -> %s: %s", name, type(e).__name__, e)


def index_size():
    return requests.get(f"{BASE}/health", timeout=10).json().get("index_size")


def revert():
    requests.post(f"{BASE}/revert", headers=AUTH, timeout=30)


# ============================ health / predict ============================
def test_health_ok():                       # TC-03
    j = requests.get(f"{BASE}/health", timeout=10).json()
    assert j["status"] == "ok" and j["model_loaded"] is True, j

def test_predict_hazard():                  # TC-06
    r = requests.post(f"{BASE}/predict", json={"text": "The charger caught fire and started smoking."}, timeout=20)
    j = r.json()
    assert r.status_code == 200 and j["is_hazard"] is True and j["hazard_prob"] >= 0.3, (r.status_code, j)

def test_predict_benign():                  # TC-07
    r = requests.post(f"{BASE}/predict", json={"text": "Great product, works well and arrived on time."}, timeout=20)
    j = r.json()
    assert r.status_code == 200 and j["is_hazard"] is False and j["hazard_prob"] < 0.3, (r.status_code, j)

def test_predict_schema():                  # TC-08
    j = requests.post(f"{BASE}/predict", json={"text": "battery got very hot"}, timeout=20).json()
    assert set(j) == {"hazard_prob", "is_hazard", "threshold"} and j["threshold"] == 0.3, j

def test_predict_empty_422():               # TC-09
    assert requests.post(f"{BASE}/predict", json={"text": ""}, timeout=10).status_code == 422

def test_predict_oversized_422():           # TC-10 / TC-56
    assert requests.post(f"{BASE}/predict", json={"text": "x" * 10001}, timeout=20).status_code == 422

def test_predict_missing_field_422():       # TC-11
    assert requests.post(f"{BASE}/predict", json={}, timeout=10).status_code == 422


# ============================ similar-reviews ============================
def test_similar_relevant():                # TC-12
    r = requests.post(f"{BASE}/similar-reviews", json={"text": "battery exploded while charging", "k": 3}, timeout=20)
    res = r.json()["results"]
    scores = [x["similarity"] for x in res]
    assert r.status_code == 200 and len(res) == 3 and scores == sorted(scores, reverse=True), (r.status_code, scores)

def test_similar_default_k():               # TC-13
    res = requests.post(f"{BASE}/similar-reviews", json={"text": "the cord melted"}, timeout=20).json()["results"]
    assert len(res) == 5, len(res)

def test_similar_k_low_422():               # TC-14
    assert requests.post(f"{BASE}/similar-reviews", json={"text": "smoke", "k": 0}, timeout=10).status_code == 422

def test_similar_k_high_422():              # TC-15
    assert requests.post(f"{BASE}/similar-reviews", json={"text": "smoke", "k": 51}, timeout=10).status_code == 422


# ============================ analyze ============================
def test_analyze_combined():                # TC-16
    j = requests.post(f"{BASE}/analyze", json={"text": "The charger caught fire.", "k": 3}, timeout=20).json()
    assert {"hazard_prob", "is_hazard", "threshold", "similar"} <= set(j) and len(j["similar"]) == 3, j

def test_analyze_consistency():             # TC-17
    text = {"text": "my kid almost choked on a broken piece"}
    a = requests.post(f"{BASE}/predict", json=text, timeout=20).json()["hazard_prob"]
    b = requests.post(f"{BASE}/analyze", json=text, timeout=20).json()["hazard_prob"]
    assert a == b, (a, b)


# ============================ products ============================
def test_products_ranked():                 # TC-18
    ps = requests.get(f"{BASE}/products", timeout=20).json()["products"]
    risks = [p["risk_score"] for p in ps]
    assert len(ps) > 0 and risks == sorted(risks, reverse=True), risks[:5]

def test_products_limit():                  # TC-19
    ps = requests.get(f"{BASE}/products?limit=5", timeout=20).json()["products"]
    assert len(ps) <= 5, len(ps)

def test_product_detail_valid():            # TC-20
    asin = requests.get(f"{BASE}/products?limit=1", timeout=20).json()["products"][0]["parent_asin"]
    j = requests.get(f"{BASE}/products/{asin}", timeout=20).json()
    assert j["parent_asin"] == asin and "reviews" in j and "by_year" in j, j.keys()

def test_product_detail_unknown_404():      # TC-21
    assert requests.get(f"{BASE}/products/DOESNOTEXIST", timeout=10).status_code == 404


# ============================ auth ============================
def test_ingest_no_key_401():               # TC-22
    assert requests.post(f"{BASE}/ingest", json={"text": "sharp edges"}, timeout=10).status_code == 401

def test_ingest_invalid_key_401():          # TC-23
    assert requests.post(f"{BASE}/ingest", json={"text": "sharp edges"}, headers={"X-API-Key": "wrong"}, timeout=10).status_code == 401

def test_save_no_key_401():                 # TC-29
    assert requests.post(f"{BASE}/save", timeout=10).status_code == 401

def test_revert_no_key_401():               # TC-45
    assert requests.post(f"{BASE}/revert", timeout=10).status_code == 401


# ==================== mutating (restored with /revert) ====================
def test_ingest_hazard_added():             # TC-24
    n0 = index_size()
    try:
        r = requests.post(f"{BASE}/ingest", headers=AUTH,
                          json={"text": "the space heater melted the outlet and sparked violently", "parent_asin": "B000TEST01"}, timeout=20)
        j = r.json()
        assert r.status_code == 200 and j["flagged"] is True and index_size() == n0 + 1, (r.status_code, j)
    finally:
        revert(); assert index_size() == n0, "revert did not restore index"

def test_ingest_benign_not_added():         # TC-25
    n0 = index_size()
    try:
        j = requests.post(f"{BASE}/ingest", headers=AUTH, json={"text": "nice colour, fast shipping"}, timeout=20).json()
        assert j["flagged"] is False and index_size() == n0, j
    finally:
        revert()

def test_ingest_duplicate_rejected():       # TC-26
    n0 = index_size()
    txt = {"text": "the wall charger burst into flames while plugged in overnight", "parent_asin": "B000TEST02"}
    try:
        requests.post(f"{BASE}/ingest", headers=AUTH, json=txt, timeout=20)          # first add
        j2 = requests.post(f"{BASE}/ingest", headers=AUTH, json=txt, timeout=20).json()  # duplicate
        assert j2.get("duplicate") is True, j2
    finally:
        revert(); assert index_size() == n0

def test_ingest_batch_counts():             # TC-27
    n0 = index_size()
    try:
        body = {"items": [
            {"text": "the power bank swelled up and started smoking"},
            {"text": "lovely design, works great"},
            {"text": "the outlet sparked and burned the wall"},
        ]}
        j = requests.post(f"{BASE}/ingest-batch", headers=AUTH, json=body, timeout=30).json()
        assert j["processed"] == 3 and j["flagged"] >= 1 and "index_size" in j, j
    finally:
        revert(); assert index_size() == n0

def test_ingest_batch_size_limit_422():     # TC-28
    body = {"items": [{"text": "x"} for _ in range(501)]}
    assert requests.post(f"{BASE}/ingest-batch", headers=AUTH, json=body, timeout=30).status_code == 422

def test_revert_reload():                   # TC-46
    n0 = index_size()
    requests.post(f"{BASE}/ingest", headers=AUTH, json={"text": "the toaster caught fire and burned the counter"}, timeout=20)
    j = requests.post(f"{BASE}/revert", headers=AUTH, timeout=30).json()
    assert index_size() == n0, (n0, j)


# ==================== /save (snapshot + restore on disk) ====================
def test_save_with_backup():                # TC-30
    snap = tempfile.mkdtemp(prefix="hw_stores_")
    for f in STORE_FILES:
        shutil.copy2(os.path.join(MODELS, f), os.path.join(snap, f))
    backups = os.path.join(MODELS, "backups")
    before = set(os.listdir(backups)) if os.path.isdir(backups) else set()
    try:
        requests.post(f"{BASE}/ingest", headers=AUTH, json={"text": "the extension cord overheated and melted"}, timeout=20)
        j = requests.post(f"{BASE}/save", headers=AUTH, timeout=60).json()
        after = set(os.listdir(backups)) if os.path.isdir(backups) else set()
        assert j.get("saved") is True and len(after) > len(before), (j, after - before)
    finally:
        for f in STORE_FILES:                          # restore original stores on disk
            shutil.copy2(os.path.join(snap, f), os.path.join(MODELS, f))
        revert()                                       # reload the restored stores into the engine
        shutil.rmtree(snap, ignore_errors=True)


# ============================ parse-parquet ============================
def _make_parquet(nrows):
    df = pd.DataFrame({"text": [f"sample review number {i}" for i in range(nrows)]})
    buf = io.BytesIO(); df.to_parquet(buf, index=False); return buf.getvalue()

def test_parse_parquet_valid():             # TC-37 (API part)
    r = requests.post(f"{BASE}/parse-parquet", data=_make_parquet(10), timeout=30)
    j = r.json()
    assert r.status_code == 200 and j["rows"] == 10 and j["truncated"] is False, (r.status_code, j)

def test_parse_parquet_rowcap():            # TC-44
    r = requests.post(f"{BASE}/parse-parquet", data=_make_parquet(50001), timeout=90)
    j = r.json()
    assert r.status_code == 200 and j["rows"] == 50000 and j["truncated"] is True, (r.status_code, j.get("rows"))

def test_parse_parquet_empty_400():         # TC-42 / TC-55
    assert requests.post(f"{BASE}/parse-parquet", data=b"", timeout=10).status_code == 400

def test_parse_parquet_malformed_400():     # TC-43
    assert requests.post(f"{BASE}/parse-parquet", data=b"not a parquet file", timeout=10).status_code == 400


# ============================ other HTTP codes ============================
def test_http_405():                        # TC-54
    assert requests.get(f"{BASE}/predict", timeout=10).status_code == 405

def test_docs_reachable():                  # TC-33
    r = requests.get(f"{BASE}/docs", timeout=10)
    assert r.status_code == 200 and "swagger" in r.text.lower(), r.status_code

def test_cors_headers():                    # TC-34
    r = requests.get(f"{BASE}/health", headers={"Origin": "http://localhost:3000"}, timeout=10)
    assert r.headers.get("access-control-allow-origin") == "*", dict(r.headers)

def test_http_429_behaviour():              # TC-57
    codes = [requests.post(f"{BASE}/predict", json={"text": "battery got hot"}, timeout=20).status_code for _ in range(20)]
    log.info("TC-57 burst codes: %s", sorted(set(codes)))
    assert set(codes) <= {200, 429}, codes  # documents: no rate limiting -> all 200


# ============================ response time ============================
def _median_ms(method, path, n=10, **kw):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); requests.request(method, BASE + path, timeout=30, **kw); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)

def test_rt_health():   # TC-47
    m = _median_ms("GET", "/health"); log.info("median %.0f ms", m); assert m < 200, f"{m:.0f} ms"
def test_rt_predict():  # TC-48
    m = _median_ms("POST", "/predict", json={"text": "the charger caught fire"}); log.info("median %.0f ms", m); assert m < 1500, f"{m:.0f} ms"
def test_rt_analyze():  # TC-49
    m = _median_ms("POST", "/analyze", json={"text": "battery exploded", "k": 5}); log.info("median %.0f ms", m); assert m < 3000, f"{m:.0f} ms"
def test_rt_products(): # TC-50
    m = _median_ms("GET", "/products?limit=20"); log.info("median %.0f ms", m); assert m < 1000, f"{m:.0f} ms"


TESTS = [
    ("TC-03 health ok", test_health_ok),
    ("TC-06 predict hazard", test_predict_hazard),
    ("TC-07 predict benign", test_predict_benign),
    ("TC-08 predict schema", test_predict_schema),
    ("TC-09 empty text -> 422", test_predict_empty_422),
    ("TC-10 oversized text -> 422", test_predict_oversized_422),
    ("TC-11 missing field -> 422", test_predict_missing_field_422),
    ("TC-12 similar relevant + sorted", test_similar_relevant),
    ("TC-13 similar default k=5", test_similar_default_k),
    ("TC-14 similar k=0 -> 422", test_similar_k_low_422),
    ("TC-15 similar k=51 -> 422", test_similar_k_high_422),
    ("TC-16 analyze combined", test_analyze_combined),
    ("TC-17 analyze consistency", test_analyze_consistency),
    ("TC-18 products ranked", test_products_ranked),
    ("TC-19 products limit", test_products_limit),
    ("TC-20 product detail valid", test_product_detail_valid),
    ("TC-21 unknown ASIN -> 404", test_product_detail_unknown_404),
    ("TC-22 ingest no key -> 401", test_ingest_no_key_401),
    ("TC-23 ingest invalid key -> 401", test_ingest_invalid_key_401),
    ("TC-29 save no key -> 401", test_save_no_key_401),
    ("TC-45 revert no key -> 401", test_revert_no_key_401),
    ("TC-24 ingest hazard added (revert)", test_ingest_hazard_added),
    ("TC-25 ingest benign not added", test_ingest_benign_not_added),
    ("TC-26 ingest duplicate rejected", test_ingest_duplicate_rejected),
    ("TC-27 ingest-batch counts", test_ingest_batch_counts),
    ("TC-28 batch size limit -> 422", test_ingest_batch_size_limit_422),
    ("TC-46 revert reload integrity", test_revert_reload),
    ("TC-30 save with backup (snapshot)", test_save_with_backup),
    ("TC-37 parse-parquet valid", test_parse_parquet_valid),
    ("TC-44 parse-parquet row cap", test_parse_parquet_rowcap),
    ("TC-42 parse-parquet empty -> 400", test_parse_parquet_empty_400),
    ("TC-43 parse-parquet malformed -> 400", test_parse_parquet_malformed_400),
    ("TC-54 GET /predict -> 405", test_http_405),
    ("TC-33 docs reachable", test_docs_reachable),
    ("TC-34 CORS headers", test_cors_headers),
    ("TC-57 rate-limit behaviour", test_http_429_behaviour),
    ("TC-47 response time /health", test_rt_health),
    ("TC-48 response time /predict", test_rt_predict),
    ("TC-49 response time /analyze", test_rt_analyze),
    ("TC-50 response time /products", test_rt_products),
]

if __name__ == "__main__":
    for _ in range(60):  # wait for the model to be warm
        try:
            if requests.get(f"{BASE}/health", timeout=3).json().get("status") == "ok":
                break
        except Exception:
            pass
        time.sleep(2)
    log.info("HazardWatch API automation — target %s", BASE)
    for name, fn in TESTS:
        run(name, fn)
    _banner(PASS, FAIL)
    print("  details in hazardwatch_api_tests.log  (run with HW_VERBOSE=1 to show detail live)")
    log.info("DONE — %d passed, %d failed", PASS, FAIL)
    sys.exit(1 if FAIL else 0)
