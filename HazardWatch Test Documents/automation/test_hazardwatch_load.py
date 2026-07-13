# -*- coding: utf-8 -*-
"""
HazardWatch — load / concurrency automation.

Simulates several clients using the system at the same time (the "open it in
multiple browsers" scenario, done here with concurrent worker threads for a
tighter, repeatable load). Confirms the API stays responsive under concurrency
and that concurrent ingestion keeps the stores aligned.

Same plain-script style as the example. Run (server must be up):

    python test_hazardwatch_load.py

Covers: TC-51 (concurrency sweep 5->10->25->50->100 clients on /analyze, with
latency percentiles), TC-52 (concurrent ingestion stays aligned), TC-53 (sustained
soak), TC-59 (bulk volume / throughput — loading thousands of reviews, timed, with
server memory + CPU capture when psutil is installed).

TC-59 answers "can the API handle loading thousands of reviews, and how long
does it take?" — it loops POST /ingest-batch (the API caps a batch at 500 items)
until it reaches each target size, timing every batch. The generated reviews are
a realistic ~35%-hazard / 65%-benign mix so the expensive path (classify -> embed
-> add to the FAISS index) is actually exercised; an all-benign load would only
measure the classifier. Everything is in-memory until /save, so the test reverts
at the end and leaves the on-disk stores untouched.

Tune with env vars:
    HW_CONC_TIERS     concurrency levels for TC-51    (default "5,10,25,50")
    HW_CONC_REQ       requests fired per tier         (default "40")
    HW_SOAK_SECONDS   TC-53 soak duration in seconds  (default "60")
    HW_VOLUME_TIERS   comma-separated target sizes    (default "1000";
                      set "1000,5000,10000" to see the throughput curve)
    HW_HAZARD_FRAC    fraction of hazard-worded rows  (default "0.35")
    HW_MIN_TPS        fail the tier if throughput drops below this reviews/sec
                      (default "0" = log only, don't enforce)
"""
import concurrent.futures as cf
import logging
import math
import os
import statistics
import sys
import threading
import time
from urllib.parse import urlparse

import requests

BASE = os.environ.get("HW_URL", "http://127.0.0.1:8000").rstrip("/")
KEY = os.environ.get("HAZARDWATCH_API_KEY", "dev-key-hazardwatch")
AUTH = {"X-API-Key": KEY}
# The key tier summaries print by default. Set HW_VERBOSE=1 to stream ALL detail
# (incl. per-test log lines) to the console live; full detail is always in the .log.
VERBOSE = os.environ.get("HW_VERBOSE", "") not in ("", "0", "false", "False")
_handlers = [logging.FileHandler("hazardwatch_load_tests.log")]
if VERBOSE:
    _con = logging.StreamHandler()
    _con.setFormatter(logging.Formatter("      %(message)s"))
    _con.addFilter(lambda r: not r.getMessage().startswith(("PASS -", "FAIL -", "ERROR -", "DONE")))
    _handlers.append(_con)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    handlers=_handlers)
log = logging.getLogger("hw-load")
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

# Concurrency sweep (TC-51): how many simultaneous clients to test, and how many
# requests to fire at each level. The request count is fixed across tiers so the
# latency numbers are comparable (same sample, increasing parallelism).
CONC_TIERS = [int(x) for x in os.environ.get("HW_CONC_TIERS", "5,10,25,50,100").split(",") if x.strip()]
CONC_REQ = int(os.environ.get("HW_CONC_REQ", "40"))


def _pct(vals, p):
    """Nearest-rank percentile of a list of numbers (p in 0..100)."""
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1)]


def _find_server_proc():
    """Locate the uvicorn process listening on the target port (needs psutil). None if unavailable."""
    try:
        import psutil
    except ImportError:
        log.info("psutil not installed — skipping server memory/CPU capture (pip install psutil)")
        return None
    port = urlparse(BASE).port or 8000
    try:
        for c in psutil.net_connections(kind="inet"):
            if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid:
                return psutil.Process(c.pid)
    except Exception as e:  # noqa - e.g. permission denied enumerating connections
        log.info("could not locate server process for resource capture: %s", e)
    return None


class _ResourceSampler(threading.Thread):
    """Background sampler of a process's RSS (MB) and CPU%% while a load runs."""
    def __init__(self, proc, interval=0.5):
        super().__init__(daemon=True)
        self.proc, self.interval = proc, interval
        self._stop_evt = threading.Event()   # not '_stop' — Thread has its own _stop()
        self.peak_rss = 0.0
        self.cpu = []

    def run(self):
        if not self.proc:
            return
        try:
            self.proc.cpu_percent(None)  # prime the first (meaningless) reading
        except Exception:
            return
        while not self._stop_evt.is_set():
            try:
                self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss / 1e6)
                self.cpu.append(self.proc.cpu_percent(None))
            except Exception:
                break
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)

    def summary(self):
        peak_cpu = max(self.cpu) if self.cpu else 0.0
        mean_cpu = statistics.mean(self.cpu) if self.cpu else 0.0
        return self.peak_rss, mean_cpu, peak_cpu


def run(name, fn):
    global PASS, FAIL
    try:
        fn(); PASS += 1; _pass(name); log.info("PASS - %s", name)
    except AssertionError as e:
        FAIL += 1; _fail("FAIL", name, e); log.error("FAIL - %s -> %s", name, e)
    except Exception as e:  # noqa
        FAIL += 1; _fail("ERR ", name, f"{type(e).__name__}: {e}"); log.error("ERROR - %s -> %s: %s", name, type(e).__name__, e)


def index_size():
    return requests.get(f"{BASE}/health", timeout=10).json()["index_size"]


def test_concurrent_analyze():
    """TC-51 — sweep concurrency (5 -> 10 -> 25 -> 50 clients) on /analyze.

    At each tier, CONC_REQ requests are fired through that many worker threads (so
    ~N are in flight at once). Every response must be 200 with a valid body and zero
    errors/timeouts; per-tier latency (median / p95 / max) is logged so you can show
    how response time grows as simultaneous users increase. The fixed request count
    keeps the latency numbers comparable across tiers.
    """
    texts = ["the charger caught fire", "nice product works well",
             "battery swelled and leaked", "arrived on time, great value"]

    def one(i):
        t = texts[i % len(texts)]
        t0 = time.perf_counter()
        try:
            # generous timeout: at high concurrency on a single CPU worker, queued
            # requests wait a long time — we want the real latency, not a timeout error
            r = requests.post(f"{BASE}/analyze", json={"text": t, "k": 3}, timeout=180)
            lat = (time.perf_counter() - t0) * 1000
            return r.status_code, ("hazard_prob" in r.json()), lat
        except Exception as e:  # noqa - timeout / connection error counts as a failure
            return type(e).__name__, False, (time.perf_counter() - t0) * 1000

    for n in CONC_TIERS:
        reqs = max(CONC_REQ, n)          # fire >= n requests so concurrency really reaches n
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(one, range(reqs)))
        dt = time.time() - t0
        codes = {c for c, _, _ in results}
        ok_bodies = all(ok for _, ok, _ in results)
        lats = [l for _, _, l in results]
        med, p95, mx = statistics.median(lats), _pct(lats, 95), max(lats)
        log.info("TC-51 conc=%d: %d reqs in %.1fs, codes=%s, latency med/p95/max %.0f/%.0f/%.0f ms",
                 n, reqs, dt, codes, med, p95, mx)
        if not VERBOSE:  # in verbose the line above is already streamed to the console
            print(f"      conc={n:>3}: {reqs} reqs in {dt:.1f}s  "
                  f"latency med {med:.0f} / p95 {p95:.0f} / max {mx:.0f} ms  codes={codes}")
        assert codes == {200} and ok_bodies, f"conc={n}: {codes}"


def test_concurrent_ingest_stays_aligned():
    """TC-52 — concurrent ingests keep the row-alignment invariant; then revert."""
    n0 = index_size()
    hazards = [f"the {w} overheated and burst into flames unit {i}"
               for i, w in enumerate(["charger", "heater", "power bank", "adapter", "battery"])]

    def one(txt):
        return requests.post(f"{BASE}/ingest", headers=AUTH,
                             json={"text": txt, "parent_asin": "B000LOAD"}, timeout=30).status_code

    try:
        with cf.ThreadPoolExecutor(max_workers=5) as ex:
            codes = list(ex.map(one, hazards))
        # engine reports its own alignment via /health; sizes must be consistent
        h = requests.get(f"{BASE}/health", timeout=10).json()
        assert set(codes) == {200}, codes
        # index grew by the flagged, non-duplicate count and is internally consistent
        assert h["index_size"] >= n0, (n0, h["index_size"])
        log.info("TC-52 concurrent ingest ok; index %d -> %d", n0, h["index_size"])
    finally:
        requests.post(f"{BASE}/revert", headers=AUTH, timeout=30)
        assert index_size() == n0, "revert did not restore the index after load"


def test_sustained_load():
    """TC-53 — sustained requests over a time window stay healthy (no errors/drift)."""
    DURATION = int(os.environ.get("HW_SOAK_SECONDS", "60"))  # raise for a longer soak
    end = time.time() + DURATION
    n = errors = 0
    while time.time() < end:
        try:
            if requests.post(f"{BASE}/predict", json={"text": "battery got hot"}, timeout=10).status_code != 200:
                errors += 1
        except Exception:
            errors += 1
        n += 1
    assert n > 0 and errors == 0, f"{errors} errors over {n} requests"
    log.info("TC-53 sustained: %d requests in %ds, %d errors", n, DURATION, errors)


# ============================ bulk volume / throughput ============================
BATCH = 500                                          # API caps /ingest-batch at 500 items
HAZARD_FRAC = float(os.environ.get("HW_HAZARD_FRAC", "0.35"))
MIN_TPS = float(os.environ.get("HW_MIN_TPS", "0"))   # 0 = log throughput but don't enforce
TIERS = [int(x) for x in os.environ.get("HW_VOLUME_TIERS", "1000").split(",") if x.strip()]

_HAZ_OBJ = ["charger", "power bank", "space heater", "extension cord", "wall adapter",
            "laptop battery", "hair dryer", "electric kettle", "phone cable", "toaster"]
_HAZ_ACT = ["burst into flames", "overheated and started smoking", "melted the outlet",
            "sparked violently and burned", "swelled up and leaked acid", "caught fire overnight"]
_BENIGN = ["works great and arrived on time", "nice design, good value for money",
           "exactly as described, very happy", "fast shipping and easy to use",
           "great colour and solid build quality", "does the job, no complaints"]


def _make_reviews(n):
    """n mixed reviews; ~HAZARD_FRAC are unique hazard-worded (so the index actually grows)."""
    out = []
    haz_every = max(1, round(1 / HAZARD_FRAC)) if HAZARD_FRAC > 0 else 10 ** 9
    for i in range(n):
        if i % haz_every == 0:
            obj, act = _HAZ_OBJ[i % len(_HAZ_OBJ)], _HAZ_ACT[i % len(_HAZ_ACT)]
            text = f"the {obj} {act} — serial LT{i:07d}, do not buy"   # id keeps each unique
        else:
            text = f"{_BENIGN[i % len(_BENIGN)]} (order {i:07d})"
        out.append({"text": text, "parent_asin": f"B00VOL{i % 50:04d}"})
    return out


def _ingest_volume(target):
    """Push `target` reviews through /ingest-batch in 500-item batches; return timing + resource stats."""
    items = _make_reviews(target)
    batch_ms, sent, flagged, errors = [], 0, 0, 0
    sampler = _ResourceSampler(_find_server_proc())   # captures server RSS/CPU during the load
    sampler.start()
    t0 = time.time()
    for s in range(0, target, BATCH):
        chunk = items[s:s + BATCH]
        b0 = time.perf_counter()
        try:
            r = requests.post(f"{BASE}/ingest-batch", headers=AUTH, json={"items": chunk}, timeout=120)
            batch_ms.append((time.perf_counter() - b0) * 1000)
            if r.status_code != 200:
                errors += 1; log.error("TC-59 batch @%d -> HTTP %d", s, r.status_code); continue
            j = r.json(); sent += j.get("processed", 0); flagged += j.get("flagged", 0)
        except Exception as e:  # noqa
            errors += 1; log.error("TC-59 batch @%d -> %s %s", s, type(e).__name__, e)
    dt = time.time() - t0
    sampler.stop()
    peak_rss, mean_cpu, peak_cpu = sampler.summary()
    return {"dt": dt, "sent": sent, "flagged": flagged, "errors": errors, "batch_ms": batch_ms,
            "tps": sent / dt if dt else 0.0,
            "peak_rss": peak_rss, "mean_cpu": mean_cpu, "peak_cpu": peak_cpu}


def test_bulk_volume_throughput():
    """TC-59 — load each target volume of reviews, timing throughput; revert to clean up."""
    n0 = index_size()
    try:
        for target in TIERS:
            requests.post(f"{BASE}/revert", headers=AUTH, timeout=30)   # every tier starts from baseline
            base = index_size()
            st = _ingest_volume(target)
            grew = index_size() - base
            med = statistics.median(st["batch_ms"]) if st["batch_ms"] else 0.0
            mx = max(st["batch_ms"]) if st["batch_ms"] else 0.0
            res = (f", server peak RSS {st['peak_rss']:.0f} MB, CPU mean/peak "
                   f"{st['mean_cpu']:.0f}/{st['peak_cpu']:.0f}%") if st["peak_rss"] else ""
            log.info("TC-59 N=%d: %.1fs, %.0f reviews/s, batch med/max %.0f/%.0f ms, "
                     "flagged=%d, index grew +%d, errors=%d%s",
                     target, st["dt"], st["tps"], med, mx, st["flagged"], grew, st["errors"], res)
            if not VERBOSE:  # in verbose the line above is already streamed to the console
                print(f"      N={target}: {st['dt']:.1f}s  {st['tps']:.0f} rev/s  "
                      f"batch med {med:.0f}ms / max {mx:.0f}ms  index +{grew}{res}")
            assert st["errors"] == 0, f"{st['errors']} batch errors at N={target}"
            assert st["sent"] == target, (st["sent"], target)
            if MIN_TPS:
                assert st["tps"] >= MIN_TPS, f"throughput {st['tps']:.0f} < {MIN_TPS} rev/s at N={target}"
    finally:
        requests.post(f"{BASE}/revert", headers=AUTH, timeout=60)
        assert index_size() == n0, "revert did not restore the index after the volume test"


if __name__ == "__main__":
    log.info("HazardWatch load automation — target %s", BASE)
    run(f"TC-51 concurrency sweep (clients={CONC_TIERS})", test_concurrent_analyze)
    run("TC-52 concurrent ingestion stays aligned", test_concurrent_ingest_stays_aligned)
    run("TC-53 sustained load", test_sustained_load)
    run(f"TC-59 bulk volume / throughput (tiers={TIERS})", test_bulk_volume_throughput)
    _banner(PASS, FAIL)
    print("  full detail in hazardwatch_load_tests.log  (run with HW_VERBOSE=1 for more live output)")
    sys.exit(1 if FAIL else 0)
