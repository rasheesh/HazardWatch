# -*- coding: utf-8 -*-
"""
HazardWatch — reliability / failure-injection automation (starts its own server).

These cases deliberately break a store or model file, start the server against the
broken artifact, and assert the API **fails gracefully** rather than crashing or
serving garbage: /health reports status "error" with model_loaded false, and the
prediction endpoints return HTTP 500 (see app/main.py get_engine / _load_engine).
Every case snapshots the file first and **restores it in a finally block**, so the
models/ directory is left exactly as it was.

Same plain-script style: test_* functions, assert, print PASS/FAIL, manual runner.

    python test_hazardwatch_reliability.py       (manages its own server; nothing else must be on :8000)

The reviewer's "database unavailable" maps here to the file stores (HazardWatch has
no SQL DB — it uses parquet + FAISS + numpy). "Disk full during save" and "API
restart mid-request" are left as guided manual checks (not safely automatable).

Covers: TC-73 (retrieval index missing), TC-74 (corrupted embeddings),
TC-75 (flagged store missing), TC-76 (model checkpoint missing).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(ROOT, "models")
PY = sys.executable
BASE = "http://127.0.0.1:8000"
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
    global PASS, FAIL
    try:
        fn(); PASS += 1; _pass(name)
    except AssertionError as e:
        FAIL += 1; _fail("FAIL", name, e)
    except Exception as e:  # noqa
        FAIL += 1; _fail("ERR ", name, f"{type(e).__name__}: {e}")


def _start():
    log = open(os.path.join(ROOT, "server_reliability.log"), "w")
    p = subprocess.Popen(
        [PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
    )
    return p, log


def _stop(proc, log):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
    log.close()


def _wait_resolved(timeout=120):
    """Poll /health until the loader resolves to 'ok' or 'error' (not 'loading')."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            j = requests.get(f"{BASE}/health", timeout=3).json()
            if j.get("status") in ("ok", "error"):
                return j
        except Exception:
            pass
        time.sleep(1)
    return None


def _assert_graceful_failure(broken_desc):
    """Start a server (with the store already broken) and assert it fails cleanly."""
    proc, log = _start()
    try:
        j = _wait_resolved(timeout=120)
        assert j is not None, "port never answered /health"
        assert j.get("status") == "error" and j.get("model_loaded") is False, j
        # endpoints must refuse cleanly with 500, not 200 / not a hang
        r = requests.post(f"{BASE}/predict", json={"text": "the charger caught fire"}, timeout=15)
        assert r.status_code == 500, f"expected 500 while {broken_desc}, got {r.status_code}"
    finally:
        _stop(proc, log)


def _break_by_rename(fname):
    """Move a store file aside; return a restore() callable."""
    src = os.path.join(MODELS, fname)
    stash_dir = tempfile.mkdtemp(prefix="hw_rel_")
    stash = os.path.join(stash_dir, fname)
    shutil.move(src, stash)

    def restore():
        shutil.move(stash, src)
        shutil.rmtree(stash_dir, ignore_errors=True)
    return restore


def _break_by_corrupt(fname):
    """Snapshot then overwrite a store file with junk bytes; return restore() callable."""
    src = os.path.join(MODELS, fname)
    stash_dir = tempfile.mkdtemp(prefix="hw_rel_")
    shutil.copy2(src, os.path.join(stash_dir, fname))
    with open(src, "wb") as f:
        f.write(b"this is not a valid artifact \x00\x01\x02 corrupted" * 100)

    def restore():
        shutil.copy2(os.path.join(stash_dir, fname), src)
        shutil.rmtree(stash_dir, ignore_errors=True)
    return restore


def test_retrieval_index_missing():         # TC-73
    restore = _break_by_rename("retrieval.index")
    try:
        _assert_graceful_failure("retrieval.index missing")
    finally:
        restore()


def test_corrupted_embeddings():            # TC-74
    restore = _break_by_corrupt("retrieval_embeddings.npy")
    try:
        _assert_graceful_failure("embeddings corrupted")
    finally:
        restore()


def test_flagged_store_missing():           # TC-75
    restore = _break_by_rename("flagged_reviews.parquet")
    try:
        _assert_graceful_failure("flagged_reviews.parquet missing")
    finally:
        restore()


def test_model_checkpoint_missing():        # TC-76
    restore = _break_by_rename("bert_bigru_stage1.pt")
    try:
        _assert_graceful_failure("model checkpoint missing")
    finally:
        restore()


def _port_busy():
    try:
        requests.get(f"{BASE}/health", timeout=2)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    print("HazardWatch reliability automation (manages its own server; breaks + restores store files)")
    if _port_busy():
        print("ERROR - something is already listening on :8000 — stop it first "
              "(this suite starts its own broken-store servers).")
        sys.exit(2)
    run("TC-73 retrieval index missing -> graceful", test_retrieval_index_missing)
    run("TC-74 corrupted embeddings -> graceful", test_corrupted_embeddings)
    run("TC-75 flagged store missing -> graceful", test_flagged_store_missing)
    run("TC-76 model checkpoint missing -> graceful", test_model_checkpoint_missing)
    _banner(PASS, FAIL)
    sys.exit(1 if FAIL else 0)
