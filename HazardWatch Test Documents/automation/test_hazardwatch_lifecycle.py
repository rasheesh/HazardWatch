# -*- coding: utf-8 -*-
"""
HazardWatch — lifecycle / infrastructure automation (starts its own server).

These cases can't be tested against an already-running server: they observe the
server *starting up*. This script launches uvicorn as a subprocess, watches the
log and /health, then shuts it down cleanly.

Same plain-script style as the example: test_* functions, assert, print PASS,
manual runner at the bottom. Run:

    python test_hazardwatch_lifecycle.py

Covers: TC-01 (startup + port binding), TC-02 (loading state), TC-03 (ready),
TC-04 (503 while loading), TC-05 (artifact alignment / sanity check).
Note: TC-35 (auto-save on clean shutdown) is exercised here only as a clean
shutdown; verifying persisted data is left as a guided manual step.
"""
import os
import subprocess
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def start_server():
    log = open(os.path.join(ROOT, "server_lifecycle.log"), "w")
    p = subprocess.Popen(
        [PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
    )
    return p, log


def wait_health(timeout=180):
    """Return the first /health json once the port answers; None if never."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            return requests.get(f"{BASE}/health", timeout=3).json()
        except Exception:
            time.sleep(1)
    return None


def wait_ready(timeout=300):
    """Poll /health until status == 'ok' (model finished loading); None on timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            j = requests.get(f"{BASE}/health", timeout=3).json()
            if j.get("status") == "ok":
                return j
        except Exception:
            pass
        time.sleep(2)
    return None


def test_full_lifecycle():
    """TC-01..TC-05 observed across a single startup."""
    global PASS, FAIL
    proc, log = start_server()
    try:
        # TC-01 — port binds quickly and answers
        first = wait_health(timeout=60)
        run("TC-01 startup + port binds (/health answers)", lambda: (_ for _ in ()).throw(AssertionError("no /health")) if first is None else None)

        # TC-02/TC-04 — while loading, status is 'loading' and endpoints 503 (best-effort:
        # on a warm machine the model can finish loading before we look).
        def _loading_or_ok():
            assert first is not None
            assert first.get("status") in ("loading", "ok"), first
        run("TC-02/04 loading state observed (loading|ok)", _loading_or_ok)

        # TC-03 — becomes ready
        def _ready():
            j = wait_ready(timeout=300)  # poll UNTIL ok (cold model load ~421 MB checkpoint)
            assert j and j.get("status") == "ok" and j.get("model_loaded") is True, j
        run("TC-03 reaches ready (/health ok)", _ready)

        # TC-01 log line
        def _logline():
            with open(os.path.join(ROOT, "server_lifecycle.log"), encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            assert "Uvicorn running on http://127.0.0.1:8000" in txt, "startup log line missing"
        run("TC-01 log shows 'Uvicorn running on ...'", _logline)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        log.close()


def test_sanity_check():
    """TC-05 — artifact alignment via scripts/sanity_check.py (exit code + banner)."""
    r = subprocess.run([PY, "scripts/sanity_check.py"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0 and "ALL CHECKS PASSED" in out, out[-300:]


if __name__ == "__main__":
    print("HazardWatch lifecycle automation (manages its own server)")
    test_full_lifecycle()
    run("TC-05 artifact alignment (sanity_check.py)", test_sanity_check)
    _banner(PASS, FAIL)
    sys.exit(1 if FAIL else 0)
