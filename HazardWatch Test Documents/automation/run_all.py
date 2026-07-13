# -*- coding: utf-8 -*-
"""
HazardWatch — run the whole automation suite in one go.

Runs, in order:
  1. test_hazardwatch_integrity.py   (white-box; no server needed)
  2. test_hazardwatch_lifecycle.py   (starts/stops its own server)
  3. test_hazardwatch_reliability.py (starts/stops its own broken-store servers)
  4. test_hazardwatch_api.py         (needs a running server)
  5. test_hazardwatch_load.py        (needs a running server)
  6. test_hazardwatch_ai.py          (needs a running server)
  7. test_hazardwatch_ui.py          (needs a running server + Chrome)

Integrity, lifecycle and reliability manage their own server and need port 8000
FREE while they run. The api / load / ai / ui suites instead need the API already
running:
    .venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

Because of that split, the cleanest way to run the full set is still the documented
per-file commands in README.md. run_all.py is a convenience for the server-backed
suites — start the server, then run it (the self-managing suites will just report a
port conflict if a server is already up, which is expected).

Usage:
    python run_all.py               # run everything (UI headed)
    HEADLESS=1 python run_all.py    # UI headless
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SUITES = [
    "test_hazardwatch_integrity.py",
    "test_hazardwatch_lifecycle.py",
    "test_hazardwatch_reliability.py",
    "test_hazardwatch_api.py",
    "test_hazardwatch_load.py",
    "test_hazardwatch_ai.py",
    "test_hazardwatch_ui.py",
]

if os.name == "nt":
    os.system("")  # enable ANSI colour codes in Windows terminals


def _clr(t, c):
    return f"\033[{c}m{t}\033[0m"


if __name__ == "__main__":
    rc = 0
    results = []
    for s in SUITES:
        print("\n" + _clr("═" * 70, "1;36") + f"\n  RUNNING {s}\n" + _clr("═" * 70, "1;36"))
        r = subprocess.run([PY, os.path.join(HERE, s)], cwd=HERE)
        rc |= r.returncode
        results.append((s, r.returncode == 0))
    print("\n" + "═" * 70)
    for s, ok in results:
        mark = _clr(" PASS ", "1;30;42") if ok else _clr(" FAIL ", "1;97;41")
        print(f"  {mark}  {s}")
    overall = _clr("  ALL SUITES PASSED  ", "1;30;42") if rc == 0 else _clr("  SOME SUITES FAILED  ", "1;97;41")
    print("═" * 70 + f"\n  {overall}\n" + "═" * 70)
    sys.exit(rc)
