# -*- coding: utf-8 -*-
"""
HazardWatch — data-integrity automation (white-box, no server needed).

The row-alignment guard (TC-31 / TC-58) can only be triggered by forcing the
stores out of sync, which isn't possible from the HTTP layer. So these are
white-box tests: they import the engine directly, corrupt the alignment in
memory, and assert that save() refuses (the FastAPI layer turns that into HTTP
500). Works on a copy — the real stores on disk are never modified.

Same plain-script style as the example. Run:

    python test_hazardwatch_integrity.py

Covers: TC-31 / TC-58 (save aborts on misalignment), TC-05 (alignment assert).
"""
import os
import sys

# import the app package from the project root
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

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


def _load_engine():
    from app.ml import HazardEngine
    return HazardEngine()


def test_aligned_by_default():
    """TC-05 — a freshly loaded engine passes its own alignment assertion."""
    eng = _load_engine()
    eng._assert_aligned()  # must not raise
    assert eng.index.ntotal == len(eng.flagged) == eng.embeds.shape[0]


def test_save_aborts_on_misalignment():
    """TC-31 / TC-58 — save() must refuse when the stores are misaligned.

    We drop one row from the in-memory parquet so index != parquet, then call
    save(); it should raise AssertionError BEFORE writing anything. (The API
    wraps this AssertionError as HTTP 500.) The on-disk stores are untouched
    because save() asserts first.
    """
    import pandas as pd
    eng = _load_engine()
    # corrupt alignment in memory only
    eng.flagged = eng.flagged.iloc[:-1].reset_index(drop=True)
    raised = False
    try:
        eng.save()
    except AssertionError:
        raised = True
    assert raised, "save() did not abort on a misaligned store"


if __name__ == "__main__":
    print("HazardWatch integrity automation (white-box)")
    run("TC-05 engine aligned on load", test_aligned_by_default)
    run("TC-31/58 save aborts on misalignment", test_save_aborts_on_misalignment)
    _banner(PASS, FAIL)
    sys.exit(1 if FAIL else 0)
