# -*- coding: utf-8 -*-
"""
HazardWatch — Selenium UI automation (prototype front-end).

Follows the team's automation method (Selenium + Python, webdriver-manager for
ChromeDriver, explicit waits, assert + print, driver.quit() in finally, logging).
Automates the prototype at http://127.0.0.1:8000/  (start the server first).

Run:
    pip install selenium webdriver-manager
    # in another terminal, start the API:
    #   .venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
    python test_hazardwatch_ui.py            # headed
    HEADLESS=1 python test_hazardwatch_ui.py # headless (CI / unattended)

Maps to test cases: TC-06/07 (analyze), TC-32 (UI), TC-36..TC-41 (live feed),
TC-18..TC-20 (product monitor), TC-77..TC-79 (bulk-import formats TSV/JSONL/TXT).
"""
import logging
import os
import sys

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

BASE_URL = os.environ.get("HW_URL", "http://127.0.0.1:8000/")
HEADLESS = os.environ.get("HEADLESS", "0") == "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler("hazardwatch_ui_tests.log")],
)
log = logging.getLogger("hazardwatch")

_passed = _failed = 0


# ----------------------------- pretty console results -----------------------------
if os.name == "nt":
    os.system("")  # enable ANSI colour codes in Windows terminals


def _clr(t, c):
    return f"\033[{c}m{t}\033[0m"


def _banner(passed, failed):
    total = passed + failed
    bar = "=" * 62
    if failed == 0:
        tag = _clr(f"  ALL {total} TESTS PASSED  ", "1;30;42")
    else:
        tag = _clr(f"  {failed} FAILED / {total}  —  {passed} passed  ", "1;97;41")
    print(f"\n{bar}\n  {tag}\n{bar}")


def create_driver():
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1400,1000")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    driver.get(BASE_URL)
    # wait for the app shell to be ready
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "reviewText")))
    return driver


def _ok(name):
    global _passed
    _passed += 1
    print(f"  {_clr(' PASS ', '1;30;42')}  {name}")
    log.info("PASS - %s", name)


def _bad(name, err):
    global _failed
    _failed += 1
    print(f"  {_clr(' FAIL ', '1;97;41')}  {name}")
    print(f"             {_clr('->', '1;31')} {type(err).__name__}: {str(err)[:200]}")
    log.error("FAIL - %s -> %s: %s", name, type(err).__name__, str(err)[:200])


# --------------------------------------------------------------- Analyze tab
def test_analyze_hazard():
    """TC-06 — a hazard review is flagged in the UI."""
    d = create_driver()
    try:
        d.find_element(By.ID, "reviewText").send_keys(
            "The charger caught fire and started smoking while charging overnight."
        )
        d.find_element(By.ID, "analyzeBtn").click()
        WebDriverWait(d, 30).until(
            lambda drv: drv.find_element(By.ID, "probnum").text.strip() not in ("", "—", "--")
        )
        verdict = d.find_element(By.ID, "badgeText").text
        prob = d.find_element(By.ID, "probnum").text
        assert "HAZARD" in verdict.upper(), f"expected a hazard verdict, got {verdict!r}"
        _ok(f"test_analyze_hazard (verdict={verdict!r}, prob={prob!r})")
    except Exception as e:
        _bad("test_analyze_hazard", e)
    finally:
        d.quit()


def test_analyze_benign():
    """TC-07 — a benign review is not flagged in the UI."""
    d = create_driver()
    try:
        d.find_element(By.ID, "reviewText").send_keys(
            "Great product, works well and arrived on time. Very happy."
        )
        d.find_element(By.ID, "analyzeBtn").click()
        WebDriverWait(d, 30).until(
            lambda drv: drv.find_element(By.ID, "probnum").text.strip() not in ("", "—", "--")
        )
        verdict = d.find_element(By.ID, "badgeText").text
        assert "NO HAZARD" in verdict.upper() or "FLAGGED" not in verdict.upper(), \
            f"benign review was flagged: {verdict!r}"
        _ok(f"test_analyze_benign (verdict={verdict!r})")
    except Exception as e:
        _bad("test_analyze_benign", e)
    finally:
        d.quit()


# --------------------------------------------------------------- Tabs / shell
def test_tabs_present():
    """TC-32 — the three prototype tabs render."""
    d = create_driver()
    try:
        for tid, label in [("tab-analyze", "Analyze"),
                           ("tab-livefeed", "Live feed"),
                           ("tab-dashboard", "Product monitor")]:
            el = d.find_element(By.ID, tid)
            assert label.lower() in el.text.lower(), f"{tid} text was {el.text!r}"
        _ok("test_tabs_present (Analyze / Live feed / Product monitor)")
    except Exception as e:
        _bad("test_tabs_present", e)
    finally:
        d.quit()


# --------------------------------------------------------------- Live feed tab
def test_livefeed_controls_present():
    """TC-36..TC-41 — the Live feed tab exposes the file input and feed controls."""
    d = create_driver()
    try:
        d.find_element(By.ID, "tab-livefeed").click()
        WebDriverWait(d, 10).until(EC.presence_of_element_located((By.ID, "fileInput")))
        for cid in ["fileInput", "feedStart", "feedStop", "feedSave", "feedRevert"]:
            d.find_element(By.ID, cid)
        _ok("test_livefeed_controls_present (fileInput, feedStart/Stop/Save/Revert)")
    except Exception as e:
        _bad("test_livefeed_controls_present", e)
    finally:
        d.quit()


def test_livefeed_load_file():
    """TC-36 — loading a CSV reviews file with a text column populates the feed.

    Writes a tiny CSV to disk and sends its path to the hidden <input type=file>.
    """
    import csv
    import tempfile
    d = create_driver()
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text"])
            w.writerow(["The charger caught fire and melted the outlet."])
            w.writerow(["Nice colour, arrived quickly, works fine."])
            w.writerow(["The space heater sparked and started smoking."])
        d.find_element(By.ID, "tab-livefeed").click()
        WebDriverWait(d, 10).until(EC.presence_of_element_located((By.ID, "fileInput")))
        d.find_element(By.ID, "fileInput").send_keys(path)
        # feedStart becomes enabled once a file is parsed and rows are queued
        WebDriverWait(d, 15).until(
            lambda drv: drv.find_element(By.ID, "feedStart").is_enabled()
        )
        _ok("test_livefeed_load_file (CSV parsed, feed ready to start)")
    except Exception as e:
        _bad("test_livefeed_load_file", e)
    finally:
        d.quit()
        if path and os.path.exists(path):
            os.remove(path)


# ------------------------------------------------- Live feed: accepted file formats
def _livefeed_format_ready(suffix, content, case):
    """Write `content` to a temp file, load it into the Live feed, and assert it
    parses (feedStart becomes enabled). Covers the client-side format parsers that
    have no backend endpoint (CSV/TSV/JSON/JSONL/TXT are parsed in the browser;
    only Parquet posts to /parse-parquet, covered by the API suite TC-37)."""
    import tempfile
    d = create_driver()
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(content)
        d.find_element(By.ID, "tab-livefeed").click()
        WebDriverWait(d, 10).until(EC.presence_of_element_located((By.ID, "fileInput")))
        d.find_element(By.ID, "fileInput").send_keys(path)
        WebDriverWait(d, 15).until(lambda drv: drv.find_element(By.ID, "feedStart").is_enabled())
        _ok(case)
    except Exception as e:
        _bad(case, e)
    finally:
        d.quit()
        if path and os.path.exists(path):
            os.remove(path)


def test_livefeed_load_tsv():
    """TC-77 — a tab-separated (.tsv) reviews file parses and queues rows."""
    content = ("text\trating\n"
               "The charger caught fire and melted the outlet.\t1\n"
               "Nice colour, arrived quickly, works fine.\t5\n"
               "The space heater sparked and started smoking.\t1\n")
    _livefeed_format_ready(".tsv", content, "test_livefeed_load_tsv (TC-77 TSV parsed, feed ready)")


def test_livefeed_load_jsonl():
    """TC-78 — a JSON Lines (.jsonl) reviews file parses and queues rows."""
    content = ('{"text": "the battery swelled up and started smoking"}\n'
               '{"text": "lovely design, works great and shipped fast"}\n'
               '{"text": "the wall charger burst into flames overnight"}\n')
    _livefeed_format_ready(".jsonl", content, "test_livefeed_load_jsonl (TC-78 JSONL parsed, feed ready)")


def test_livefeed_load_txt():
    """TC-79 — a plain-text (.txt) file with one review per line parses and queues rows."""
    content = ("the power bank overheated and burned my hand\n"
               "nice product, fast shipping and easy to use\n"
               "the cord melted and filled the room with smoke\n")
    _livefeed_format_ready(".txt", content, "test_livefeed_load_txt (TC-79 TXT one-per-line parsed, feed ready)")


# --------------------------------------------------------------- Live feed full flow
def test_livefeed_full_flow():
    """TC-38/39/41 — load a file, stream (counters advance), stop, then revert.

    Reverting at the end rolls the index back to the last saved state, so this
    test leaves your data unchanged.
    """
    import csv
    import tempfile
    d = create_driver()
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text"])
            for i in range(12):
                w.writerow([f"the charger unit {i} caught fire and melted the outlet"])
                w.writerow([f"nice product {i}, works great and shipped fast"])
        d.find_element(By.ID, "tab-livefeed").click()
        WebDriverWait(d, 10).until(EC.presence_of_element_located((By.ID, "fileInput")))
        d.find_element(By.ID, "fileInput").send_keys(path)
        WebDriverWait(d, 15).until(lambda drv: drv.find_element(By.ID, "feedStart").is_enabled())

        d.find_element(By.ID, "feedStart").click()                       # TC-38 start
        # wait until the Processed counter advances past zero
        WebDriverWait(d, 30).until(
            lambda drv: (drv.find_element(By.ID, "fProcessed").text.strip() or "0") not in ("", "0")
        )
        processed = d.find_element(By.ID, "fProcessed").text
        d.find_element(By.ID, "feedStop").click()                        # TC-39 stop
        # revert to restore the index (TC-41) — button enables once there are unsaved changes
        WebDriverWait(d, 15).until(lambda drv: drv.find_element(By.ID, "feedRevert").is_enabled())
        d.find_element(By.ID, "feedRevert").click()
        # Revert asks for confirmation via a JS confirm() dialog — accept it (per the
        # waits/alerts lecture: switch to the alert and accept).
        WebDriverWait(d, 5).until(EC.alert_is_present())
        d.switch_to.alert.accept()
        WebDriverWait(d, 20).until(lambda drv: not drv.find_element(By.ID, "feedRevert").is_enabled())
        _ok(f"test_livefeed_full_flow (streamed processed={processed!r}, stopped, reverted)")
    except Exception as e:
        _bad("test_livefeed_full_flow", e)
    finally:
        d.quit()
        if path and os.path.exists(path):
            os.remove(path)


# --------------------------------------------------------------- Product monitor
def test_product_monitor_renders():
    """TC-18 — the Product monitor tab renders the product table."""
    d = create_driver()
    try:
        d.find_element(By.ID, "tab-dashboard").click()
        WebDriverWait(d, 15).until(EC.presence_of_element_located((By.ID, "productRows")))
        _ok("test_product_monitor_renders (productRows present)")
    except Exception as e:
        _bad("test_product_monitor_renders", e)
    finally:
        d.quit()


TESTS = [
    test_analyze_hazard,
    test_analyze_benign,
    test_tabs_present,
    test_livefeed_controls_present,
    test_livefeed_load_file,
    test_livefeed_load_tsv,
    test_livefeed_load_jsonl,
    test_livefeed_load_txt,
    test_livefeed_full_flow,
    test_product_monitor_renders,
]

if __name__ == "__main__":
    log.info("HazardWatch UI automation — target %s (headless=%s)", BASE_URL, HEADLESS)
    for t in TESTS:
        t()
    _banner(_passed, _failed)
    print("  details saved to hazardwatch_ui_tests.log")
    log.info("DONE — %d passed, %d failed", _passed, _failed)
    sys.exit(1 if _failed else 0)
