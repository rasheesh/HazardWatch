# HazardWatch — Test Automation (Selenium + API)

Automated tests for HazardWatch, written in the **same plain-script style as the
team's automation example** (`test automation example.docx`): `test_*` functions,
`assert`, `print PASS/FAIL`, `driver.quit()` in a `finally` block, and a **manual
runner at the bottom of each file** — run with `python <file>.py` (no pytest, no
PyCharm required).

## Files & coverage

| Script | What it does | Test cases | Needs a server? |
|--------|--------------|-----------|-----------------|
| `test_hazardwatch_api.py` | HTTP status codes, schemas, response time, mutating flows (safely reverted) | TC-03, 06–30, 33, 34, 37, 42–50, 54, 55, 57 | yes (running) |
| `test_hazardwatch_ui.py` | Selenium: Analyze, tabs, **live-feed flow** (load→stream→stop→revert), product monitor | TC-06/07, 18, 32, 36, 38, 39, 41 | yes (running) + Chrome |
| `test_hazardwatch_lifecycle.py` | Starts its **own** server; startup, loading→ready, sanity check | TC-01, 02, 03, 04, 05 | no (spawns one) |
| `test_hazardwatch_integrity.py` | White-box: forces store misalignment, asserts save aborts | TC-05, 31, 58 | no |
| `test_hazardwatch_load.py` | Concurrency **sweep** (5→10→25→50→100 clients, latency percentiles) + write-safety + soak **+ bulk volume/throughput** (thousands of reviews, timed, server mem/CPU) | TC-51, 52, 53, 59 | yes (running) |
| `test_hazardwatch_ai.py` | **AI robustness** (gibberish, emoji, URL, non-/mixed-language, adversarial, …) + behaviour + false-positive/false-negative rates | TC-60…72 | yes (running) |
| `test_hazardwatch_reliability.py` | **Failure injection**: breaks a store/model file, asserts the server fails *gracefully* (health→error, endpoints→500), then restores | TC-73…76 | no (spawns broken-store servers) |
| `run_all.py` | Runs the suites in order | — | — |

Together these automate essentially all of TC-01…TC-76. A verification run passed
**40 API · 7 UI · 5 lifecycle · 2 integrity · load (TC-51/52/53/59) · 13 AI · 4 reliability**.

## Setup
```bash
pip install -r requirements-test.txt        # selenium, webdriver-manager, requests, pandas, pyarrow
# Google Chrome must be installed; webdriver-manager downloads the matching driver.
```

## Run
Start the API first (for the api / load / ui suites):
```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# wait until GET /health reports "status":"ok"
```
Then:
```bash
python test_hazardwatch_api.py
python test_hazardwatch_load.py
python test_hazardwatch_ai.py            # AI robustness / behaviour / FP-FN
python test_hazardwatch_ui.py            # headed
HEADLESS=1 python test_hazardwatch_ui.py # headless

# These manage their own server — run them with :8000 FREE (stop the manual server first):
python test_hazardwatch_lifecycle.py     # startup / loading / ready
python test_hazardwatch_reliability.py   # failure injection (breaks + restores store files)
python test_hazardwatch_integrity.py     # no server needed (white-box)

python run_all.py                        # everything (start the server first for api/load/ui)
```
Each run prints a colour-coded **PASS**/**FAIL** line per test and a summary banner,
and writes full timestamped detail to `hazardwatch_*_tests.log`.

### Showing detail live (verbose)
By default the console stays clean (just PASS/FAIL + banner). To also stream the
detail — latencies, hazard probabilities, throughput, memory/CPU — to the console
**while the tests run**, set `HW_VERBOSE=1` (applies to the `api`, `load`, and `ai`
suites; the UI suite already shows its verdicts inline):
```bash
HW_VERBOSE=1 python test_hazardwatch_api.py
HW_VERBOSE=1 python test_hazardwatch_ai.py
```
```powershell
$env:HW_VERBOSE = "1"; ..\..\.venv\Scripts\python.exe test_hazardwatch_ai.py
```
The PASS/FAIL badges are never duplicated in verbose mode — only the extra detail
is added.

### Concurrency sweep (TC-51)
`test_hazardwatch_load.py` sweeps simultaneous-client levels (**5 → 10 → 25 → 50**
by default) against `/analyze`, firing a fixed number of requests at each level and
reporting **latency median / p95 / max** plus any errors. Every response must be
`200` with zero errors/timeouts; the point is to show how response time grows as
concurrent users increase (on this single-worker CPU server, inference is largely
serialized, so higher concurrency mainly queues → higher latency, not more errors).
```bash
HW_CONC_TIERS=5,10,25,50 HW_CONC_REQ=40 python test_hazardwatch_load.py
HW_SOAK_SECONDS=120 python test_hazardwatch_load.py   # longer TC-53 soak
```

### Bulk volume / throughput (TC-59)
`test_hazardwatch_load.py` includes a volume test that loads thousands of reviews
via `POST /ingest-batch` (500 per batch) and reports **total time, throughput
(reviews/sec), per-batch latency, and index growth**. It defaults to 1,000 reviews
so the normal run stays quick; set env vars to scale it up:
```bash
HW_VOLUME_TIERS=1000,5000,10000 python test_hazardwatch_load.py  # show the throughput curve
HW_MIN_TPS=50 python test_hazardwatch_load.py                     # also fail if throughput < 50/s
```
The data is a ~35% hazard / 65% benign mix (tune with `HW_HAZARD_FRAC`) so the
classify→embed→index path is actually exercised. It is in-memory only and
`/revert`s at the end, so the on-disk stores are left untouched.

## Data safety (important)
The mutating tests (ingest / batch / save / live feed) **restore your index**:
they either call `POST /revert` in a `finally` block or snapshot the `models/`
store files and copy them back. After a full run the index is left exactly as it
was (verified: 1344 → 1344). The white-box integrity test works on an in-memory
copy and never writes to disk.

## Notes on method (from the Automation folder)
- **Selenium** drives the browser UI; **requests** covers what the browser can't
  see (HTTP status codes, latency). The two layers complement each other.
- **Explicit waits** (`WebDriverWait` + `expected_conditions`) are used instead of
  `time.sleep`, so the dynamic UI is handled reliably.
- **Alerts:** the Revert button opens a JS `confirm()` dialog; the UI test accepts
  it with `driver.switch_to.alert.accept()` (per the "Waits, Alerts, and Tabs" lecture).
- **Headless** mode (`HEADLESS=1`) is supported for unattended runs.
- Stable **locator ids** used: `reviewText`, `analyzeBtn`, `ingestBtn`, `probnum`,
  `badgeText`, `tab-analyze`, `tab-livefeed`, `tab-dashboard`, `fileInput`,
  `feedStart`, `feedStop`, `feedSave`, `feedRevert`, `fProcessed`, `productRows`.

## What stays manual
A few cases are intentionally not fully automated and remain guided manual checks:
TC-35 (auto-save on clean shutdown — signal handling differs on Windows) and
TC-56 (413 very-large-payload behaviour). These are documented in the Test Case
document's **Test Automation** column.
