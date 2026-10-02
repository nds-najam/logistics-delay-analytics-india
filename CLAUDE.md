# SRD Logistics – Service Performance & Live Reporting Dashboard

Context file for Claude (desktop or mobile). Read this first; it records what the project is, what is built,
what is not, and the decisions made so far. Last updated 2 Oct 2026.

## What this is
A Streamlit dashboard for SRD Logistics Pvt. Ltd. built from their 3-page requirement document
(service % vs configurable targets, route / source / destination / via / business / customer performance,
journey-stage delay drill-down, LR search, period comparison, booking-to-main-hub tracking, live reports,
system-generated improvement suggestions). It extends an earlier shipment-level demo, kept as `legacy_app.py`.
Prepared by DataQ. All data is **synthetic** until SRD supplies real extracts.

## Run
- `streamlit run app.py` (SRD dashboard) | `python src/srd_generator.py` regenerates `data/srd_lr_data.csv`
- `pytest tests/test_srd.py` (SRD, 27 tests) and `pytest tests/test_pipeline.py` (legacy, 32 tests)
- On the Windows dev machine run Python from PowerShell with `venv`; Git Bash's Python hit an Application
  Control block on a scikit-learn DLL.

## Code map
- `app.py` – 11 pages: Overview, Service vs Target, Network Performance, Delay Analysis, LR & Customer,
  Period Reports, Booking to Main Hub, Live Operations, Improvement Suggestions, Delay Risk (ML), Configuration.
  Sidebar holds the global filters and the "Target days basis" (benchmark / fixed days / benchmark ± days).
- `src/srd_generator.py` – one row per LR with a timestamp at each stage, via/hub, reasons and remarks.
- `src/srd_analytics.py` – targets, `prepare()`, `summarize()`, stage delay attribution, periods, hub movement, suggestions.
- `src/srd_model.py` – delay-risk model (trained on delivered LRs, booking-time features only; AUC about 0.72).
- `src/srd_live.py` – live feed (simulated by default; `HttpSource` if `SRD_API_BASE_URL` is set) and live reports.
- `data/srd_targets.csv` (route target days and target %), `data/srd_rules.json` (created on save, git-ignored).
- `docs/SRD_Solution_Architecture.pdf` – customer-facing architecture, requirements checklist and Q&A.
- `docs/SRD_effort_estimate.xlsx` – internal effort estimate (not for the customer).

## Definitions used (confirm with SRD)
- Actual days = calendar days from booking date to delivery date (to snapshot date if undelivered).
- Delayed = actual days > target days; an open LR counts as delayed once it breaches target.
- Actual service % = on-track LRs / total LRs. Variation % = actual − target service %.
- Delay point = stage with the largest excess over that route's median stage time.
- "KT weight" is assumed to be total LR weight in tonnes; "via" is the intermediate hub.

## Status
Built and tested (synthetic data): everything in the code map above.
**Not built yet:** Excel/CSV Upload & Validate page; PostgreSQL storage (data is a CSV in the repo; targets are
CSV files); real API integration (SRD's API spec is unknown, the HTTP contract in `srd_live.py` is a guess);
login/SSO. Config edits are lost on Streamlit Community Cloud restarts because its disk is temporary.

## Decisions and answers given so far
- **Two options were estimated:** A) this Streamlit/Python app including a data layer: about 350 h base + 100 h
  buffer = 450 h expected (376–571). B) Power BI on a Pentaho → PostgreSQL DWH built by the client, 264 h base +
  100 h buffer = 364 h (304–472). Cost is 0 until a rate is entered in the spreadsheet (Assumptions!B7).
- **Recommended direction:** keep the finished Streamlit reports and ML, and feed them from PostgreSQL.
- **Partner's four questions** (answered in PDF page 4): (1) Excel via a new Upload & Validate page with column
  mapping, validation report and confirm-to-load; (2) stored in PostgreSQL (hosted by SRD, DataQ or a managed
  service); (3) loads are upserts by LR number, targets/rules persist in the DB, measures are recalculated by
  the Python engine, model retrained after loads, upload log for audit; (4) Streamlit reads the DB through a
  read-only account, users open a web link, filters and CSV export on every table, login to be agreed.
- **Hosting:** Streamlit Community Cloud for demo/pilot only; private cloud or on-premise for real LR data.
  Deploy steps: share.streamlit.io, repo `nds-najam/logistics-delay-analytics-india`, branch `main`, file `app.py`,
  Python 3.11. Deploying needs the owner's Streamlit login; Claude cannot do it.

## Needed from SRD (full list in the PDF)
Sample Excel/CSV extracts (3–12 months), master lists (branches, hubs, via, consignors, consignees),
benchmark days and target % per route, delay-reason/remark catalogue, definitions above, API documentation and
credentials, hosting and security choice, PostgreSQL instance, user list/login preference, named contacts.

## Conventions
Match the surrounding code style. Targets and rules must stay configurable (never hard-code them). Do not commit
`sdr_requirements.pdf` (client document, git-ignored). Commit messages end with the Co-Authored-By line given by the harness.
