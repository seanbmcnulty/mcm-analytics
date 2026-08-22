# MCM Analytics — project instructions

Read this before doing any work on this repo. It's written for a Claude
Code / Cowork session that has no memory of past sessions — it should be
enough context to pick up work safely on its own.

## What this is

A Deribit-only crypto derivatives analytics bot, functionally equivalent to
a prior "FalconX bot," rebuilt because FalconX/Amberdata data isn't
available. Streamlit app, deployed on Streamlit Community Cloud at
**https://mcm-analytics.streamlit.app**. GitHub repo:
**seanbmcnulty/mcm-analytics** (owner: Sean McNulty / seanbmcnulty@gmail.com).

Covers 4 assets: **BTC, ETH, SOL, HYPE** (`lib/constants.py:ASSET_CONFIG`).
BTC/ETH have a Deribit DVOL index; SOL/HYPE don't (`has_dvol: False`) — that
asymmetry is permanent and intentional, not a bug to "fix."

**One deliberate exception to "Deribit-only":** `pages/11_Fear_Greed_Signal.py`
pulls the Crypto Fear & Greed Index from alternative.me (`lib/fng.py`) since
Deribit has no sentiment index at all. Confirmed with the user before adding
it (2026-08-22) — see the Session log entry below and `lib/fng.py`'s module
docstring. Everything else on that page (price data) is still Deribit perp
OHLC via `lib/deribit.py`, same as every other page.

Local repo lives on the owner's machine (Windows) — reached in Cowork
sessions via the `mcp__remote-devices__*` device-bridge tools, folder name
`mcm-analytics` (nested: the connected root contains a `mcm-analytics`
subfolder, which is the actual repo — check with `device_list_dir` if unsure).

## Deployment model — the thing most likely to bite you

Streamlit Community Cloud auto-redeploys on every push to `main`, and its
**filesystem is ephemeral** — anything written to disk at runtime (e.g.
`data/snapshots/*.csv`) is wiped on the next redeploy. This drove most of
the architecture below. Don't assume local file writes from the running app
persist.

The app doesn't always restart cleanly after a push — if the user reports a
stale error right after pushing, check the actual committed source via
`WebFetch` on `raw.githubusercontent.com/seanbmcnulty/mcm-analytics/main/<path>`
(bypasses any caching) before assuming it's a code bug. If the code is
correct but the error persists, it's usually a stuck Streamlit Cloud worker
— tell the user to use Manage app → ⋮ → Reboot app (this can't be done from
a Cowork session; Chrome automation here isn't authenticated as the app
owner).

## Historical data: why it's reconstructed, not queried

Deribit's public API serves no option-IV history. `lib/history.py`
reconstructs it via a three-tier preference chain (see the module docstring
for full detail):

1. **Recorded snapshots** — real history, from `data/snapshots/{asset}_surface.csv`,
   merged from two sources: the local (ephemeral) file, and a **durable**
   copy fetched over HTTPS from a dedicated `snapshots` git branch (see
   next section). `estimated=False`.
2. **DVOL-anchored re-levelling** — BTC/ETH only. Current surface shape
   scaled by `DVOL(t)/DVOL(now)`. `estimated=True`.
3. **Realized-vol-proxy re-levelling** — SOL/HYPE (no DVOL). Same idea,
   driven by a Parkinson RV ratio from perp OHLC candles. `estimated=True`.

Lookback labels ("24h ago", "1 week ago") are **adaptive/honest**: if the
recorded history doesn't actually reach back that far, the label reflects
the real elapsed time instead of silently showing the nearest available
point under a wrong label. See `history.surface_row_near()` and
`cmd_vol._lookback_points()`/`_fmt_ago()`.

## Persistent snapshot storage (GitHub Actions)

`data/snapshots/` is gitignored on `main`. A separate pipeline keeps it
durable without touching `main` (which would trigger a Streamlit Cloud
redeploy on every commit):

- `scripts/record_snapshot.py` — standalone CLI (no streamlit import),
  appends one row per asset via `lib.history.record_snapshot()`.
- `.github/workflows/record_snapshots.yml` — runs it hourly via cron, plus
  `workflow_dispatch` for manual runs. Commits to a **`snapshots`** branch
  (never `main`) using `git push --force`, since it's a data mirror, not a
  code history. Distinguishes "branch doesn't exist yet" from "couldn't
  reach it" via `git ls-remote`'s exit code — treating a network blip as
  "start fresh" would force-push over and destroy all prior history, so a
  real failure aborts the job instead.
- `lib/history.py:load_snapshots()` fetches that branch's CSV over HTTPS
  (`raw.githubusercontent.com/.../snapshots/data/snapshots/...`), cached
  5 min in-process, and merges it with whatever the local ephemeral file
  has (local wins on overlap — it's always at least as fresh).
- `lib/history.py:record_snapshot()` also runs `_thin_snapshot_file()`
  after every write: full hourly resolution for the last 14 days, one
  row/day beyond that. Without this, hourly recording grows the CSV
  forever. **If you touch this again, be careful about timestamp string
  format** — the file mixes rows written by `datetime.isoformat()` (fresh
  appends) and rows rewritten by thinning; both must produce the exact
  same string format, or pandas' fast CSV datetime parser silently drops
  rows as unparseable on a later read. This actually happened once — verify
  round-trip integrity (`rows written == rows read back`) before shipping
  any change here, not just "does it run."
- Repo one-time setup required: Settings → Actions → General → Workflow
  permissions → "Read and write permissions" (not default-on).

## Caching layers (all in-process, per-asset-keyed, safely bounded)

- `lib/deribit.py:_cache` — raw API responses, self-prunes at 500 entries.
  Process-wide (one Streamlit Community Cloud process serves every page and
  user), so this is already a shared cache across the whole app, not just
  the page that happens to call it.
- `lib/surface.py:_VOLS_CACHE` — 10s TTL, avoids redundant full option-chain
  + Black-Scholes-delta-search recomputation (was up to 113 redundant calls
  per asset per page load before this was added).
- `lib/history.py:_DRIVER_CACHE` — 300s TTL, the DVOL/RV re-levelling series.
- `lib/history.py:_REMOTE_SNAPSHOT_CACHE` — 300s TTL, the GitHub-branch fetch.
- Each page also layers its own `@st.cache_data(ttl=...)` around its
  fetch functions. **2026-08-22: TTLs are now named tiers in
  `lib/constants.py`** (`TTL_FAST=10` … `TTL_DAILY_EXTERNAL=3600`, see that
  file) instead of ad hoc bare numbers, so the same kind of data gets the
  same freshness window regardless of which page fetches it. Use one of
  these for any new `ttl=` rather than inventing a new number, unless you
  have a specific reason to deviate (say why in a comment if so).
- **`lib/cache.py`** (added 2026-08-22) is now the single place that knows
  about all of the above: `clear_all_caches()` clears `st.cache_data` +
  all three lib-level caches in one call, and `render_refresh_button()`
  renders a standard "🔄 Refresh data (clear cache)" button wired to it.
  If you add a new cache anywhere, wire its clear function into
  `clear_all_caches()` — don't hand-roll a page-local clear list again.
  All 7 analytics pages now have a refresh button through this helper
  (previously only `01_MCM_Bot.py` and `06_Time_Based_Realized_Vol.py`
  did, and `06`'s only cleared `deribit`'s cache, not `surface`/`history` —
  harmless only because that page doesn't read through those two).

## Working conventions for this session type

**This is a device-bridge workflow, not a direct-push workflow.** The
session's cloud container and the owner's local repo are separate
filesystems.

- To **read** a file from the local repo: `device_stage_files`, then `Read`
  from `/mnt/user-data/uploads/...`.
- To **write** a file back: edit it in the cloud workspace, `SendUserFile`,
  then `device_commit_files` to the exact Windows path
  (`C:\Users\seanb\OneDrive\Documents\Python\mcm-analytics\mcm-analytics\...`).
- **`.github/workflows/*.yml` cannot be written via `device_commit_files`**
  — it's a protected path on the device bridge. Work around it: commit to
  a staging path instead (e.g. `_workflow_staging/record_snapshots.yml`),
  then have the user run `add_workflow.bat` (already in the repo root) —
  it moves the staged file into `.github\workflows\` and runs
  `push_to_github.bat` for you. Re-use this pattern for any future
  `.github/workflows/*` change.
- **Never push to GitHub yourself.** The owner always runs
  `push_to_github.bat` (handles stale lockfiles, remote setup, rebase,
  push) themselves after you've committed files to their disk.
- Don't create empty/redundant `.md` files or READMEs unless asked.

## Testing — do this before shipping any change

There's no live Deribit access from the cloud sandbox, so testing is
offline-only, against `tests/fake_deribit.py` (monkeypatches
`lib.deribit`'s public functions with a synthetic feed):

```
PYTHONPATH=/tmp/stubs:<repo> python3 tests/test_math.py           # 17 numeric assertions
PYTHONPATH=/tmp/stubs:<repo> python3 tests/run_all.py BTC ETH SOL HYPE   # all 21 commands x 4 assets
```

`/tmp/stubs` needs minimal `plotly` and `streamlit` stub packages (pandas/
numpy/scipy are real). Recreate them if missing — a `streamlit` stub whose
`__getattr__` returns a no-op is enough for `lib/*.py` changes, since those
modules don't touch `st` directly. The `plotly.graph_objects` stub needs at
least `Scatter`/`Heatmap`/`Bar`/`Waterfall`/`Box` as no-op trace stand-ins —
`tests/run_all.py` exercises `forward_vol_steepness*` and `atm_iv_box_plot`,
which construct `go.Waterfall`/`go.Box` specifically.

**`pages/*.py` files are a different story** — they call `st.tabs`,
`st.columns`, `st.popover`, `st.session_state.x = y` (attribute-style),
etc., which a bare no-op stub can't handle (session_state needs attribute
access, `st.columns`/`st.tabs` need to return indexable context managers).
There is no live-Streamlit test path available in this sandbox. Before
shipping a `pages/*.py` change, build (or reuse) a richer stub that at
least: returns list-of-context-managers from `columns`/`tabs`, supports
attribute access on `session_state`, and — importantly — **tracks widget
`key=` uniqueness and raises on a duplicate**, since Streamlit's real
`StreamlitDuplicateElementKey` error is exactly the kind of bug an
under-powered stub won't catch (this happened once — a leftover duplicate
button survived because the first version of the stub didn't check for
it). Run the page through `runpy.run_path` under that stub for at least:
a cold load, each button-click path, and a run with real cached results
populated (drive `cmdreg.run_command` through the `fake_deribit` patch so
`render_output` actually executes on real data, not just no-ops).

There's no visual/browser verification available in this sandbox (no
authenticated Chrome session) — UI layout changes are verified by control-
flow testing plus careful manual read-through, not by looking at the
rendered page. Say so explicitly when shipping a layout change, and ask the
user to sanity-check the live result.

## Code map

```
Home.py                        landing page
pages/01_MCM_Bot.py             the main bot: toolbar + per-asset tabs + single-command tab
pages/02_Block_Trades.py        block trade flow — NOTE: calls Deribit directly via
                                 requests.get(), bypassing lib/deribit.py's shared
                                 cache/rate-limiter. Has its own st.cache_data + pacing,
                                 not reckless, but not coordinated with the rest of the
                                 app either. Worth unifying eventually, not urgent.
pages/06_Time_Based_Realized_Vol.py  RV across hedging frequencies x lookbacks (BTC/ETH perps),
                                 7 estimators + decision matrix; renamed 2026-08-22 from
                                 06_Realized_Vol.py, ported from exodus-analytics
pages/07,08,10_*.py             secondary analytics pages (regime, correlation, macro)
pages/11_Fear_Greed_Signal.py   contrarian delta-lean backtest vs alternative.me F&G Index
                                 (the one non-Deribit data source in this app — see above)
pages/03,04,05,09,12,13,14      retired — moved to _to_delete/ on the device, not in git

lib/deribit.py                  Deribit public API client, caching, rate limiting
lib/surface.py                  option chain parsing, BS delta search, vol surface math
lib/history.py                  historical IV reconstruction (see above), snapshot I/O
lib/cache.py                    clear_all_caches() + render_refresh_button() — the one
                                 place that clears every cache layer app-wide (added 2026-08-22)
lib/cmd_vol.py                  vol-related chart/table commands (term structure, skew,
                                 smile, forward vols, ATM box plot, etc.)
lib/cmd_market.py               market/basis/funding/block-trade commands
lib/commands.py                 command registry (COMMAND_NAMES, run_command dispatch)
lib/constants.py                ASSET_CONFIG, ASSETS, color scheme
lib/fx_style.py                 plotly theming, SGT timezone helpers (local_now, to_local),
                                 watermark, chart title/legend/margin layout (finalize())
lib/vol_math.py                 Black-Scholes, implied vol, delta math
lib/fng.py                      alternative.me Crypto Fear & Greed Index client (pure
                                 functions, no streamlit import — same shape as deribit.py)
lib/instruments.py, telegram.py instrument parsing; Telegram send integration

scripts/record_snapshot.py      standalone CLI recorder, run by the GH Action
.github/workflows/record_snapshots.yml   hourly snapshot recorder (see above)

tests/fake_deribit.py           synthetic Deribit feed for offline testing
tests/test_math.py              numeric assertions
tests/run_all.py                exercises every command x asset combo

push_to_github.bat              user-run: commit + rebase + push to main
add_workflow.bat                user-run: move staged .github/workflows file into place, then push
```

## Style conventions established in this codebase

- `width="stretch"` on Streamlit elements, not the deprecated
  `use_container_width=True` — use the `_stretch()` helper defined in
  `pages/01_MCM_Bot.py` (`dict(width="stretch")`) where present.
- All display timestamps go through `fx_style.local_now()` /
  `fx_style.to_local()` / `fx_style.to_local_ts()` — Asia/Singapore is the
  display timezone (`fx_style.DISPLAY_TZ`). Never use bare
  `datetime.now(timezone.utc)` for anything user-facing.
- Charts go through `fx_style.apply_theme(fig)` and, where a title/legend/
  note is involved, `fx_style.finalize(fig, note=..., legend_rows=...)` —
  don't set `fig.update_layout(template=...)` directly, it bypasses the
  shared theming and watermark.
- Per-asset in-process caching pattern: `_CACHE: dict[str, tuple[float, Any]]`
  keyed by asset (or `(asset, days)` etc.), TTL-checked on read. Follow this
  shape for any new cache — it's what `clear_cache()` functions expect.
- Best-effort, silent failure is the norm for recording/caching code paths
  (`record_snapshot`, `_thin_snapshot_file`, `_fetch_remote_snapshots` all
  swallow exceptions and return `None`/`False`) — this must never be the
  reason the live app breaks. Don't let a new failure mode there raise.

## Known backlog (not urgent, flagged during review)

- `pages/02_Block_Trades_-_Deribit.py`'s direct Deribit calls could still be
  routed through `lib/deribit.py` for a single shared rate budget — it got
  a refresh button and named TTLs on 2026-08-22 (see Session log), but its
  fetch layer itself is still the page-local `requests.get`/
  `_get_json_with_retry`, not `lib/deribit.py`'s shared cache.
- A few `frame.iterrows()` loops in `lib/cmd_vol.py` (smile snapshot
  high/low band) are a minor pandas anti-pattern — low-risk to vectorize,
  but not currently a real bottleneck since history frames are now
  size-bounded by the thinning logic above.

## Session log

Keep this updated: when a session makes a non-trivial change, decision, or
finds a bug worth remembering, add a dated entry below before the session
ends. Newest entry on top. This is how continuity works across sessions —
nothing here persists otherwise.

### 2026-08-22 — Time Based Realized Vol: dropped the 4h frequency (Deribit returns no candles at that resolution)

User tested the live page and saw every lookback window (1d through 30d)
report `4h: No candles returned for BTC-PERPETUAL (4h)` — not a transient
outage, every window hit it. `INTERVAL_TO_RESOLUTION` mapped `"4h" -> "240"`
with a comment claiming 240 "isn't documented but works in practice" — that
claim was never actually verified against live Deribit (this sandbox has no
live network to it), and the user's real test disproved it. Removed 4h
entirely rather than trying to work around it, since 2h and 12h already
bracket that gap in the frequency ladder: dropped from `INTERVAL_OPTIONS`,
`INTERVAL_TO_RESOLUTION`, `interval_to_ms`'s duration map, and the
`fig_rolling_multi` marker-symbol map (`pages/06_Time_Based_Realized_Vol.py`).
9 hedging frequencies remain: 1m, 5m, 10m, 15m, 30m, 1h, 2h, 12h, 1d.

Re-verified offline (same mock-streamlit/mock-plotly/fake-Deribit harness as
the entries below): smoke test now shows 67 fetch calls (was 68) across 9
intervals, no exceptions, no `st.error`/`st.warning`; re-ran the cache/
resilience checks (cold fetch, warm-TTL reuse, hard refresh, outage
fallback, multi-interval batch) against the smaller interval set — all
still pass. Not verified: real Deribit connectivity (same constraint as
always) — ask the user to confirm the "excluded frequencies: 4h" warnings
are gone on the next live load.

### 2026-08-22 — Cache consolidation + refresh button on every page

User asked two related questions: can data be cached efficiently across all
the pages, and can every page get a button to pull the latest data. Read
through `lib/deribit.py`, `lib/surface.py`, `lib/history.py`, and all 7
analytics pages first to answer both from the actual code rather than
guessing.

Findings: `lib/deribit.py`'s `_cache` is already a single process-wide
cache shared across every page and user (Streamlit Community Cloud runs
one process for the whole multi-page app) — that part was already
efficient. What wasn't: each page's own `@st.cache_data(ttl=...)` wrapper
picked its own TTL number independently (30/60/120/300/600/3600 all
appeared, sometimes for conceptually the same kind of data), and only 2 of
7 analytics pages (`01_MCM_Bot.py`, `06_Time_Based_Realized_Vol.py`) had
any refresh control at all — the other 5 just waited out whatever TTL
their fetchers used (up to an hour for the Fear & Greed page's daily
external feed). Also, `06`'s existing "Hard refresh" only cleared
`st.cache_data` + `lib.deribit`'s cache, not `lib.surface`/`lib.history` —
harmless today only because that page doesn't read through those two, but
that was incidental, not by design.

Fix, in three parts:
1. `lib/constants.py`: added eight named TTL tiers (`TTL_FAST=10` through
   `TTL_DAILY_EXTERNAL=3600`) so the same freshness cadence gets the same
   name everywhere. Every existing `ttl=<number>` in `lib/deribit.py` and
   the 5 previously-unbuttoned pages was swapped to the matching constant
   — verified this was a pure rename with zero value changes (printed the
   constants and diffed against the original hardcoded numbers).
2. New `lib/cache.py`: `clear_all_caches()` clears all four cache layers
   (`st.cache_data` + `deribit`/`surface`/`history`) in one call, and
   `render_refresh_button()` renders a standard "🔄 Refresh data (clear
   cache)" button wired to it. `01_MCM_Bot.py`'s "Clear data cache" and
   `06`'s "Hard refresh" buttons now both call through this instead of
   hand-rolling which caches to clear (`06` picks up the `surface`/
   `history` clears it was missing, for free).
3. Added `render_refresh_button()` to the 5 pages that had nothing:
   `02_Block_Trades_-_Deribit.py`, `07_Regime_Identifier.py`,
   `08_Spot_Vol_Correlation.py`, `10_Macro_Event_Impact.py`,
   `11_Fear_Greed_Signal.py` — each in the sidebar alongside that page's
   other controls.

Did not change: `pages/02`'s fetch layer is still its own standalone
`requests.get`/`_get_json_with_retry`, not routed through `lib/deribit.py`
(pre-existing, flagged in Known backlog above) — its refresh button clears
its `st.cache_data` wrappers (which is everything it uses) via
`clear_all_caches()`, but the underlying HTTP calls aren't deduplicated
against the rest of the app the way `lib/deribit.py`-backed pages are.

Verified offline (no live Deribit/PyPI access in this sandbox): `py_compile`
on every touched file; a real (non-stubbed) import of `lib/constants.py`
and `lib/deribit.py` confirming the TTL rename introduced no typos and no
value changes; a purpose-built `streamlit`+`plotly` stub (see Testing
section) used to `runpy.run_path` all 7 modified pages end-to-end — all
either completed or reached an expected `st.stop()`, with the new
`cache_lib.render_refresh_button()`/`clear_all_caches()` call sites
exercised on both the not-clicked and (button forced `True`) clicked path;
and the existing `tests/run_all.py` suite (all 21 commands x BTC/SOL
against `fake_deribit`) still passes unchanged against the edited
`lib/deribit.py`. No live/visual check was possible — user should
sanity-check the deployed pages, in particular that the new sidebar button
placement doesn't crowd existing controls.

### 2026-08-22 — Replaced "Load selected…" popover with a per-tab "Run <asset>" button

User found the "Load selected…" popover (pick assets + commands from two
multiselects, then click through) cumbersome for the common case of "I'm
already looking at BTC's tab, just run BTC." Removed the popover entirely
and added a small "Run {asset}" button at the top of each asset tab in
`pages/01_MCM_Bot.py` — one click runs all commands for just that asset
and replaces that asset's cached data, no menu needed. "Load all" and
"Refresh all" are unchanged.

Also removed the `visible_assets` filter that used to hide an asset's tab
entirely until it had at least one cached result (a holdover from
"Load selected" letting you load a subset of assets — since every tab now
has its own way to populate itself, hiding it first made no sense). All 4
asset tabs are now always visible, each showing its own "Not loaded, click
Run {asset}" hint per command when empty.

Toolbar went from 4 columns (expiry, Load all, Refresh all, Load
selected…) to 3 (expiry, Load all, Refresh all).

**Testing gap found and fixed along the way:** no prior offline test had
ever driven `run_reports()` through a full page run — every earlier test
either pre-populated `mcm_all_results` directly via `cmdreg.run_command`
(bypassing `run_reports`/`st.progress` entirely) or triggered a button
whose data was already fully cached, so the "missing → run it" branch was
never hit. The new "Run BTC" test is the first to actually exercise it,
and it immediately hit a stub gap: `/tmp/stubs/streamlit.py` had no
`st.progress()`, so `run_reports()`'s `bar = st.progress(0.0)` fell
through to the generic no-op `__getattr__` (returns `None`), and the next
line, `bar.progress(...)`, crashed with `AttributeError: 'NoneType' object
has no attribute 'progress'`. Added a minimal `_Progress` class (`progress()`
and `empty()` as no-ops) to the stub. Also needed `st._seen_keys.clear()`
between two `runpy.run_path()` calls in the same test process — the
stub's duplicate-widget-key tracking is module-global and doesn't reset
between independent script runs on its own.

Verified offline (new `tests/_smoke_run_asset_button.py`): (1) with an
empty results cache, all 4 asset tabs still render with no duplicate-key
errors (confirms dropping `visible_assets` is safe); (2) simulating a
click on "Run BTC" with an empty cache loads exactly BTC's full command
set into `mcm_all_results` and nothing for ETH/SOL/HYPE, then hits
`st.rerun()` as expected (caught via the stub's intentional
`RuntimeError`).

### 2026-08-22 — Time Based Realized Vol: fixed slow/stuck cold load (sequential Deribit fetch -> shared thread pool)

User reported the deployed Time Based Realized Vol page (previous entry
below) "loaded fine just taking a long time," then clarified it actually
**never finished** — the spinner ran indefinitely and no data was ever
presented. Root cause: a cold load with the default "all 10 frequencies"
selection needs ~68 paginated Deribit chunk requests (1m/30d alone is ~44
chunks at 1000 bars/call), fetched **one at a time** in the original port —
and `lib/deribit.py`'s own retry logic (`_request`: 3 attempts, 15s timeout
each, backoff up to `2**attempt`s) means a single slow/failing chunk can
alone take up to ~45-50s before giving up. Stacked sequentially across 68
chunks, a handful of slow chunks (not even outright failures) is enough to
turn "slow" into "looks hung."

**Fix:** replaced the single-interval-at-a-time pagination
(`fetch_deribit_klines_paginated` + a per-interval `load_or_fetch_klines`
loop in the main script) with a batch fetch layer that dispatches **every**
chunk request across **every** selected hedging frequency into one shared
`concurrent.futures.ThreadPoolExecutor` (`max_workers=16`):
`_build_chunk_plan` (time-range chunk boundaries), `_fetch_chunks_parallel`
(one shared pool, `as_completed`), `_assemble_klines` (concat + dedupe,
same pandas datetime64 fix as before), `load_klines_for_intervals` (new
multi-interval entry point — preserves the existing per-interval UI-cache/
snapshot-fallback semantics exactly), with `load_or_fetch_klines` kept as a
thin single-interval wrapper for `maybe_recompute_suspicious_interval`'s
call site. The main script now calls `load_klines_for_intervals` once for
all `compare_intervals` instead of looping. Deliberately one shared pool
rather than one pool per interval, to avoid a nested-executor deadlock risk.

Also vectorized the `pa_var` pre-averaging loop in `compute_advanced_
estimators` (was a per-bar Python `for` loop building a list; now a single
`numpy.lib.stride_tricks.sliding_window_view` + matrix-vector product) — a
secondary, much smaller win, but free correctness-preserving cleanup found
while touching this section.

**Verified offline** (same sandbox constraint — no live Deribit network
here): re-ran the existing mock-streamlit/mock-plotly/fake-Deribit smoke
test end-to-end against the refactored file — identical output (68 fetch
calls, no exceptions, no `st.error`/`st.warning`, same RV numbers given the
same synthetic seed). Confirmed `pa_var` vectorized vs. original loop are
numerically identical (max diff ~1e-9 across n=1..2000, including the n<m
edge cases) with a ~35x speedup at n=5000. Directly re-verified all cache/
resilience semantics against the new `load_klines_for_intervals` (cold
fetch, warm within-TTL reuse with zero network calls, hard-refresh bypass,
forced-refresh-during-outage has no fallback by design, organic outage
after TTL expiry correctly falls back to last-good snapshot, multi-interval
batch call) — all matched the semantics verified against the old function.

**Measured the actual fix, not just the model:** built a chunk-plan-exact
timing harness (`_fetch_chunks_parallel` vs. a fully sequential loop over
the identical 68-chunk plan) with the real `lib/deribit.py` rate limiter
active. At the rate limiter's own floor (100ms/call, 10 req/s) old and new
are equal (~6.9s) — the limiter is a hard floor no amount of concurrency
beats. The real win shows up once per-call latency exceeds that floor: at a
more realistic ~300ms/call, sequential = ~20.5s, parallel = ~7.1s (**~2.9x**),
and the mechanism that fixes the "stuck" symptom specifically is that a
single chunk's worst-case ~45-50s retry-with-backoff no longer blocks the
other 67 chunks behind it — it just makes that one thread slow, not the
whole page.

**Not verified:** real Deribit connectivity/latency (same sandbox
constraint as always) and the actual rendered page in a browser — ask the
user to re-test the live cold load after this ships and confirm it now
completes.

### 2026-08-22 — Renamed + rewrote Realized Vol -> Time Based Realized Vol (ported from exodus-analytics)

User asked for `pages/06_Realized_Vol.py` to be rebuilt on the model of
exodus-analytics's `analytics_frontend/streamlit/pages/Time_Based_Realized_Vol.py`
(found via the `exodus-analytics-backup-2026-08-06` backup nested inside the
connected `mcm-analytics` folder), renamed to match, using perp instead of
spot (this app has no spot leg anyway), scoped to BTC/ETH perps for now.

**Renamed** `pages/06_Realized_Vol.py` -> `pages/06_Time_Based_Realized_Vol.py`
(kept the `06` ordinal slot, same pattern as the `02_Block_Trades_-_Deribit.py`
rename on 2026-08-21). Old file moved to `_to_delete/pages/` (not deleted —
device bridge can't unlink mounted files, same constraint as always). Updated
`Home.py`'s page directory entry to match.

**What changed vs. the old page:** the old page was a simple multi-asset RV
matrix (5 estimators x fixed day-count windows, BTC/ETH/SOL/HYPE, one
timeframe at a time). The new page answers a different, more specific
question per exodus's design — "if I hedge every X minutes/hours, what
realized vol do I actually experience" — for one asset at a time: 10 hedging
frequencies (1m-1d) x 6 lookbacks (1d-30d) x 7 estimators (close-to-close,
Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang, Bipower variation, a
Realized-Kernel/pre-averaging proxy blend) averaged into a Composite RV used
for long/short-gamma ranking, plus an EWMA next-window forecast, data
completeness tracking, gap-fill + robust-outlier controls (winsorize/drop via
median+MAD z-score), a cross-lookback heatmap, an interactive 3D decision
matrix (lookback x frequency x metric), and a full Telegram report. All the
math/quality-control functions are direct, exchange-agnostic ports from
exodus (they operate on OHLC arrays, not on Binance specifics).

**Adapted from exodus, not ported verbatim:**
- Data source: exodus fetched Binance spot/perp klines directly; this page
  fetches Deribit tradingview OHLC via `lib/deribit.get_tradingview_ohlc`
  (BTC-PERPETUAL/ETH-PERPETUAL from `ASSET_CONFIG`), paginated in
  1000-candle chunks (`fetch_deribit_klines_paginated`) since a 1m/30d
  request is 43,200 candles and Deribit's tradingview endpoint has no
  documented per-call cap.
- No spot/perp toggle — Deribit's public API has no spot market to compare
  against, so this app doesn't have that axis at all; the "in-progress live
  bar" checkbox for 12h/1d frequencies is kept (still relevant on perp) but
  the "market mode" selector is dropped.
- Assets: BTC/ETH only for now (`TBRV_ASSETS`), not the ~28-asset Binance
  list — could extend to SOL/HYPE once their Deribit tradingview history
  depth at 1m/5m is checked.
- Restyled to this app's conventions: `width="stretch"` (not
  `use_container_width`), `fx_style.apply_theme`/`add_watermark` on every
  chart, timestamps through `fx_style.to_local()` (SGT display), `lib/
  telegram.py`'s `send_message`/`send_photo`/`is_configured()` gate (button
  is disabled with a caption when not configured, rather than exodus's
  always-on button), `ASSET_COLORS`/`PLOTLY_LAYOUT` from `lib/constants.py`.
- Dropped exodus's cross-page Telegram batch sequence (`st.session_state
  ["telegram_run_sequence"]`, `page_paths.py`, `st.switch_page` chaining) —
  that's infrastructure from exodus's `Home.py` that this app doesn't have;
  kept the single-page "Send to Telegram" button instead, which covers the
  same reporting functionality for this page on its own.
- Dropped a few functions exodus defined but never actually called from its
  own main UI (`fig_price`, `fig_rolling_vol`, `fig_price_vol_dual`,
  `fig_return_hist`, `fig_regime_bands`, `build_rolling_regime_bands`) —
  verified by reading exodus's full ~3400-line source, not assumed.

**Real bug found + fixed during offline verification (see CLAUDE.md's own
"Testing" section for why offline verification matters here — no live
network to Deribit from this sandbox either):** the pandas installed in this
sandbox (3.0.x) resolves `pd.to_datetime(..., unit="ms")` to
`datetime64[ms]`, not the historically-default `datetime64[ns]` — so
`.astype("int64") // 1_000_000` (the natural way to get milliseconds back
from a nanosecond-resolution datetime column) silently returned timestamps
1,000,000x too small on this pandas version. Fixed by forcing
`.to_numpy().astype("datetime64[ms]").astype("int64")` before extracting the
epoch integer, which is correct regardless of which resolution pandas
defaults to. This would have broken every fetch in production if that
pandas version (or newer) ever got resolved by `pandas>=2.1.0` in
`requirements.txt` — worth knowing about for any other page doing the same
datetime64->int64 round-trip.

**Verified offline** using the stub approach this file's own "Testing"
section prescribes (no live Deribit network from this sandbox): built a
richer `streamlit` + `plotly.graph_objects` stub (columns/tabs as reusable
context managers, attribute-access `session_state`, permissive Figure/trace
stand-ins since `plotly` itself isn't installable in this sandbox either),
patched `lib.deribit.get_tradingview_ohlc` with a deterministic synthetic
random-walk feed, and ran the real page script end-to-end via
`importlib.util.exec_module` for: a cold load (all 10 frequencies x 6
lookbacks, BTC), the "Send to Telegram" button path (confirmed it degrades
gracefully — no credentials in this sandbox, so sends report `False` without
raising), and the "Hard refresh" button path (confirmed it clears cache and
calls `st.rerun()` as expected). Also directly exercised the resilience
fallback: a successful fetch followed by a simulated Deribit outage on the
next organic (non-forced) fetch correctly returns the last-good snapshot
with `from_cache_fallback=True`; a *forced* refresh that then fails does
NOT fall back (it deletes the snapshot before retrying) — this matches
exodus's original semantics exactly (a forced refresh means "discard stale
data," so no fallback is used), not a bug. Checked for duplicate widget
keys (`grep`'d every `st.<widget>("label"...)` call) since this file's
Testing section flags that as a real recurring bug class — none found, all
labels unique. All 10 lookback x frequency combinations returned 100%
candle completeness against the synthetic feed, RV decreased monotonically
from 1m to 1d as expected for the synthetic random walk, and long/short-
gamma ranks were assigned correctly.

**Not verified:** real Deribit connectivity (same sandbox constraint as
every other page), and the actual rendered Streamlit page in a browser —
say so to the user, and suggest a visual sanity-check on the live deploy
after this ships.

### 2026-08-22 — New page: Fear & Greed Signal (contrarian delta-lean backtest)

User asked for a new page answering: at what Crypto Fear & Greed Index
levels has leaning delta long (fear) or short (greed) historically paid
off, with a probability/confidence measure to gauge whether to follow the
signal. Confirmed three design decisions with the user before building
(all went with the recommended option): (1) OK to add alternative.me's
Crypto Fear & Greed Index as a new external, non-Deribit data source —
Deribit has nothing like it; (2) backtest against the perpetual (what
delta lean is actually traded on), not spot/index; (3) let the page run
against all 4 assets (F&G is one market-wide series, applied to each
asset's own price), not BTC-only.

**New files:** `lib/fng.py` (alternative.me client — pure functions, no
streamlit import, same shape/testability as `lib/deribit.py`: retry/backoff,
no in-process cache of its own since the page layer owns caching via
`st.cache_data`) and `pages/11_Fear_Greed_Signal.py`.

**What the page does:** merges daily F&G value against the asset's daily
perp close, classifies each day into Extreme Fear/Fear/Neutral/Greed/
Extreme Greed (thresholds adjustable via sidebar sliders, default matches
alternative.me's own 25/45/55/75), then for each bucket × forward-return
horizon computes: n, mean/median forward return, win rate (probability the
contrarian call — long on fear, short on greed — was the right direction),
a Wilson-score 95% CI on that win rate, and a two-sided binomial-test
p-value vs. a coin flip. Also: a threshold-free decile view (same idea
without picking bucket edges), a value-vs-forward-return scatter with
OLS regression (slope/r/R²/p), and a daily-rebalanced equity curve
(bucket signal × next-day return) vs. buy-and-hold. Current-signal panel
up top surfaces today's F&G reading, its bucket, a suggested delta lean
(±1 for Extreme, ±0.5 for Fear/Greed, scaled to a sidebar-set max %), and
the historical win rate/CI/p-value for that exact bucket at the
user-chosen "headline" horizon.

**Deliberately flagged, not modeled:** funding, fees, slippage, leverage
— this isolates the directional signal, it isn't a tradable strategy
backtest. Also flagged: forward-return windows overlap (a 7d window
starting today shares 6 days with tomorrow's), so the significance stats
are indicative, not textbook independent-sample rigorous. Both caveats are
in the page's docstring and footer caption, not just here — didn't want
this framed as investment advice; kept language factual/descriptive per
usual (win rate, CI, p-value) rather than issuing calls to action.

**Verified offline** (no live network to api.alternative.me from this
sandbox either — same constraint as Deribit, see Testing section):
1. `lib/fng.py` unit-tested against a payload shaped exactly like
   alternative.me's documented response (values/timestamps as strings,
   per the real API): happy-path parsing + ascending sort, duplicate-date
   collapse, HTTP 500 → `None` not an exception, malformed body (missing
   `data`) → `None`, a 429 backing off and succeeding on retry, and
   `classify()`'s bucket edges including the `None`/`NaN` → `"Unknown"`
   path.
2. Built a `/tmp/stubs/streamlit.py` + `/tmp/stubs/plotly/graph_objects.py`
   pair (plotly wasn't installable in this sandbox either — no PyPI
   egress — so it's a stub, not the real package; pandas/numpy/scipy are
   real) rich enough to run `pages/*.py` end-to-end: sidebar/columns/
   expander act as both context managers and no-op method targets (real
   Streamlit lets you call `col.metric(...)` directly on a column object,
   which the first stub draft missed and had to fix), widgets return their
   coded default so the page exercises its actual cold-load path, and
   `st.cache_data` is a no-op passthrough.
3. **The real test, not just "doesn't crash":** generated 1500 days of
   synthetic data with a *known* injected effect — an autocorrelated F&G
   oscillator plus daily returns with a genuine (small, noisy) negative
   loading on `(F&G − 50)` — and ran the full page via `runpy` for all 4
   assets. Confirmed the statistics pipeline actually recovers the
   injected relationship: regression slope negative and p≈0 over n≈1493,
   Extreme Fear and Extreme Greed win rates both >50% with real sample
   sizes, Wilson CI bounds sane (contain the point estimate, within
   [0,1]), and the equity curve is finite and behaves sensibly (in this
   synthetic world where price drifts down with a real contrarian tilt
   baked in, the signal strategy beats buy-and-hold, as it should). Also
   directly tested `classify()`'s edge behavior (boundary values fall into
   the expected bucket, e.g. exactly 25 is `"Fear"` not `"Extreme Fear"`)
   and `wilson_ci()`'s degenerate `n=0` case.

No live Streamlit/browser check possible from this sandbox (same
limitation as every other page here) — ask the user to sanity-check the
live page after deploying, particularly the chart layout (dual y-axis
price/F&G overlay, bar chart error bars) which can't be visually verified
offline.

Also added the page to `Home.py`'s page grid and reused `pages/11` for
this — it had been vacated by an earlier retired page per the Code map
below (not in git, so no actual collision).

### 2026-08-21 — Added a "BTC+ETH" combined Telegram send button

Small follow-up to the per-asset send buttons. User asked for a button that
sends just BTC and ETH's reports (not all 4 assets, not one at a time).
Added a `BTC+ETH` button in `pages/01_MCM_Bot.py`'s toolbar, between the
per-asset buttons and "All" — reuses the existing `_send_to_telegram()`
helper (already shared by the per-asset loop and "All") with
`_BTC_ETH = [a for a in ASSETS if a in ("BTC", "ETH")]` as its asset list,
so it gets the same caching/loading behavior (only fetches whichever of
BTC/ETH aren't cached yet) and the same render/send failure breakdown as
every other send button. Disabled if Telegram isn't configured or (in
principle, though this can't currently happen) if `_BTC_ETH` were empty.

Verified offline with a new `tests/_smoke_btc_eth_button.py` (adapted from
the existing `_smoke_button.py`): simulates clicking `mcm_send_btc_eth`
with all 4 assets' results cached, confirms BTC and ETH captions are sent
and SOL/HYPE are not.

**Bridge gotcha hit again this session** (documented here since it bit a
second time): calling `device_stage_files` on a path to refresh its mtime
before a commit *overwrites the local edited copy* with whatever's
currently on the device — if that happens after an edit but before a
commit, the edit is silently lost from the local file (though not from
disk, since it was never committed). Recovered by restoring from the
offline test copy at `/tmp/mcm/pages/01_MCM_Bot.py` (which still had the
correct content, since it was copied there before the re-stage). Lesson:
either commit immediately after an edit (don't leave edits uncommitted
across a re-stage call), or keep a known-good copy outside the uploads
path to restore from if a re-stage clobbers it.

### 2026-08-20 — Telegram sends: found the actual root cause (not rate limiting)

Follow-up to the diagnostics added just below. User ran a send with the
diagnostics live: "Sent 28 report(s)... (5 failed to render/send)" — all
5 tagged **send**, 0 **render**, and specifically:
`forward_vol_steepness`, `forward_vol_steepness_25d_call`,
`forward_vol_steepness_25d_put`, `atm_iv_box_plot`, `forward_vol_matrix`
(each chart 1/1). This confirmed the earlier spacing/429 fix was
treating the wrong disease: a *genuine* 429 with 3 retries honoring
Telegram's own `retry_after` essentially never fails outright — that it
kept failing meant it probably wasn't 429 at all.

**Root cause, found by reading the actual code, not guessing:**
`lib/cmd_vol.py`'s `cmd_forward_vol_steepness` (and its 25d_call/25d_put
variants, which share the same helper) builds a chart note: `"Assumes
static curve shape; excludes <=3DTE; carry normalized to..."`. That note
gets folded into the figure's `title.text` by `fx_style.finalize()` as
`f"{base}<br><span style='...'>{note}</span>"`. `send_result_to_telegram`
was using that raw `title.text` as the Telegram caption verbatim, under
`parse_mode="HTML"`. The `<=3DTE` is literal text, not markup — but
followed later in the same string by a real `</span>` closing tag, it
reads to Telegram's HTML parser as one long malformed tag stretching
from `<=3DTE` to `</span>`, and Telegram rejects the whole send with
HTTP 400 ("can't parse entities"). This is 100% deterministic — every
send of these 3 commands' charts would hit it, every time, which is
exactly what was observed. `atm_iv_box_plot` and `forward_vol_matrix`
have clean plain-text titles (verified by reading their code — no `<` or
`&` anywhere), so their 2 failures are NOT explained by this and remain
genuinely unexplained; see below for how those are now covered anyway.

**Fix, two layers:**
1. `pages/01_MCM_Bot.py:_caption_from_title()` — the correct, targeted
   fix for the 3 confirmed cases: converts `<br>` to a newline, strips
   Plotly's `<span style=...>` tags (regex anchored to literally start
   with "span" so it can't accidentally swallow unrelated `<...>` text
   the way a generic `<[^>]+>` strip would have — that was tried and
   rejected during this fix because it would silently eat the `<=3DTE`
   note text up to the next real `>`), then `html.escape()`s whatever
   text remains so a literal `<=`/`&`/`>` displays as itself instead of
   being parsed as markup. Also truncates to 700 chars before escaping
   (escaping can expand length; do this before any 1024-char API-side
   cap, not after, so a cut can't land mid-entity).
2. `lib/telegram.py:send_message`/`send_photo` — a defensive safety net,
   not the primary fix: on any HTTP 400 while `parse_mode="HTML"` was
   used, automatically retries once with `parse_mode=None` (plain
   caption/text) rather than losing the message. This is what actually
   covers the 2 unexplained `atm_iv_box_plot`/`forward_vol_matrix`
   failures — whatever caused those (still unknown), if it was any kind
   of entity-parsing issue, this catches it too. Also added
   `_log_failure()`: any send that still fails after all retries now
   `print()`s Telegram's actual error body, so it shows up in Streamlit
   Cloud's logs — previously a failure was just `False`, no trace of why.

Removed the old `try: telegram.send_photo(..., parse_mode="HTML") except
TypeError: ...` dance in `pages/01_MCM_Bot.py:_send_photo()` now that
`send_photo` takes `parse_mode` as a real (optional, default `"HTML"`)
parameter — that dance existed only because the signature didn't used to
accept it.

Verified offline: reproduced the *exact* broken caption
(`"{base}<br><span style='...'>excludes <=3DTE...</span>"`) in a unit
test and confirmed `_caption_from_title` removes every unescaped `<`
while preserving the literal "<=3DTE" text (as `&lt;=3DTE`) and dropping
the Plotly span styling — this is a direct reproduction of the live bug,
not a synthetic one. Separately unit-tested the 400-downgrade path in
isolation (400 → auto-retry plain-text → succeeds; a 400 that also fails
plain-text gives up after exactly 2 attempts, no infinite loop; a 429
never touches the 400 path, stays on HTML throughout). Reran the full
existing suite (page smoke test, button-scoping test, 429-backoff tests,
math tests, full 21×4 command matrix) — all still pass.

**Still open:** the `atm_iv_box_plot`/`forward_vol_matrix` failures
aren't explained by this fix — their titles are clean. The 400-downgrade
safety net should paper over them if they're any variant of an
entities-parsing problem, but if they recur with the safety net in
place, `_log_failure`'s new log line is the next thing to check (or the
render/send + reason breakdown already in the UI, if it's not actually a
send-layer issue at all).

### 2026-08-20 — Telegram sends still failing after the spacing/429 fix — added diagnostics instead of guessing again

User tried the spacing + 429-backoff fix live and reported it recurred:
"Sent 21 report(s) to Telegram. (12 failed to render/send — retry if
this persists.)" 12/33 image attempts failing is a lot, and — important
tell — the earlier fix (3.2s spacing + honoring Telegram's `retry_after`
on 429) *should* have resolved a pure rate-limit problem almost
completely, since 3 retries with the exact wait Telegram asks for rarely
fails to eventually get through. That it didn't budge is a signal the
real cause may not be Telegram-side at all — could just as easily be
kaleido/Chromium failing to render under Streamlit Cloud's constrained
container (memory/subprocess limits), which the previous fix does
nothing for.

Rather than guess at a third fix with no way to verify it live (no
Telegram or Streamlit Cloud access from this sandbox), made the failure
mode observable instead:

- `lib/fx_style.py:fig_to_png()` silently swallowed its exception with no
  logging at all (unlike `dataframe_to_table_image()`, which already
  printed the error). Added a matching `print(f"Error generating chart
  image: {e}")` — this alone should make Streamlit Cloud's app logs
  (Manage app → ⋮ → Logs) show the real exception next time a chart
  fails to render, whether or not the UI-level diagnostics below get
  used.
- `send_result_to_telegram()` (`pages/01_MCM_Bot.py`) now returns
  `(ok, reasons)` instead of a bare bool — `reasons` is a list of
  per-piece failure strings, each explicitly tagged **render** (kaleido
  never produced a PNG — a Deribit/rendering-side problem) or **send**
  (the PNG was fine but Telegram rejected/didn't get it — rate limiting,
  network, timeout, even after retries). This distinction is the whole
  point: render vs. send point at completely different fixes, and a flat
  "12 failed" collapses that.
- `_send_to_telegram()` now collects every reason across the batch and
  shows a `st.expander` breaking down "N failed to render, M failed to
  send" with the full per-item list. The single-command tab's send
  button surfaces the same reasons inline. Updated all three call sites
  (`_send_to_telegram`, the "Run single command" tab) for the new tuple
  return.

**Not yet fixed — deliberately.** This change doesn't change send
behavior at all, only what's reported. Next step depends entirely on
what the expander (or the Streamlit Cloud logs) actually says next time
someone sends: if it's mostly **render** failures, the fix is on the
kaleido/rendering side (image size, subprocess/resource limits, possibly
a persistent kaleido process instead of one-shot); if it's mostly
**send** failures even after backoff, the 3.2s pacing needs to go higher
still or `_MAX_429_RETRIES` needs raising. Guessing further without that
signal risks another round-trip like this one — ask the user to check
the expander (or paste the Streamlit Cloud log lines) after the next
"Send" attempt before changing anything else here.

Verified offline: updated the existing unit test for
`send_result_to_telegram`'s new `(ok, reasons)` signature (all 4
scenarios — normal, retry-succeeds, render-failure-no-text-fallback,
text-only — still pass, with the render-failure case additionally
asserting a "render" reason is present) and reran the full page smoke
test and the button-scoping test; both still pass unmodified other than
the signature adaptation.

### 2026-08-20 — Telegram send spacing: proper 429 handling, not just a bigger delay

Follow-up to the images-only rewrite below: user reported it "looked good
overall" but a few charts were still missing after a "Send all". Root
cause: `lib/telegram.py`'s `_MIN_SEND_INTERVAL` was 1.5s (~40 sends/min)
flat, with no handling at all for Telegram's 429 "Too Many Requests" —
`send_photo`/`send_message` just returned `False` and the caller silently
dropped that image. Telegram's Bot API caps a single chat at ~1 msg/sec,
and specifically caps *group* chats at 20/min — if `chat_id` is a group,
1.5s spacing (40/min) blows straight through that, and "Send all" queues
100+ photos to one chat, so hitting it was likely, not an edge case.

Two independent fixes, deliberately not just "increase the delay":
1. Raised `_MIN_SEND_INTERVAL` to 3.2s — under the 20/min group cap with
   a buffer. This alone would probably have fixed the reported symptom.
2. Added `_post_with_backoff()`, used by both `send_message` and
   `send_photo`: on a 429, Telegram's response body includes
   `parameters.retry_after` — the exact number of seconds it wants you to
   wait. Sleep that (+0.5s buffer) and retry, up to `_MAX_429_RETRIES=3`
   times, instead of treating 429 as a hard failure. This is the part
   that actually matters: (1) alone is a best-effort guess at a safe
   pace, but a burst can still get through (right after a redeploy,
   another process hitting the same chat_id, Telegram-side variance) —
   without backoff, that single burst silently eats whichever image was
   mid-flight, which is exactly what was reported. `send_photo` rebuilds
   its `io.BytesIO` fresh on every retry attempt since a consumed stream
   can't be replayed.

Verified offline with a fake clock (`time.time`/`time.sleep` monkeypatched
so the test doesn't actually take 3.2s+ per case) and fake
`requests.post` responses: (1) two consecutive sends are correctly
>=3.2s apart, (2) a 429 with `retry_after=7` is followed by a real ~7.5s
sleep and then succeeds on retry, (3) persistent 429s give up cleanly
after exactly `_MAX_429_RETRIES + 1` attempts rather than looping
forever, (4) a network exception returns `False` without raising. Also
re-ran the earlier images-only unit tests and the full page smoke test —
all still pass. No live Telegram check possible from here (same
limitation as the images-only fix); the group-vs-private/channel chat
type is an assumption, not confirmed — if images are still occasionally
missing after this, the next thing to check is whether `chat_id` is
actually a group (20/min) vs. a channel/private chat (looser), since that
changes how tight `_MIN_SEND_INTERVAL` needs to be.

### 2026-08-20 — Telegram sends: images-only, per-asset send buttons

User feedback: Telegram sends were coming through as text instead of
chart/table images, and there was only one "Send all → Telegram" button
(no way to send just one asset).

**Root cause (best guess, unverified live — no Telegram/Streamlit Cloud
access from this sandbox):** `requirements.txt` pinned `kaleido>=0.2.1`
with no upper bound. Kaleido 1.x (released after that pin was written)
dropped its bundled Chromium and needs a separate browser install step;
on a headless host like Streamlit Cloud, `fig.to_image()` /
`write_image()` fails under kaleido 1.x, and the old
`send_result_to_telegram` caught that failure and silently fell back to
dumping the table as a raw `<pre>` text block (or, for charts, just
dropped them with no fallback at all). Pinned to `kaleido>=0.2.1,<0.3.0`
(the legacy bundled-Chromium branch) in `requirements.txt`. If sends are
still text after this ships, the pin wasn't the (whole) cause — check the
GitHub Action / Streamlit Cloud build logs for the kaleido version
actually installed, and whether a Chrome binary is reachable at runtime.

**`send_result_to_telegram` rewritten** (`pages/01_MCM_Bot.py`) to be
images-only: table → `fx_style.dataframe_to_table_image` → photo, each
chart → `fx_style.fig_to_png` → photo, one retry each if the first
attempt returns `None` (kaleido's headless Chromium occasionally misfires
cold). The raw-text-dump fallback for a failed table image is gone
entirely — a failed render is now just not sent, and shows up as a
"failed to render/send" count in the toolbar's success/error toast
instead of masquerading as the chart. The *only* remaining text message
is the command's own status string (e.g. "No basis data available."),
and only when there's neither a table nor a chart to render in the first
place — that's not a fallback, there was never an image to send.

**Per-asset send buttons.** The single "Send all → Telegram" button is
now a row of 5: one per asset (`BTC`/`ETH`/`SOL`/`HYPE`, generated from
`ASSETS` so it stays correct if that list changes) plus `All`. Refactored
the queue-building and load-then-send logic (previously inlined under
`if send_all_btn:`) into two shared helpers so both paths use the same
code: `_telegram_queue(assets, results)` builds the ordered
(asset, cmd, fig, df, text) list (handles the `vol_run` table / vol
surface expansion, same as before), `_send_to_telegram(assets, label)`
loads only whatever commands aren't already cached for those assets
(not a full 4-asset reload for a single-asset button) then sends the
queue and reports sent/failed counts.

Verified offline: unit-tested `send_result_to_telegram` in isolation
(monkeypatched `fx_style.dataframe_to_table_image`/`fig_to_png` and
`telegram.send_photo`/`send_message`) across 4 scenarios — normal
table+chart send, first-attempt image failure with a successful retry,
total image failure (confirms no text fallback fires), and the genuine
text-only case — all passed. Also ran the full page under the same rich
streamlit stub as the row-layout change above, with all 4×21 results
pre-cached and `st.button` patched so only the BTC send button reports
"clicked," then captured every `telegram.send_photo`/`send_message` call
made: exactly BTC's reports went out, nothing from ETH/SOL/HYPE — confirms
the per-asset scoping actually works, not just that it doesn't crash. No
live Telegram delivery or Streamlit Cloud check possible from here —
ask the user to try one asset button after deploying and report whether
images now come through; if not, revisit the kaleido-version hypothesis
above.

### 2026-08-20 — basis_run / block_trades_summary get dedicated full-width rows

User feedback: `/basis_run` and `/block_trades_summary` (both wide,
column-heavy tables) were being squeezed into a 1/3-width grid column
alongside chart commands, same as `/vol_term_structure` etc., and getting
visually cut off. Wanted them to render full-width, like `/vol_run`'s
table already does.

The dashboard grid (`pages/01_MCM_Bot.py`, the `asset_tabs` loop) used to
chunk each asset's commands into fixed groups of 3 straight from
`MCM_ORDER`, rendered via `st.columns(3)` — `vol_run` only got full-width
treatment because it happens to return a `list` fig, which trips the
pre-existing `multi_idx` special case (skip_figures + full-width render,
then its 2 row-mates get pulled into a smaller column row below it).
`basis_run` and `block_trades_summary` return a single (non-list) fig/df,
so they never hit that path.

Added a `FULL_ROW_COMMANDS = {"basis_run", "block_trades_summary"}` set
and rewrote the chunking from a fixed `range(0, len(order_a), 3)` to a
`while` loop: a command in that set renders alone via `render_output`
directly (no `st.columns` wrapper) and advances by 1; everything else is
still gathered into runs of up to 3, but the run stops early if the next
item is a full-row command, so that command starts clean on its own row
instead of being pulled into the preceding chunk. Also fixed a latent bug
this surfaced: the non-multi branch always did `st.columns(3)` regardless
of how many items were actually in the row, which would have left a
partial last row (now possible, since full-row commands can produce
uneven remainders) with 1-2 populated columns and dead blank space beside
them — changed to `st.columns(len(row_items))`.

Verified offline (no live Streamlit available in this sandbox — see
Testing below): built a richer streamlit stub than existed before
(`_Col` context-manager columns/tabs/popover, session_state attribute
access, widget `key=` duplicate detection) at `/tmp/stubs/streamlit.py`
in the sandbox (not committed — sandbox-local, rebuild next time per the
Testing section), ran `pages/01_MCM_Bot.py` through `runpy.run_path` with
a fully populated `mcm_all_results` cache (all 4 assets × 21 commands via
`fake_deribit`) so the grid loop actually executes end-to-end, and
additionally instrumented `st.caption`/column nesting depth to confirm
`basis_run` and `block_trades_summary` land at the same depth as
`vol_run`'s table (own row) while their former row-mates stay one level
deeper inside a shared `st.columns()` block. No duplicate-key errors, no
exceptions. No live/visual check — flagged to the user per usual.

### 2026-08-20 — persistent storage, code review, UI overhaul, project setup

Starting point: a live `AttributeError` on `fx_style.local_now()` that
persisted after a push (turned out to be a stale Streamlit Cloud worker,
not a code bug — confirmed via `WebFetch` on the raw GitHub content).

- **Full code review** surfaced and fixed a real regression in
  `lib/fx_style.py` (SGT timezone conversion, chart watermark, and
  legend/note collision-avoidance had all silently broken from an earlier
  edit) — fixed surgically, not via blind revert, since dependent code had
  co-evolved. Extended the same timezone/theme/watermark treatment to 5
  pages that never had it (`02, 06, 07, 08, 10`), cleaned up ~27 deprecated
  `use_container_width` usages app-wide.
- **Fixed a real bug behind the user's "make lookback periods honest"
  request**: `history.surface_row_at()` always returned *some* row with no
  distance check, so short-history assets silently showed the same stale
  row under two different wrong labels ("1 Week Ago" and "1 Month Ago").
  Added `surface_row_near()` + `_lookback_points()`/`_fmt_ago()` to relabel
  honestly and dedupe.
- **Performance**: found `surface.option_vols_by_dte()` being recomputed up
  to 113x per asset per "Load all" pass; added a 10s cache
  (`_VOLS_CACHE`), verified 113→1 calls.
- **Built the persistent-storage pipeline** (this session's main feature):
  `scripts/record_snapshot.py` + `.github/workflows/record_snapshots.yml`
  (hourly cron, commits to a `snapshots` branch, not `main`, to avoid
  triggering redeploys) + `lib/history.py` remote-fetch/merge. Then a
  follow-up **code review of that pipeline itself** caught and fixed two
  real bugs before they could bite in production: (1) the workflow's
  "does the branch exist" check swallowed *any* failure, including a
  transient network blip, and would have force-pushed over — i.e.
  destroyed — all accumulated history on a bad run; fixed via
  `git ls-remote`'s exit code to distinguish "confirmed absent" from
  "couldn't tell." (2) the "skip commit if nothing new" check compared
  against the wrong git ref and would have committed every hour regardless
  of whether new data existed; fixed to compare against `origin/snapshots`.
  Also added CSV size-bounding (`_thin_snapshot_file` — full resolution for
  14 days, 1 row/day beyond that) since hourly recording would otherwise
  grow forever; caught a timestamp-format bug in that fix during
  verification (thinned rows and fresh-appended rows used different
  string formats, which made pandas silently drop rows on read) —
  documented above under "Persistent snapshot storage," worth remembering
  if this code is touched again.
- **Fixed `pages/01_MCM_Bot.py`**: `history.record_snapshot(ASSETS[0])` was
  only ever recording BTC locally, regardless of which asset the user was
  viewing — the user caught this by noticing ETH/SOL/HYPE seemed to be on
  "old methodology." Now loops over all 4 assets (still cheap — each is
  independently throttled to once per 30 min).
- **Fixed a GitHub Actions Node.js 20 deprecation warning** by bumping
  `actions/checkout` and `actions/setup-python` to their latest majors
  (both now run on Node 24).
- **UI overhaul** on user's report of "confusing and cluttered, have to
  scroll down to see everything": collapsed the Settings + Load-reports
  sections (2 subheaders, 4 always-visible captions, several stacked
  buttons) into one compact toolbar row, moved "Load selected" into a
  popover, moved "What each report shows" to the sidebar. Replaced the
  serial per-asset dashboard loop (scroll past all of BTC, then ETH, then
  SOL, then HYPE, then the single-command tool at the very bottom) with
  `st.tabs()` — one tab per asset plus a "Run single command" tab.
  Built a purpose-made Streamlit stub (tracks widget `key=` uniqueness,
  supports `session_state` attribute access, returns real context managers
  from `columns`/`tabs`/`popover`) specifically to smoke-test this since
  the existing offline harness never exercised `pages/*.py` — caught a
  leftover duplicate-widget-key bug (would have crashed the live page)
  before shipping.
- **Set up this file** and a Cowork Project ("MCM Analytics") so future
  sessions don't start from zero — see the top of this file for what it
  covers.
