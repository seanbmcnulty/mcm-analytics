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
- `lib/surface.py:_VOLS_CACHE` — 10s TTL, avoids redundant full option-chain
  + Black-Scholes-delta-search recomputation (was up to 113 redundant calls
  per asset per page load before this was added).
- `lib/history.py:_DRIVER_CACHE` — 300s TTL, the DVOL/RV re-levelling series.
- `lib/history.py:_REMOTE_SNAPSHOT_CACHE` — 300s TTL, the GitHub-branch fetch.
- All four get cleared together by the sidebar "Clear data cache" button
  in `pages/01_MCM_Bot.py` — if you add a new cache, wire it in there too.

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
modules don't touch `st` directly.

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
pages/06,07,08,10_*.py          secondary analytics pages (RV, regime, correlation, macro)
pages/03,04,05,09,11,12,13,14   retired — moved to _to_delete/ on the device, not in git

lib/deribit.py                  Deribit public API client, caching, rate limiting
lib/surface.py                  option chain parsing, BS delta search, vol surface math
lib/history.py                  historical IV reconstruction (see above), snapshot I/O
lib/cmd_vol.py                  vol-related chart/table commands (term structure, skew,
                                 smile, forward vols, ATM box plot, etc.)
lib/cmd_market.py               market/basis/funding/block-trade commands
lib/commands.py                 command registry (COMMAND_NAMES, run_command dispatch)
lib/constants.py                ASSET_CONFIG, ASSETS, color scheme
lib/fx_style.py                 plotly theming, SGT timezone helpers (local_now, to_local),
                                 watermark, chart title/legend/margin layout (finalize())
lib/vol_math.py                 Black-Scholes, implied vol, delta math
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

- `pages/02_Block_Trades.py`'s direct Deribit calls could be routed through
  `lib/deribit.py` for a single shared rate budget.
- A few `frame.iterrows()` loops in `lib/cmd_vol.py` (smile snapshot
  high/low band) are a minor pandas anti-pattern — low-risk to vectorize,
  but not currently a real bottleneck since history frames are now
  size-bounded by the thinning logic above.

## Session log

Keep this updated: when a session makes a non-trivial change, decision, or
finds a bug worth remembering, add a dated entry below before the session
ends. Newest entry on top. This is how continuity works across sessions —
nothing here persists otherwise.

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
