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
