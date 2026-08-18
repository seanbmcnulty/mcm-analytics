# Historical vol data: the problem and the options

## The problem

Deribit's public API has **no option-IV history endpoint**. It serves the live
option chain (`mark_iv` per instrument, right now) and that is all. The FalconX
bot got every historical vol series from Amberdata's constant-maturity delta
surface, which is exactly the thing Deribit does not publish.

Charts that need history:

| Chart | What it needs |
|---|---|
| ATM IV box plot | 90 days of ATM IV per tenor (the whole distribution) |
| Vol / skew term structure | ATM + 25Δ surface at 24h, 1w, 1m ago |
| Vol / skew time series | 30 days of ATM, 25Δ, 10Δ for one expiry |
| Intraday vol / skew | 24 hours of the same, hourly or better |
| Vol smile | Smile at 1d / 1w / 1m ago, plus the high–low cloud |

Everything else in the bot — forward vols, carry, matrix, basis, flow, RV,
funding, moon — is pure live data and is already exact.

**Today's behaviour:** the app re-levels the current surface through Deribit's
DVOL index and labels the result "reconstructed". Level moves are real; the
*shape* (skew, smile curvature) is frozen at today's. Good enough to be
directional, not good enough to trade a skew view off.

---

## Option 1 — Rebuild from the Deribit trade tape ★ best free source

`public/get_last_trades_by_currency_and_time` accepts arbitrary start/end
timestamps and every option trade carries its own `iv` and `index_price`. That
is **genuinely observed historical implied vol**, not a proxy.

Method: pull the tape for a window → for each trade compute its BS delta from
its own IV and index price → bucket into ATM / 25Δ / 10Δ per expiry → take an
hourly volume-weighted median per bucket.

- **Pros** — real observed data; free; covers all four assets; covers every
  tenor that actually trades; can backfill *retrospectively*, so the box plot
  works on day one.
- **Cons** — sparse where strikes are illiquid (SOL/HYPE wings, long tenors);
  many paginated calls for 90 days; unknown retention depth on the public
  endpoint.
- **Unknown to resolve first:** how far back the tape goes and how dense it is.
  `tools_probe_history.py` answers exactly that — run it and read the verdict.
- **Effort:** medium–high. Best as a one-off backfill written to the snapshot
  store, then kept current by Option 2.

## Option 2 — Scheduled snapshot recorder (GitHub Actions) ★ best effort/reward

A workflow on a cron (every 15–30 min) calls Deribit, computes the surface with
the same code the app uses, and commits a row to `data/snapshots/`.

- **Pros** — trivial to build; free on a public repo; identical maths to the
  live charts, so no methodology drift; survives Streamlit restarts (which the
  current in-app recorder does not — Streamlit Cloud's disk is ephemeral).
- **Cons** — only accrues *forward*. The 90-day box plot is only fully
  populated after 90 days.
- **Cost:** ~50 KB/day at 30-minute granularity. Negligible.
- **Effort:** low. This is the one I'd start today regardless of what else we do.

## Option 3 — Same recorder, but into a real database

Identical to Option 2 but writing to Supabase / Neon Postgres / MongoDB Atlas
free tier instead of committing CSVs into a code repo.

- **Pros** — proper time-range queries; no repo churn; scales past a year.
- **Cons** — another service and a credential to manage.
- **Effort:** low–medium. Worth it if the snapshot store outlives a few months.

## Option 4 — DVOL re-levelling (what runs today)

- **Pros** — instant, free, no infrastructure; the level moves are real for
  BTC/ETH.
- **Cons** — smile/skew shape is frozen; SOL and HYPE fall back to a realized-vol
  proxy, which is weaker still.
- **Keep it** as the day-one fallback and as the gap-filler when the recorded
  store has holes. It should never be the primary source once Option 1 or 2 lands.

## Option 5 — Invert option candles

`get_tradingview_chart_data` on individual option instruments gives traded price
history; invert Black-Scholes to get IV.

- **Pros** — real, candle-granular.
- **Cons** — one API call per instrument; only where trades occurred; a fixed
  strike drifts away from constant-delta as spot moves; inverse (BTC-quoted)
  prices need converting. Mostly dominated by Option 1, which gets the same
  information in fewer calls.

## Option 6 — Pay for history

Amberdata (what FalconX used), Laevitas, Block Scholes, Tardis.dev, Coinglass.

- **Pros** — instant, full-fidelity, restores FalconX behaviour exactly.
- **Cons** — cost. But note the "Deribit-only" constraint was a practical
  starting point, not a principle — if fidelity matters more than cost, this is
  the honest answer and it is the shortest path.

---

## Recommendation

1. **Ship Option 2 now** — the recorder starts accruing real history today, and
   it is an afternoon of work.
2. **Run `tools_probe_history.py`** to find out whether Option 1 is viable. If
   the tape is deep and dense, backfill from it so the box plot and time series
   are meaningful immediately rather than in three months.
3. **Keep Option 4** as the labelled fallback for gaps and for the first days.
4. **Revisit Option 6** only if you need exact historical smile shape — e.g.
   for backtesting a skew signal, where a frozen-shape reconstruction would be
   actively misleading.

Options 1, 2 and 4 compose: the app already prefers recorded snapshots over the
reconstruction, so a backfill and a recorder both slot in behind the same
interface (`lib/history.surface_history`) with no chart changes.
