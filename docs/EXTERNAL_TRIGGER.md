# External 15-minute trigger for the snapshot recorder

GitHub's own `schedule:` trigger is best-effort and in practice starts only a
small fraction of runs (about 9% between Oct 2 and Oct 16 2026). The reliable
fix is an outside timer that calls the workflow's `workflow_dispatch` API
every 15 minutes. The crons in `record_snapshots.yml` stay as a backup, and
`--min-interval 780` (13 min) means duplicate triggers do nothing.

Nothing in this doc has been set up yet. You need to do both steps yourself.

## 1. Create a fine-grained GitHub token (github.com)

Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate new token

- **Name:** `cron-job.org snapshot trigger`
- **Expiration:** 1 year (put a calendar reminder to renew it; when it expires, runs
  quietly stop and the watchdog's stale alert will fire)
- **Repository access:** *Only select repositories* → `seanbmcnulty/mcm-analytics`
- **Repository permissions:** **Actions: Read and write**. Leave everything else at
  *No access*. (Metadata: Read-only gets added automatically.)

Copy the token. Don't paste it into chat or commit it.

## 2. Create the cron job (cron-job.org, free account)

Create cronjob:

- **URL:** `https://api.github.com/repos/seanbmcnulty/mcm-analytics/actions/workflows/record_snapshots.yml/dispatches`
- **Schedule:** every 15 minutes (`*/15 * * * *`). The backup crons (:17/:47) are harmless: `--min-interval` makes them no-ops.
- **Advanced → Request method:** `POST`
- **Advanced → Headers:**
  - `Authorization: Bearer <your token>`
  - `Accept: application/vnd.github+json`
  - `X-GitHub-Api-Version: 2022-11-28`
  - `Content-Type: application/json`
- **Advanced → Request body:** `{"ref":"main"}`
- **Notifications:** turn on "notify on failure".

Use **Test run**. A good response is **HTTP 204** with an empty body, and a new
"Record vol snapshots" run appears under the repo's Actions tab within about a
minute. A 401 means a bad token, 403 means the Actions permission is missing,
and 404 means a wrong URL or a token not scoped to this repo.

## Checking it

- After a few hours, each run's **Summary** page shows the health table: rows
  added, latest row age, and gaps over 2h in the last 24h.
- The `Snapshot watchdog` workflow alerts on Telegram if the newest row is more
  than 2h old.
