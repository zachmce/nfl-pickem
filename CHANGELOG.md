# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-07-16

Feature release — the Discord bot learns to reason about the slate. A deterministic
Elo rating engine, built over a seeded 1999–2025 corpus of real NFL results and
closing lines, now backs both the whole-week `slate_predictions` answer and the
single-game `prediction` answer. The framing is deliberate and unchanged: the bot is
an **explainer and an independent cross-check, not a tipster** — a feasibility spike
established that a simple Elo does not beat the closing line, so the model is
surfaced as color against the market, never as a bet. Also folds in five fixes and a
dependency sweep that clears a HIGH CVE from the published images.

### ⚠️ Upgrading (deploy stack)

- **New migration (0017) — run it before serving traffic.** `historical_game` is a
  new table, deliberately separate from `game` so the static 1999–2025 corpus can
  never enter the active-season `max(game.season)` computation. It carries none of
  `game`'s poller/odds-freeze/notify state and is fully reversible.
- **First boot does real work.** The table is created empty and filled by an
  idempotent, fail-loud startup upsert of a committed 7,276-game artifact (~425K
  CSV). Expect the first backend start after upgrading to take noticeably longer;
  subsequent boots are no-ops. No manual seeding step is required.

### Added

- **Deterministic Elo rating engine (#137).** A pure, read-only engine
  (`app/services/ratings.py`) turns the historical corpus plus live FINAL games into
  per-team ratings and expected margins. The union stream is ordered by
  (season, week, date) with a 1/3 regression at each season boundary. No table, no
  migration — a compute layer only.
- **Historical NFL results + closing lines, seeded from nflverse (#135).** The
  `historical_game` corpus (7,276 games, 1999–2025) plus a dev-only regeneration
  script. This is the cold-start data the rating model needs.
- **`slate_predictions` — model vs. the line for the whole week.** A new @mention Q&A
  intent that surfaces the engine's expected margin against the market for every game
  on the current slate, framed as an independent cross-check. Long replies are chunked
  to respect Discord's 2000-character limit.
- **Pick-type vocabulary for bot answers (#139).** You can now ask in the app's own
  language — "who do you like as a favorite/underdog this week?", "best mortal lock?".
  Favorite/underdog rank by the strongest cover lean against the market; mortal lock
  ranks by outright win probability (a deliberately different axis). Over/under
  honestly declines: there is no totals model. The selection is code-derived, not
  LLM-chosen.
- **A worked pick-type example on the Help page (#134).** A new "See it in action"
  section resolves all four base pick types against a single Chiefs 27 / Broncos 20
  example, so the scoring rules are shown rather than only described.

### Changed

- **The single-game `prediction` intent now shares the slate model.** Both intents
  route through one `_model_line_lean` helper, so they can no longer contradict each
  other; the old line-parroting "My call: favorite to cover" is gone. Model margins
  round to whole points ("a hair" below one) and gaps of 7+ points against the line
  carry a low-confidence hedge, so noisy outliers don't read as authoritative.

### Fixed

- **MISC pick scoring is gated on the game being FINAL (#128).** A graded MISC pick on
  a future-week game leaked its points into season standings before the game had been
  played.
- **Discord Q&A news is team-optional (#132).** Asking for news about something that
  isn't a real team now falls back to league-wide news instead of erroring as unknown.
- **`game.final` quips no longer cluster on unpicked games (#131).** A deterministic
  per-matchup angle (selected by a stable hash of the matchup, not `hash()`) forces
  structural divergence across independent LLM calls.
- **AppSetting model metadata aligned to migration head (#129).** Clears a
  false-positive Alembic autogen drift. DB-level uniqueness is unchanged; no new
  migration.
- **Frontend lint residue from react-hooks@7 resolved (#130).** Ten set-state-in-effect
  findings fixed and the rules promoted to error.

### Security

- **Backend base image clears a HIGH CVE (CVE-2026-11940).** The Chainguard python
  builder and runtime digests move to a base carrying `python-3.14` `3.14.6-r3`,
  fixing a `tarfile.extractall()` filter bypass present in `3.14.6-r1`. The CVE was
  published after the previous release, so v1.2.4's published images carry the
  vulnerable package — **this release is how the fix reaches the images.**

### Build

- **Dependency sweep.** `codeql-action` v4.36.3 → v4.37.0 and a `setup-uv` SHA repin
  (#146); node:26-alpine (#143) and oss-fuzz base-builder-python (#141) digest
  refreshes; vite 8.1.3 → 8.1.4 (#144), which also resyncs the `package-lock.json`
  version stamp that had drifted from `package.json`.

## [1.2.4] - 2026-07-11

Enhancement patch — the Rules page becomes a Help page that also documents the
Discord bot. User-facing content/navigation change; no API or schema changes.

### Changed

- **The "Rules" page is now a "Help" page (#124).** The dedicated Rules page is
  replaced by a single `/help` page that folds in all the existing rules content
  (picks, roster, pick window, lines lock, scoring, standings) **and** adds a guide to
  using the Discord bot — how to @mention it, the questions it can answer (with example
  phrasings), and the updates it posts automatically. Every topic is a collapsible
  accordion. The old `/rules` URL redirects to `/help`, so existing links keep working.

## [1.2.3] - 2026-07-11

Bug-fix patch — a single frontend fix for confusing post-password-change behavior.
User-facing behavior improves; no API or schema changes.

### Fixed

- **Password change now redirects to the login page (#119).** Changing a password
  invalidates the current session server-side (correct), but the app left the user on the
  Profile page holding a dead session — the next navigation failed with no explanation.
  On a successful change the app now clears the local session and redirects to `/login`
  with a notice ("Password changed — sign in again with your new password."), so the user
  understands they must sign in again with the new password. Error paths (confirmation
  mismatch, wrong current password) are unchanged and still report inline on the Profile
  page.

## [1.2.2] - 2026-07-09

Bug-fix patch — two frontend data-freshness fixes surfaced during exploration.
User-facing behavior improves; no API or schema changes.

### Fixed

- **Current-week status now refreshes during in-app navigation (#103).** The header
  week chip and the sub-header context bar (week number + pick-window state) were fed by
  a fetch that ran only once per full page load, inside the persistent app shell that
  client-side navigation never remounts — so the labels went stale until a hard browser
  refresh (a stale "picks open" label could mislead a player into attempting to edit
  locked picks). The two displays now share a single `WeekProvider` that re-fetches on
  route change and on window focus, so the status stays current without a reload.
- **Admin pick-override editor no longer jumps when changing weeks (#104).** Switching
  the week (or season) in the admin edit-picks panel tore the roster down to a "Loading…"
  placeholder on every change; because the panel sits at the bottom of the page, the
  collapse shrank the document and the browser's scroll clamp made the whole page lurch.
  The editor now keeps the current roster on screen while the next week loads (with a
  subtle "Updating…" hint), holding the layout stable.

## [1.2.1] - 2026-07-08

Cleanup patch — the TIER 3 items from the external Fable code review (internal
consistency, release ergonomics, and documentation accuracy). No user-facing
behavior changes.

### Changed

- **Backend version is single-sourced from package metadata (T16).**
  `FastAPI(version=...)` now reads `importlib.metadata.version("nfl-pickem-backend")`
  instead of a hardcoded string, so `pyproject.toml` is the sole authority and the
  release version stamp is one file smaller. (#99)

### Fixed

- **`User.created_at` default consistency (T13).** Switched the default from a SQL
  function element (`sa.func.now`) to `_utcnow`, matching every other timestamp
  column — the in-memory attribute is now a real `datetime` before flush rather than a
  SQL expression. Python-side only; no schema change. (#100)

### Removed

- **Dead `make shell-backend` target (T15).** The runtime image is distroless (no
  shell) and the dev backend builds that same stage, so the target could never work. (#98)

### Documentation

- **Pick-window write/read contract clarified (T17).** The `pick_window` docstrings now
  state that the week-level window is the operative gate on user writes, and that
  per-game `is_game_locked` is defense-in-depth for writes and the operative primitive
  for read paths (standings pick-visibility). Behavior unchanged. (#101)
- **Cookie-auth same-origin constraint documented (T18).** Noted (in `config.py` and
  `DEPLOY.md`) that the `SameSite=Lax` session cookie makes cross-origin cookie auth
  unsupported regardless of CORS configuration — deploy the SPA and API on the same
  origin behind the proxy. (#101)

## [1.2.0] - 2026-07-08

Hardening & quality pass — the six actioned TIER 2 findings from the external Fable
code review. The frontend gains its first linter and test suite; the deploy stack
gains network segmentation and Redis authentication; the SPA gains a baseline CSP
and security headers.

### ⚠️ Upgrading (deploy stack)

- **Redis now requires a password.** `docker-compose.deploy.yml` starts Redis with
  `--requirepass` fail-closed, so the stack refuses to start unless your `.env`
  sets `REDIS_PASSWORD` **and** an authenticated
  `REDIS_URL=redis://:<password>@redis:6379/0` (both — the URL is what the
  backend/worker/bot use to authenticate). See `.env.example` / `DEPLOY.md`.
- **Update the compose file on the server, not just the images.** `docker compose
  pull` refreshes images only; the new `redis` command, the segmented networks, and
  the authenticated healthcheck live in `docker-compose.deploy.yml` itself — copy the
  new version over before `up -d`. The `redisdata` volume is preserved (Redis simply
  gains a password).

### Security

- **Deploy-stack network segmentation + Redis auth (T8).** The internet-facing nginx
  container no longer shares a network with the data plane. Two bridges (`frontend_net`,
  `backend_net`) isolate it: nginx reaches only the backend, while `db`/`redis` sit on
  the backend network alone, so a hypothetical nginx compromise has no route to
  postgres:5432 or redis:6379. Redis additionally requires a password. (#96)
- **Baseline Content-Security-Policy + security headers on the SPA (T7).** nginx now
  emits `Content-Security-Policy` (scoped `img-src` for the Discord avatar and ESPN
  team-logo CDNs), `X-Content-Type-Options: nosniff`, `Referrer-Policy`, and
  `server_tokens off`. (#91)
- **Session cookie hygiene (T11).** Logout now deletes the session and CSRF cookies
  with attributes matched to how they were set, so a secure-context browser reliably
  clears them; `SESSION_COOKIE_NAME=__Host-session` is documented for prod. (#92)

### Added

- **Frontend linting and tests (T12).** ESLint (flat config: typescript-eslint +
  react-hooks + react-refresh) and Vitest (+ Testing Library + jsdom) are now part of
  the frontend, wired as **blocking** CI gates. `react-hooks/exhaustive-deps` runs as
  an error gate (the hand-rolled data-fetching hooks are clean). (#93)

### Changed

- **CSRF middleware converted to pure ASGI (T10).** The double-submit CSRF check moved
  off Starlette's `BaseHTTPMiddleware` to a class-based ASGI middleware — dropping the
  response-buffering / BackgroundTask / contextvar caveats and per-request overhead.
  Externally byte-identical (same guards, same 403 envelope, same middleware order). (#95)
- **Notifications reuse a single Redis client (T9).** The Discord event publisher and
  cooldown claim now share a memoized Redis client instead of constructing (and
  discarding) a new connection pool per event. Best-effort / fail-open contracts
  unchanged. (#94)

### Fixed

- **nginx stale-`index.html` after deploy + stock-config shadowing (T7).**
  `index.html` is now served `no-cache`, so a redeploy's purged content-hashed assets
  can't strand a client on a stale index. The Chainguard base image's stock
  `nginx.default.conf` (which shadowed our server block for `Host: localhost`, dropping
  headers and the SPA/API routing) is now overwritten. The no-flash theme bootstrap was
  moved to an external script so it survives the strict CSP. (#91)

## [1.1.8] - 2026-07-07

Code-review hardening pass — the four actioned TIER 1 findings from an external
review (the other two were reviewed and consciously deferred as accepted risk for
the private, single-operator deployment).

### Security

- **Prompt-injection defense on the LLM chat layer.** User-controlled MISC
  prediction text is now sanitized and wrapped in a single labeled data-fence
  before it crosses into the chat-personality LLM prompt: fence markers
  (`<<<` / `>>>`) and control characters are stripped from the input (so a player
  cannot break out of the fence or smuggle a fake instruction line) and the text
  is length-capped; the misc-graded role prompt now instructs the model to treat
  the fenced text as quoted data and never follow instructions inside it. A player
  can no longer steer the bot's public output by phrasing their prediction as an
  instruction. (#89)

### Fixed

- **Concurrent pick submissions no longer return a raw 500.** A check-then-insert
  race on the same pick slot (two simultaneous submits) could hit the partial
  unique index at commit and surface an unhandled `IntegrityError`. Both the user
  submit and admin set-pick paths now translate the Postgres unique-violation
  (SQLSTATE 23505) into the standard 409 conflict envelope, honoring the "never a
  raw 500" contract. (#87)

### Changed

- **Backend image installs dependencies frozen from `uv.lock`.** The builder stage
  now runs `uv sync --frozen` against the committed lockfile instead of
  re-resolving `pyproject.toml` ranges at build time, so the dependency set that
  CI audits and SBOMs is byte-for-byte the set the image ships; a lock/manifest
  drift now fails the build. (#86)
- **Image publish is gated on the full quality set.** `publish-backend` now
  additionally requires `typecheck`, `deps-audit`, `semgrep`, and `osv-scan`, and
  `publish-frontend` requires `semgrep` and `osv-scan` — so a signed image can no
  longer ship past a failing type check, dependency audit, or SAST/OSV gate. The
  path-skip-tolerant gating pattern is preserved. (#88)

## [1.1.7] - 2026-07-07

### Security

- Added **osv-scanner** as a blocking CI dependency gate over OSV.dev, scanning
  the backend (Python) and frontend (npm) manifests and uploading SARIF to the
  Security tab. Its marginal value over the existing pip-audit / npm audit /
  Trivy / dependency-review stack is the `ossf/malicious-packages` (`MAL-`)
  advisories — known-malicious package IDs that the CVE-only tools don't carry.
  Blocking from day one. (#84)
- Added a two-pass **Semgrep** SAST job to CI with a validated custom canary
  ruleset, gated through `ci-complete`. (#81)
- **Dependabot** now waits a 5-day cooldown before opening version-bump PRs for
  the `github-actions`, `pip`, and `npm` ecosystems, so a malicious or broken
  release has time to be yanked before it is pulled. The container base-image
  (`docker`) ecosystems stay uncooled so CVE patches remain prompt. (#82)
- Refreshed the backend Chainguard / Wolfi Python base-image digest (#83) and the
  frontend Chainguard nginx base-image digest (#79).

### Changed

- No application or runtime behavior changes — this is a CI / supply-chain
  hardening release. The shipped images are rebuilt on refreshed Chainguard base
  digests but are otherwise functionally identical to 1.1.6.

## [1.1.6] - 2026-07-07

### Security

- Backend image migrated to a Chainguard / Wolfi glibc-minimal multi-stage base:
  released-digest Trivy findings dropped 163 → 0. The `perl-base` package and both
  previously-unfixed CRITICALs (CVE-2026-8376, CVE-2026-42496) are gone from the
  shipped image. (Shell-free `app.bootstrap` migrate/seed entrypoint, #72;
  Chainguard Python base, #73.)
- Frontend runtime migrated to a Chainguard nginx base: released-digest Trivy
  findings dropped 13 → 0. (#77.)

### Changed

- Operator / upgrade note: the backend worker's non-root uid changed 10001 →
  65532, so a pre-existing `celerybeat` named volume needs a one-time
  chown/recreate on staging and dev (see DEPLOY.md §4). The frontend has no such
  volume.

## [1.1.5] - 2026-07-06

### Fixed

- Released-digest Trivy SARIF now lands on the `main` branch instead of the
  release tag. `release.yml` runs on the `v*` tag push, so the report-only
  shipped-image scans (added in v1.1.4) were filed under `refs/tags/v*` — and
  GitHub's code-scanning alert list only shows branches and pull requests, never
  tags, so those findings were stored but invisible in the Security tab. The
  upload steps now pass an explicit `ref`/`sha` so the results appear in the
  browsable default-branch view and refresh in place each release. Still
  report-only — never gates a release.

## [1.1.4] - 2026-07-06

### Added

- The Discord `freeze.week` card now also fires when a week's betting lines
  freeze by the clock (the time-based freeze), mirroring the window open/close
  cards from #55 — a week whose lines froze purely on schedule is no longer
  announced silently.
- CI publishes Trivy (image scan) and gitleaks (secret scan) SARIF results for
  the released image digest to the GitHub Security tab, so vulnerability and
  secret findings for shipped images surface in code scanning.

### Fixed

- Demo reseed now clears the `lines_frozen` admin override, which could
  otherwise remain stale after the demo time anchor jumps to a new week.

## [1.1.3] - 2026-07-06

### Fixed

- Release workflow: the image-signature verification gate now retries until the
  publish job's signing step completes, closing a race that could false-fail a
  release cut when `release.yml` ran ahead of `ci.yml`'s image signing. (Also
  reverted an earlier, incorrect cosign-v3 workaround for the same symptom.)

## [1.1.2] - 2026-07-06

### Fixed

- Release workflow: the image-signature verification gate now uses cosign v3, to
  match the signature format the images are actually signed with. cosign v2.6.3
  (pinned for the checksum blob-signing steps) could not read a v3 signature,
  which blocked the v1.1.1 signed GitHub Release. The v1.1.1 container images
  were published and signed correctly; this release completes the signed Release
  that v1.1.1 could not produce.

## [1.1.1] - 2026-07-06

### Added

- Admin grade mode can now edit the content text of an existing MISC pick, not
  just its grade — fixing a typo or clarifying the prompt no longer requires
  deleting and recreating the pick.

### Fixed

- Time-based week open/close now fires the Discord `window.opened` /
  `window.closed` cards. Previously these only fired on an explicit state change,
  so a week that opened or closed purely by the clock was announced silently
  (#55).
- The "My Picks" context bar no longer shows a "closes …" clause once the pick
  window is closed; the clause is now gated on the window actually being open
  (#58).

### Security

- **Release images are now built once and the exact bytes are scanned and
  shipped.** The Trivy CVE scan runs against the same OCI artifact that is
  promoted to GHCR (scanned digest == published digest), instead of a separately
  rebuilt image. Per-PR image scanning is unchanged.
- **Published image digests are cosign-verified before a release is cut.** The
  release workflow fails closed if a digest was not keyless-signed by our own
  publish pipeline, so a stray or tampered digest can never be checksummed and
  released.
- **An SPDX SBOM is now attached to each GitHub Release** (`sbom-backend.spdx.json`,
  `sbom-frontend.spdx.json`), so consumers can inspect the exact dependency
  inventory of the images they pull.

## [1.1.0] - 2026-07-05

### Added

- Rich Discord chat embed cards for every key moment of the week:
  - **Weekly recap** "closing ceremony" card — full standings with each player's
    weekly point gain, the week's best call / biggest bust (by upset magnitude),
    and a mortal-lock scoreboard.
  - **Game-final** result card with per-player Busted/Cashed impacts and
    mortal-lock markers.
  - **MISC-graded** result card (prediction, verdict, points) crediting the grader.
  - **Pick-window** open/close cards.
  - **Lines Locked** card announcing when a week's betting lines freeze.
- Season-storyline callbacks woven into the weekly recap narration (mortal-lock
  streaks, lead changes, and season superlatives).

### Changed

- Discord bot personality: less repetitive sign-offs and stock phrasing, tighter
  emoji/formatting, and no self-signing.
- The notifier can post a single event to both the ops-log and the main chat
  channel (independent dual-channel dispatch).
- The "My Picks" per-game badge now reflects game status (Scheduled / In progress /
  Final) instead of a misleading per-game lock label.

### Fixed

- The Weekly view stays on an in-progress week instead of jumping ahead to next
  week mid-slate.
- CI publishes each container image independently, so a change to a single image
  still ships that image.

## [1.0.0] - 2026-07-02

### Added

- Weekly NFL pick submission with per-game lock windows tied to kickoff.
- Scoring engine and season-long leaderboards.
- Discord bot for picks, results, and standings.
- LLM chat personality layer with admin-controlled voice/personality swapping.
- Time-shifted demo-season walkthrough for end-to-end validation without live games.
- Hardened CI and supply chain: SBOM generation, SLSA provenance, cosign-signed
  container images, and pinned GitHub Actions.

[1.1.0]: https://github.com/zachmce/nfl-pickem/releases/tag/v1.1.0
[1.0.0]: https://github.com/zachmce/nfl-pickem/releases/tag/v1.0.0
