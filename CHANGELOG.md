# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
