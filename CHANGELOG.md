# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
