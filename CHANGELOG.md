# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
