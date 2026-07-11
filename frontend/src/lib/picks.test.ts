import { describe, it, expect } from "vitest";

import { finalScoreLabel } from "./picks";
import type { GameStatus, SlateGame } from "./picks";

/**
 * Minimal SlateGame fixture — only the fields finalScoreLabel reads (status,
 * home/away score, and each team's abbreviation). Everything else is filled
 * with inert placeholders so the object type-checks as a SlateGame.
 */
function makeGame(overrides: {
  status: GameStatus;
  home_score: number | null;
  away_score: number | null;
}): SlateGame {
  return {
    game_id: 1,
    kickoff_at: null,
    home_team: { team_id: 1, abbreviation: "HOM", display_name: "Home" },
    away_team: { team_id: 2, abbreviation: "AWY", display_name: "Away" },
    spread: null,
    total: null,
    favorite_team_id: null,
    underdog_team_id: null,
    locked: false,
    eligibility: {
      UNDERDOG_COVER: false,
      FAVORITE_COVER: false,
      OVER: false,
      UNDER: false,
      MISC: false,
    },
    ...overrides,
  };
}

describe("finalScoreLabel", () => {
  it("returns 'AWAY n @ HOME n' for a FINAL game with both scores", () => {
    const game = makeGame({ status: "FINAL", home_score: 24, away_score: 17 });
    expect(finalScoreLabel(game)).toBe("AWY 17 @ HOM 24");
  });

  it("returns null for a FINAL game with a null score (no invented 0-0)", () => {
    const game = makeGame({ status: "FINAL", home_score: 24, away_score: null });
    expect(finalScoreLabel(game)).toBeNull();
  });

  it("returns null for a SCHEDULED game even when scores are present", () => {
    const game = makeGame({
      status: "SCHEDULED",
      home_score: 24,
      away_score: 17,
    });
    expect(finalScoreLabel(game)).toBeNull();
  });
});
