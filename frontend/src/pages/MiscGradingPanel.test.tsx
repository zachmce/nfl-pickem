import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import * as admin from "../lib/admin";
import { gameLabel, gradeCounts, type AdminMiscPickRow } from "../lib/admin";
import MiscGradingPanel from "./MiscGradingPanel";

function row(overrides: Partial<AdminMiscPickRow>): AdminMiscPickRow {
  return {
    user_id: 1,
    display_name: "ada",
    game_id: 10,
    misc_text: "Mahomes 3+ TDs",
    result: "PENDING",
    points: 0,
    away_abbr: "KC",
    home_abbr: "BUF",
    game_status: "FINAL",
    away_score: 24,
    home_score: 21,
    ...overrides,
  };
}

describe("misc grading helpers", () => {
  it("labels the game by status", () => {
    expect(gameLabel(row({}))).toBe("KC @ BUF 24–21 · final");
    expect(gameLabel(row({ game_status: "SCHEDULED", away_score: null, home_score: null }))).toBe(
      "KC @ BUF · not started",
    );
    expect(gameLabel(row({ game_status: "IN_PROGRESS" }))).toBe("KC @ BUF 24–21 · in progress");
  });

  it("counts graded and ungraded picks", () => {
    expect(gradeCounts([row({}), row({ result: "WIN" }), row({ result: "LOSS" })])).toEqual({
      graded: 2,
      ungraded: 1,
    });
  });
});

describe("MiscGradingPanel", () => {
  // globals:false → no auto-cleanup; unmount between tests.
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  beforeEach(() => {
    vi.spyOn(admin, "getWeekMiscPicks").mockResolvedValue({
      season: 2026,
      week: 6,
      picks: [row({}), row({ user_id: 2, display_name: "bo", result: "WIN", points: 3 })],
    });
  });

  it("opens on the server's default week and shows the summary", async () => {
    render(<MiscGradingPanel />);
    const summary = await screen.findByTestId("misc-grading-summary");
    expect(admin.getWeekMiscPicks).toHaveBeenCalledWith();
    expect(summary.textContent).toContain("Season 2026 · Week 6: 2 picks");
    expect(summary.textContent).toContain("1 graded · 1 ungraded");
    expect((screen.getByLabelText("Week") as HTMLSelectElement).value).toBe("6");
  });

  it("loads the chosen week", async () => {
    render(<MiscGradingPanel />);
    await screen.findByTestId("misc-grading-summary");
    fireEvent.change(screen.getByLabelText("Week"), { target: { value: "4" } });
    fireEvent.click(screen.getByText("Load misc picks"));
    await waitFor(() =>
      expect(admin.getWeekMiscPicks).toHaveBeenLastCalledWith({ season: 2026, week: 4 }),
    );
  });

  it("saves a grade for one row and updates the summary", async () => {
    const grade = vi.spyOn(admin, "gradeMisc").mockResolvedValue({
      id: 99,
      game_id: 10,
      week_id: 6,
      pick_type: "MISC",
      is_mortal_lock: false,
      result: "LOSS",
      points: -1,
    });
    render(<MiscGradingPanel />);
    await screen.findByTestId("misc-grading-summary");

    expect(screen.getAllByText("Save grade")[0]).toHaveProperty("disabled", true);

    fireEvent.click(screen.getAllByText("Incorrect")[0]);
    fireEvent.change(screen.getAllByLabelText("Points")[0], { target: { value: "-1" } });
    fireEvent.click(screen.getAllByText("Save grade")[0]);

    await waitFor(() =>
      expect(grade).toHaveBeenCalledWith(1, 2026, 6, { result: "LOSS", points: -1 }),
    );
    await waitFor(() =>
      expect(screen.getByTestId("misc-grading-summary").textContent).toContain(
        "2 graded · 0 ungraded",
      ),
    );
  });

  it("hides graded rows on request", async () => {
    render(<MiscGradingPanel />);
    await screen.findByTestId("misc-grading-summary");
    expect(screen.getAllByTestId("misc-grade-row")).toHaveLength(2);
    fireEvent.click(screen.getByLabelText("Hide graded"));
    expect(screen.getAllByTestId("misc-grade-row")).toHaveLength(1);
  });
});
