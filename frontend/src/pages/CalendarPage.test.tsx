import { describe, it, expect, afterEach, vi } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";

import type { CalendarGame, CalendarPickOutcome } from "../lib/calendar";
import CalendarPage from "./CalendarPage";

const { getCalendar } = vi.hoisted(() => ({ getCalendar: vi.fn() }));
vi.mock("../lib/calendar", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/calendar")>()),
  getCalendar,
}));

function game(
  id: number,
  away: string,
  status: CalendarGame["status"],
  my_pick_result: CalendarPickOutcome | null,
): CalendarGame {
  const final = status === "FINAL";
  return {
    game_id: id,
    // Mid-month noon UTC: inside the visible grid of the current month.
    kickoff_at: new Date(
      Date.UTC(new Date().getFullYear(), new Date().getMonth(), 15, 12),
    ).toISOString(),
    home_team: { abbreviation: "HOM" },
    away_team: { abbreviation: away },
    status,
    home_score: final ? 20 : null,
    away_score: final ? 10 : null,
    my_pick_result,
  };
}

describe("CalendarPage pick highlights (issue #301)", () => {
  afterEach(() => cleanup());

  it("bolds a picked upcoming game, greens a win, reds a loss", async () => {
    getCalendar.mockResolvedValue({
      from_date: "",
      to_date: "",
      games: [
        game(1, "NOP", "SCHEDULED", null),
        game(2, "PEN", "SCHEDULED", "PENDING"),
        game(3, "WIN", "FINAL", "WIN"),
        game(4, "LOS", "FINAL", "LOSS"),
      ],
    });
    render(<CalendarPage />);

    const plain = await screen.findByText("NOP @ HOM");
    expect(plain.className).toContain("font-medium");
    expect(plain.className).not.toContain("font-bold");

    const pending = screen.getByText("PEN @ HOM");
    expect(pending.className).toContain("font-bold");
    expect(pending.className).toContain("text-fg");

    const won = screen.getByText("WIN 10 @ HOM 20");
    expect(won.className).toContain("text-success-fg");
    expect(won.parentElement?.className).toContain("bg-success-bg");

    const lost = screen.getByText("LOS 10 @ HOM 20");
    expect(lost.className).toContain("text-danger-fg");
    expect(lost.parentElement?.className).toContain("bg-danger-bg");
  });
});
