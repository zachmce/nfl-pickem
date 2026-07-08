import { describe, it, expect } from "vitest";

import {
  LOADING_WEEKLY,
  LOADING_MY_PICKS,
  LOADING_STANDINGS,
  ERROR_WEEKLY,
  ERROR_WEEK_STATUS,
  EMPTY_WEEKLY,
  EMPTY_STANDINGS,
} from "./strings";

// These are the LOCKED loading/error/empty copy constants (site-consistency
// decision #6). Asserting a representative set guards the exact wording from
// drift — a copy change here must be a deliberate, reviewed edit.
describe("page-state copy constants", () => {
  it("has the locked loading copy", () => {
    expect(LOADING_WEEKLY).toBe("Loading this week's picks…");
    expect(LOADING_MY_PICKS).toBe("Loading your picks…");
    expect(LOADING_STANDINGS).toBe("Loading the season scoreboard…");
  });

  it("has the locked error copy", () => {
    expect(ERROR_WEEKLY).toBe(
      "Couldn't load the weekly results. Please try again later.",
    );
    expect(ERROR_WEEK_STATUS).toBe("Week status unavailable");
  });

  it("has the locked empty copy", () => {
    expect(EMPTY_WEEKLY).toBe("No picks have been made for this week yet.");
    expect(EMPTY_STANDINGS).toBe(
      "No scores have been posted yet — the scoreboard will fill in once the season is underway.",
    );
  });
});
