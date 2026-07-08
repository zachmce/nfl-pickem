import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

// Hoisted module mocks — useWeekly calls THROUGH these lib wrappers, not fetch.
vi.mock("../lib/currentWeek");
vi.mock("../lib/results");
vi.mock("../lib/picks");

import { useWeekly } from "./useWeekly";
import { getCurrentWeek } from "../lib/currentWeek";
import { getWeekResults } from "../lib/results";
import { getSlate } from "../lib/picks";

describe("useWeekly", () => {
  beforeEach(() => vi.resetAllMocks());

  it("settles to 'ok' with week/maxWeek from getCurrentWeek (happy path)", async () => {
    vi.mocked(getCurrentWeek).mockResolvedValue({
      season: 2026,
      week: 3,
    } as never);
    vi.mocked(getWeekResults).mockResolvedValue({ results: [] } as never);
    vi.mocked(getSlate).mockResolvedValue({ games: [] } as never);

    const { result } = renderHook(() => useWeekly());
    expect(result.current.status).toBe("loading"); // initial

    await waitFor(() => expect(result.current.status).toBe("ok"));
    expect(result.current.season).toBe(2026);
    expect(result.current.week).toBe(3);
    expect(result.current.maxWeek).toBe(3);
  });

  it("settles to 'error' when getCurrentWeek rejects (error path)", async () => {
    vi.mocked(getCurrentWeek).mockRejectedValue(new Error("boom"));

    const { result } = renderHook(() => useWeekly());
    await waitFor(() => expect(result.current.status).toBe("error"));
  });
});
