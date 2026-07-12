import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";

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

  // Behavior-parity guard for the set-state-in-effect refactor (issue #112):
  // a week change MUST re-show the "loading" placeholder while the new week's
  // parallel fetch is in flight — the same UX the in-effect setStatus gave.
  it("re-enters 'loading' when the week changes (re-fetch placeholder)", async () => {
    vi.mocked(getCurrentWeek).mockResolvedValue({
      season: 2026,
      week: 3,
    } as never);
    vi.mocked(getWeekResults).mockResolvedValue({ results: [] } as never);
    vi.mocked(getSlate).mockResolvedValue({ games: [] } as never);

    const { result } = renderHook(() => useWeekly());
    await waitFor(() => expect(result.current.status).toBe("ok"));

    // Make the previous week's results hang so the loading transition is
    // observable deterministically (no race on a resolved microtask).
    let release!: () => void;
    const pending = new Promise((resolve) => {
      release = () => resolve({ results: [] });
    });
    vi.mocked(getWeekResults).mockReturnValueOnce(pending as never);

    act(() => result.current.prev());
    expect(result.current.week).toBe(2);
    expect(result.current.status).toBe("loading");

    await act(async () => {
      release();
      await pending;
    });
    await waitFor(() => expect(result.current.status).toBe("ok"));
  });
});
