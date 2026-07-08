import { describe, it, expect } from "vitest";

import { formatLocalDateTime } from "./datetime";

describe("formatLocalDateTime", () => {
  it("returns 'Time TBD' for null", () => {
    expect(formatLocalDateTime(null)).toBe("Time TBD");
  });

  it("returns 'Time TBD' for an empty string", () => {
    expect(formatLocalDateTime("")).toBe("Time TBD");
  });

  it("returns 'Time TBD' for an unparseable input", () => {
    expect(formatLocalDateTime("not-a-real-date")).toBe("Time TBD");
  });

  it("renders a non-empty local string with the middot separator for a valid ISO", () => {
    const out = formatLocalDateTime("2026-01-15T17:30:00Z");
    expect(out).not.toBe("Time TBD");
    expect(out.length).toBeGreaterThan(0);
    expect(out).toContain("·");
  });
});
