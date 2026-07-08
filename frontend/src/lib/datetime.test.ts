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

  it("renders real Intl-derived date+time content for a valid ISO", () => {
    const out = formatLocalDateTime("2026-01-15T17:30:00Z");
    expect(out).not.toBe("Time TBD");
    // Assert on the Intl-derived parts, NOT just the hardcoded separator/commas:
    // the `·` and `,` are literals in the source template, so a test that only
    // checks for `·` would still pass if every Intl part rendered empty. A real
    // hour:minute time and a numeric day must appear — these fail if formatToParts
    // regresses to blanks. Timezone-agnostic: the ISO instant shifts by zone but a
    // time and a day are always present.
    expect(out).toMatch(/\d{1,2}:\d{2}/); // hour:minute from Intl
    expect(out).toMatch(/\b\d{1,2}\b/); // numeric day
    expect(out).toContain("·"); // standardized separator still present
  });
});
