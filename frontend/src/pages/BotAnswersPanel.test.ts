import { describe, expect, it } from "vitest";
import { filterAnswers, type BotAnswer } from "../lib/admin";

function answer(overrides: Partial<BotAnswer>): BotAnswer {
  return {
    at: "2026-09-24T12:00:00+00:00",
    conversation: "1",
    asker: "ada",
    question: "q",
    intent: "open_nfl",
    path: "open",
    tools: [],
    rounds: 0,
    fallback: null,
    latency_ms: 1000,
    vendor: "openai",
    model: "m",
    answer: "a",
    ...overrides,
  };
}

const rows = [
  answer({ tools: [{ name: "lookup_qbr", args: {}, outcome: "ok" }] }),
  answer({ intent: "scores", path: "grounded" }),
  answer({ fallback: "open_degrade" }),
];

describe("filterAnswers", () => {
  it("keeps every row with no filter", () => {
    expect(filterAnswers(rows, "", "", "all")).toHaveLength(3);
  });

  it("filters by tool, intent and fallback together", () => {
    expect(filterAnswers(rows, "lookup_qbr", "", "all")).toEqual([rows[0]]);
    expect(filterAnswers(rows, "", "scores", "all")).toEqual([rows[1]]);
    expect(filterAnswers(rows, "", "", "fallback")).toEqual([rows[2]]);
    expect(filterAnswers(rows, "", "open_nfl", "clean")).toEqual([rows[0]]);
  });
});
