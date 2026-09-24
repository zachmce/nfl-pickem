import { describe, expect, it } from "vitest";
import { filterEntries, transcriptExportUrl, type BotTranscriptEntry } from "../lib/admin";

function entry(overrides: Partial<BotTranscriptEntry>): BotTranscriptEntry {
  return {
    at: "2026-09-24T12:00:00+00:00",
    kind: "member",
    channel: "chat",
    author: "ada",
    addressed_by: "mention",
    decision: "answered",
    question: "q",
    content: null,
    intent: "open_nfl",
    classifier: null,
    path: "open",
    tools: [],
    rounds: 0,
    fallback: null,
    history_turns: 0,
    latency_ms: 1000,
    vendor: "openai",
    model: "m",
    answer: "a",
    ...overrides,
  };
}

const rows = [
  entry({ tools: [{ name: "lookup_qbr", args: {}, outcome: "ok" }] }),
  entry({ intent: "scores", path: "grounded" }),
  entry({ fallback: "open_degrade" }),
  entry({ decision: "not_addressed:room_gate_said_no", intent: null, answer: null }),
  entry({ kind: "bot_post", decision: null, content: "Week 3 locked", question: null }),
];

describe("filterEntries", () => {
  it("keeps every row with no filter", () => {
    expect(filterEntries(rows, "all", "", "", "all")).toHaveLength(5);
  });

  it("splits answered, skipped and bot posts", () => {
    expect(filterEntries(rows, "answered", "", "", "all")).toHaveLength(3);
    expect(filterEntries(rows, "skipped", "", "", "all")).toEqual([rows[3]]);
    expect(filterEntries(rows, "posts", "", "", "all")).toEqual([rows[4]]);
  });

  it("filters by tool, intent and fallback together", () => {
    expect(filterEntries(rows, "all", "lookup_qbr", "", "all")).toEqual([rows[0]]);
    expect(filterEntries(rows, "all", "", "scores", "all")).toEqual([rows[1]]);
    expect(filterEntries(rows, "all", "", "", "fallback")).toEqual([rows[2]]);
  });
});

describe("transcriptExportUrl", () => {
  it("is the whole store with no bounds", () => {
    expect(transcriptExportUrl("", "")).toBe("/api/admin/bot-transcript/export");
  });

  it("sends each bound as a UTC instant", () => {
    const url = transcriptExportUrl("2026-09-20T08:00", "");
    const since = new URL(url, "http://x").searchParams.get("since");
    expect(since).toBe(new Date("2026-09-20T08:00").toISOString());
    expect(url).not.toContain("until");
  });
});
