/**
 * Bot transcript panel (issues #248 item 15, #252): the chat channel as the bot saw it.
 * Each member message shows whether the bot answered and why not; an answer shows the
 * intent, the tools and their outcomes, any fallback and the time it took. The export
 * downloads the stored entries as JSON Lines for a time window.
 */
import { useEffect, useMemo, useState } from "react";
import { ApiError } from "../lib/api";
import {
  filterEntries,
  listBotTranscript,
  transcriptExportUrl,
  type BotTranscriptEntry,
  type EntryView,
  type FallbackFilter,
} from "../lib/admin";
import { formatLocalDateTime } from "../lib/datetime";

type LoadStatus = "loading" | "ok" | "error";

const ALL = "";

function messageFor(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  return "Something went wrong. Please try again.";
}

function formatArgs(args: Record<string, unknown>): string {
  const parts = Object.entries(args).map(([k, v]) => `${k}=${String(v)}`);
  return parts.length ? `(${parts.join(", ")})` : "()";
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="text-sm">
      <span className="block text-xs font-medium text-fg-muted">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 block w-44 rounded-md border border-border px-2 py-1.5 text-sm"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function Entry({ e }: { e: BotTranscriptEntry }) {
  const isPost = e.kind === "bot_post" || e.kind === "other_bot";
  const answered = e.decision === "answered";
  return (
    <li className="space-y-1 py-3 text-sm">
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-fg-muted">
        <span>{formatLocalDateTime(e.at)}</span>
        <span>{e.author ?? "unknown"}</span>
        {e.channel && <span>#{e.channel}</span>}
        {isPost ? (
          <span>{e.kind === "bot_post" ? "bot post" : "other bot"}</span>
        ) : (
          <span className={answered ? "" : "font-semibold"}>
            {e.decision ?? "no decision"}
            {e.addressed_by ? ` (${e.addressed_by})` : ""}
          </span>
        )}
        {answered && (
          <>
            <span>
              {e.intent ?? "no intent"} · {e.path}
            </span>
            <span>
              {e.rounds} round{e.rounds === 1 ? "" : "s"}
            </span>
            {e.latency_ms !== null && <span>{(e.latency_ms / 1000).toFixed(1)} s</span>}
            <span>{e.model ?? e.vendor}</span>
          </>
        )}
        {e.fallback && <span className="font-semibold text-danger-fg">{e.fallback}</span>}
      </div>
      <p className={isPost ? "text-fg-muted" : "font-medium"}>
        {isPost ? e.content : e.question}
      </p>
      {e.tools.length > 0 && (
        <ul className="space-y-0.5 font-mono text-xs">
          {e.tools.map((t, j) => (
            <li key={j} className="break-all">
              {t.name}
              {formatArgs(t.args)}
              {t.outcome !== "ok" && <span className="text-danger-fg"> {t.outcome}</span>}
              {t.note && <span className="text-fg-muted"> — {t.note}</span>}
            </li>
          ))}
        </ul>
      )}
      {e.answer && <p className="whitespace-pre-wrap text-fg-muted">{e.answer}</p>}
    </li>
  );
}

export default function BotTranscriptPanel() {
  const [entries, setEntries] = useState<BotTranscriptEntry[]>([]);
  const [available, setAvailable] = useState(true);
  const [status, setStatus] = useState<LoadStatus>("loading");
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<EntryView>("all");
  const [tool, setTool] = useState(ALL);
  const [intent, setIntent] = useState(ALL);
  const [fallback, setFallback] = useState<FallbackFilter>("all");
  const [since, setSince] = useState("");
  const [until, setUntil] = useState("");

  useEffect(() => {
    let cancelled = false;
    listBotTranscript()
      .then((r) => {
        if (cancelled) return;
        setEntries(r.entries);
        setAvailable(r.available);
        setStatus("ok");
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(messageFor(err));
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const tools = useMemo(
    () => [...new Set(entries.flatMap((e) => e.tools.map((t) => t.name)))].sort(),
    [entries],
  );
  const intents = useMemo(
    () => [...new Set(entries.map((e) => e.intent).filter((i): i is string => !!i))].sort(),
    [entries],
  );
  const shown = useMemo(
    () => filterEntries(entries, view, tool, intent, fallback),
    [entries, view, tool, intent, fallback],
  );

  return (
    <section
      data-testid="bot-transcript-panel"
      className="space-y-3 rounded-lg border border-border bg-surface p-4"
    >
      <div>
        <h2 className="text-lg font-bold">Bot Transcript</h2>
        <p className="mt-0.5 text-sm text-fg-muted">
          The chat channel as the bot saw it, newest first: every message, whether the bot
          answered and why not, and for each answer the intent, the tools and the time it
          took.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <label className="text-sm">
          <span className="block text-xs font-medium text-fg-muted">From</span>
          <input
            type="datetime-local"
            value={since}
            onChange={(e) => setSince(e.target.value)}
            className="mt-1 block rounded-md border border-border px-2 py-1 text-sm"
          />
        </label>
        <label className="text-sm">
          <span className="block text-xs font-medium text-fg-muted">To</span>
          <input
            type="datetime-local"
            value={until}
            onChange={(e) => setUntil(e.target.value)}
            className="mt-1 block rounded-md border border-border px-2 py-1 text-sm"
          />
        </label>
        <a
          href={transcriptExportUrl(since, until)}
          download
          data-testid="bot-transcript-export"
          className="rounded-md border border-border px-3 py-1.5 text-sm font-medium hover:bg-surface-raised"
        >
          {since || until ? "Download range" : "Download all"} (.jsonl)
        </a>
      </div>

      {status === "loading" ? (
        <p className="text-sm text-fg-muted">Loading…</p>
      ) : status === "error" ? (
        <p className="text-sm text-fg-muted">{error}</p>
      ) : !available ? (
        <p className="text-sm text-fg-muted">
          The transcript store is not reachable right now. Please try again later.
        </p>
      ) : (
        <>
          <div className="flex flex-wrap items-end gap-3 border-t border-border pt-3">
            <SelectField
              label="Show"
              value={view}
              onChange={(v) => setView(v as EntryView)}
              options={[
                { value: "all", label: "Everything" },
                { value: "answered", label: "Answered" },
                { value: "skipped", label: "Not answered" },
                { value: "posts", label: "Bot posts" },
              ]}
            />
            <SelectField
              label="Tool"
              value={tool}
              onChange={setTool}
              options={[
                { value: ALL, label: "Any tool" },
                ...tools.map((t) => ({ value: t, label: t })),
              ]}
            />
            <SelectField
              label="Intent"
              value={intent}
              onChange={setIntent}
              options={[
                { value: ALL, label: "Any intent" },
                ...intents.map((i) => ({ value: i, label: i })),
              ]}
            />
            <SelectField
              label="Fallback"
              value={fallback}
              onChange={(v) => setFallback(v as FallbackFilter)}
              options={[
                { value: "all", label: "All" },
                { value: "fallback", label: "Fallback only" },
                { value: "clean", label: "No fallback" },
              ]}
            />
            <span className="text-xs text-fg-muted">
              {shown.length} of {entries.length}
            </span>
          </div>

          {shown.length === 0 ? (
            <p className="text-sm text-fg-muted">No entries match.</p>
          ) : (
            <ul className="divide-y divide-border">
              {shown.map((e, i) => (
                <Entry key={`${e.at}-${i}`} e={e} />
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  );
}
