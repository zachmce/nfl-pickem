/**
 * Bot answers panel (issue #248, item 15): the last answers the Discord bot gave,
 * with the intent, the tools it called, the rounds, any fallback and the latency.
 * Read-only; the filters run on the client over the at most 200 stored answers.
 */
import { useEffect, useMemo, useState } from "react";
import { ApiError } from "../lib/api";
import {
  filterAnswers,
  listBotAnswers,
  type BotAnswer,
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
        className="mt-1 block w-52 rounded-md border border-border px-2 py-1.5 text-sm"
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

export default function BotAnswersPanel() {
  const [answers, setAnswers] = useState<BotAnswer[]>([]);
  const [available, setAvailable] = useState(true);
  const [status, setStatus] = useState<LoadStatus>("loading");
  const [error, setError] = useState<string | null>(null);
  const [tool, setTool] = useState(ALL);
  const [intent, setIntent] = useState(ALL);
  const [fallback, setFallback] = useState<FallbackFilter>("all");

  useEffect(() => {
    let cancelled = false;
    listBotAnswers()
      .then((r) => {
        if (cancelled) return;
        setAnswers(r.answers);
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
    () => [...new Set(answers.flatMap((a) => a.tools.map((t) => t.name)))].sort(),
    [answers],
  );
  const intents = useMemo(
    () => [...new Set(answers.map((a) => a.intent).filter((i): i is string => !!i))].sort(),
    [answers],
  );
  const shown = useMemo(
    () => filterAnswers(answers, tool, intent, fallback),
    [answers, tool, intent, fallback],
  );

  return (
    <section
      data-testid="bot-answers-panel"
      className="space-y-3 rounded-lg border border-border bg-surface p-4"
    >
      <div>
        <h2 className="text-lg font-bold">Bot Answers</h2>
        <p className="mt-0.5 text-sm text-fg-muted">
          The last answers the Discord bot gave, newest first: the intent, the tools it
          called, the rounds, any fallback and the time it took.
        </p>
      </div>

      {status === "loading" ? (
        <p className="text-sm text-fg-muted">Loading…</p>
      ) : status === "error" ? (
        <p className="text-sm text-fg-muted">{error}</p>
      ) : !available ? (
        <p className="text-sm text-fg-muted">
          The answer store is not reachable right now. Please try again later.
        </p>
      ) : (
        <>
          <div className="flex flex-wrap items-end gap-3 border-t border-border pt-3">
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
                { value: "all", label: "All answers" },
                { value: "fallback", label: "Fallback only" },
                { value: "clean", label: "No fallback" },
              ]}
            />
            <span className="text-xs text-fg-muted">
              {shown.length} of {answers.length}
            </span>
          </div>

          {shown.length === 0 ? (
            <p className="text-sm text-fg-muted">No answers match.</p>
          ) : (
            <ul className="divide-y divide-border">
              {shown.map((a, i) => (
                <li key={`${a.at}-${i}`} className="space-y-1 py-3 text-sm">
                  <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-fg-muted">
                    <span>{formatLocalDateTime(a.at)}</span>
                    <span>{a.asker ?? "unknown asker"}</span>
                    <span>
                      {a.intent ?? "no intent"} · {a.path}
                    </span>
                    <span>
                      {a.rounds} round{a.rounds === 1 ? "" : "s"}
                    </span>
                    {a.latency_ms !== null && (
                      <span>{(a.latency_ms / 1000).toFixed(1)} s</span>
                    )}
                    <span>{a.model ?? a.vendor}</span>
                    {a.fallback && (
                      <span className="font-semibold text-danger-fg">{a.fallback}</span>
                    )}
                  </div>
                  <p className="font-medium">{a.question}</p>
                  {a.tools.length > 0 && (
                    <ul className="space-y-0.5 font-mono text-xs">
                      {a.tools.map((t, j) => (
                        <li key={j} className="break-all">
                          {t.name}
                          {formatArgs(t.args)}
                          {t.outcome !== "ok" && (
                            <span className="text-danger-fg"> {t.outcome}</span>
                          )}
                        </li>
                      ))}
                    </ul>
                  )}
                  <p className="whitespace-pre-wrap text-fg-muted">{a.answer}</p>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  );
}
