/**
 * Admin page — the web counterpart to the Discord `/admin` cog (QT-D).
 *
 * Renders as CONTENT ONLY inside AppShell behind RequireAdmin (no shell, header,
 * nav, or auth guard here — mirrors MyPicksPage.tsx). The seeded web admin lands
 * here (/admin) and sees one row per user with their identity, origin, state, and
 * pick count, plus per-row controls to deactivate/reactivate, grant/revoke admin,
 * and delete any OTHER user.
 *
 * Server truth is authoritative: every mutation returns the updated AdminUser
 * (or 204 for delete), which is merged back into the table by id (or removed) —
 * never an optimistic local flip. The current admin's own active-toggle,
 * admin-toggle, and delete controls are disabled to mirror the server self-guards
 * (cannot_act_on_self / last_admin), and a per-row in-flight guard prevents
 * double-submit. Rejected 4xx mutations surface their envelope message inline
 * (already unwrapped by api() into ApiError.message) without corrupting the table.
 */
import { useCallback, useEffect, useState } from "react";

import { ApiError } from "../lib/api";
import {
  deactivateUser,
  deleteUser,
  freezeWeek,
  getBotPersonality,
  grantAdmin,
  ingestSeason,
  listUsers,
  reactivateUser,
  revokeAdmin,
  setBotPersonality,
  type AdminPickSet,
  type AdminUser,
  type BotPersonality,
  type FreezeWeekDispatch,
  type IngestSeasonDispatch,
} from "../lib/admin";
import { getCurrentWeek } from "../lib/currentWeek";
import {
  errorKey,
  slotKey,
  type PickResult,
  type PickType,
  type SlateGame,
} from "../lib/picks";
import type { AdminMiscGrade } from "../lib/admin";
import { useAuth } from "../auth/useAuth";
import { useAdminPickEditor } from "./useAdminPickEditor";
import type { PicksBySlot } from "./useMyPicks";

type LoadStatus = "loading" | "ok" | "error";

const PICK_TYPE_LABEL: Record<PickType, string> = {
  UNDERDOG_COVER: "Underdog",
  FAVORITE_COVER: "Favorite",
  OVER: "Over",
  UNDER: "Under",
  // MISC is its own non-base type (the MISC sub-panel renders it); it never
  // appears as a per-game base button (the base-button loops use BASE_SLOTS,
  // which excludes MISC). The label exists only to satisfy the widened Record.
  MISC: "Misc",
};

/** Order the base slots appear in the roster tracker / game cards. MISC is
 * intentionally EXCLUDED — it is not a base bet (rendered by its own sub-panel). */
const BASE_SLOTS: PickType[] = [
  "UNDERDOG_COVER",
  "FAVORITE_COVER",
  "OVER",
  "UNDER",
];

/** The user currently targeted by the pick-override editor panel. */
interface EditorTarget {
  id: number;
  display_name: string;
  season: number;
  week: number;
}

/** Format an ISO created_at; tolerate null/invalid like MyPicksPage's friendlyKickoff. */
function friendlyDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/** Pull a human message out of a rejected mutation (ApiError carries the envelope). */
function messageFor(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  return "Something went wrong. Please try again.";
}

export default function AdminPage() {
  const { user } = useAuth();
  const [status, setStatus] = useState<LoadStatus>("loading");
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [pageError, setPageError] = useState<string | null>(null);
  const [pending, setPending] = useState<Record<number, boolean>>({});
  const [rowError, setRowError] = useState<Record<number, string>>({});
  // The user whose picks the override editor is currently editing (null = closed).
  const [editorTarget, setEditorTarget] = useState<EditorTarget | null>(null);
  // Sensible season/week to seed the editor on first open — getCurrentWeek's
  // values, which the admin can then change to any past/future week.
  const [seed, setSeed] = useState<{ season: number; week: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    listUsers()
      .then((rows) => {
        if (cancelled) return;
        const sorted = [...rows].sort((a, b) => a.id - b.id);
        setUsers(sorted);
        setStatus("ok");
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setPageError(messageFor(err));
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Seed the editor's default season/week from the current week (best-effort —
  // the editor falls back to season 0 / week 1 if this never resolves, and the
  // admin can always type the week they want).
  useEffect(() => {
    let cancelled = false;
    getCurrentWeek()
      .then((cw) => {
        if (cancelled) return;
        setSeed({ season: cw.season, week: cw.week });
      })
      .catch(() => {
        /* non-fatal: the editor still opens with a typed/default week. */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  /** Open the pick-override editor for a user (allowed for ANY user incl. self). */
  const openEditor = useCallback(
    (row: AdminUser) => {
      setEditorTarget({
        id: row.id,
        display_name: row.display_name,
        season: seed?.season ?? new Date().getFullYear(),
        week: seed?.week ?? 1,
      });
    },
    [seed],
  );

  const closeEditor = useCallback(() => setEditorTarget(null), []);

  /** Replace a row in the table by id with the server-returned truth. */
  const mergeUser = useCallback((updated: AdminUser) => {
    setUsers((prev) =>
      prev.map((u) => (u.id === updated.id ? updated : u)),
    );
  }, []);

  /** Drop a row from the table (after a successful delete). */
  const dropUser = useCallback((id: number) => {
    setUsers((prev) => prev.filter((u) => u.id !== id));
  }, []);

  /**
   * Run a per-row mutation behind the in-flight + inline-error guard. Ignores
   * the click if the row is already mutating; clears that row's prior error;
   * applies `onSuccess` to the result; clears pending in finally.
   */
  const runRowAction = useCallback(
    async <T,>(id: number, action: () => Promise<T>, onSuccess: (result: T) => void) => {
      let alreadyPending = false;
      setPending((prev) => {
        if (prev[id]) {
          alreadyPending = true;
          return prev;
        }
        return { ...prev, [id]: true };
      });
      if (alreadyPending) return;

      setRowError((prev) => {
        if (!(id in prev)) return prev;
        const next = { ...prev };
        delete next[id];
        return next;
      });

      try {
        const result = await action();
        onSuccess(result);
      } catch (err: unknown) {
        setRowError((prev) => ({ ...prev, [id]: messageFor(err) }));
      } finally {
        setPending((prev) => {
          const next = { ...prev };
          delete next[id];
          return next;
        });
      }
    },
    [],
  );

  const onToggleActive = useCallback(
    (row: AdminUser) => {
      const action = row.is_active
        ? () => deactivateUser(row.id)
        : () => reactivateUser(row.id);
      void runRowAction(row.id, action, mergeUser);
    },
    [runRowAction, mergeUser],
  );

  const onToggleAdmin = useCallback(
    (row: AdminUser) => {
      const action = row.is_admin
        ? () => revokeAdmin(row.id)
        : () => grantAdmin(row.id);
      void runRowAction(row.id, action, mergeUser);
    },
    [runRowAction, mergeUser],
  );

  const onDelete = useCallback(
    (row: AdminUser) => {
      if (!window.confirm(`Delete ${row.display_name}? This removes their picks too.`)) {
        return;
      }
      void runRowAction(row.id, () => deleteUser(row.id), () => dropUser(row.id));
    },
    [runRowAction, dropUser],
  );

  if (status === "loading") {
    return (
      <div>
        <h1 className="text-2xl font-bold">Admin</h1>
        <p className="mt-2 text-gray-500">Loading users…</p>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div>
        <h1 className="text-2xl font-bold">Admin</h1>
        <p className="mt-2 text-gray-600">
          {pageError ?? "Couldn't load users. Please try again later."}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold">Player management</h1>
        <p className="mt-1 text-sm text-gray-500">
          {users.length} {users.length === 1 ? "user" : "users"} · deactivate,
          grant/revoke admin, or delete any other player.
        </p>
      </header>

      {users.length === 0 ? (
        <p className="text-gray-500">No users yet.</p>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 text-left text-xs font-semibold uppercase tracking-wide text-gray-500">
                <th className="px-4 py-3">Player</th>
                <th className="px-4 py-3">Origin</th>
                <th className="px-4 py-3">Active</th>
                <th className="px-4 py-3">Admin</th>
                <th className="px-4 py-3">Created</th>
                <th className="px-4 py-3 text-right">Picks</th>
                <th className="px-4 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map((row) => (
                <UserRow
                  key={row.id}
                  row={row}
                  isSelf={row.id === user?.id}
                  busy={Boolean(pending[row.id])}
                  error={rowError[row.id]}
                  onToggleActive={onToggleActive}
                  onToggleAdmin={onToggleAdmin}
                  onDelete={onDelete}
                  onEditPicks={openEditor}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      <IngestionPanel />

      <BotPersonalityPanel />

      {editorTarget && (
        <PickOverrideEditor
          key={editorTarget.id}
          target={editorTarget}
          onClose={closeEditor}
        />
      )}
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Ingestion / Odds panel (QT-3)
//
// The frontend half of the live-ingest-worker thread: two admin-only controls
// that DISPATCH a Celery background task via the frozen QT-2 endpoints and get
// back HTTP 202 + a task_id. The 202 means accepted/queued — NOT complete (there
// is no status/progress endpoint), so this panel renders the returned task_id as
// a background-dispatch confirmation and deliberately avoids any done-state
// phrasing. Each action owns its own in-flight guard so one button disables
// independently; a rejected 4xx/403 surfaces messageFor(err) inline (already
// unwrapped from the envelope by api()) without unmounting the section. No new
// auth guard here — the whole page is RequireAdmin-gated, and the server enforces
// require_admin on both endpoints.
// --------------------------------------------------------------------------- //

/** The blue primary-button styling shared by the two dispatch controls. */
function primaryButtonClass(disabled: boolean): string {
  return [
    "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
    "border-blue-600 bg-blue-600 text-white hover:bg-blue-700",
    disabled ? "cursor-not-allowed opacity-50" : "",
  ].join(" ");
}

/** A labeled number input mirroring PickOverrideEditor's Season/Week inputs. */
function NumberField({
  label,
  value,
  width,
  onChange,
}: {
  label: string;
  value: number;
  width: string;
  onChange: (n: number) => void;
}) {
  return (
    <label className="text-sm">
      <span className="block text-xs font-medium text-gray-600">{label}</span>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={`mt-1 ${width} rounded-md border border-gray-300 px-2 py-1 text-sm`}
      />
    </label>
  );
}

function IngestionPanel() {
  // Sensible seeds — current calendar year, week 1 (the demo/live admin can
  // retype either). No getCurrentWeek dependency: keep this panel self-contained.
  const [ingestSeasonValue, setIngestSeasonValue] = useState<number>(
    new Date().getFullYear(),
  );
  const [freezeSeasonValue, setFreezeSeasonValue] = useState<number>(
    new Date().getFullYear(),
  );
  const [freezeWeekValue, setFreezeWeekValue] = useState<number>(1);

  // Per-action in-flight + result + error so each control disables/reports
  // independently. The result holds the dispatched 202 body (task_id rendered).
  const [ingestBusy, setIngestBusy] = useState(false);
  const [ingestResult, setIngestResult] = useState<IngestSeasonDispatch | null>(
    null,
  );
  const [ingestError, setIngestError] = useState<string | null>(null);

  const [freezeBusy, setFreezeBusy] = useState(false);
  const [freezeResult, setFreezeResult] = useState<FreezeWeekDispatch | null>(
    null,
  );
  const [freezeError, setFreezeError] = useState<string | null>(null);

  const onIngest = useCallback(async () => {
    setIngestBusy(true);
    setIngestError(null);
    setIngestResult(null);
    try {
      const dispatch = await ingestSeason(ingestSeasonValue);
      setIngestResult(dispatch);
    } catch (err: unknown) {
      setIngestError(messageFor(err));
    } finally {
      setIngestBusy(false);
    }
  }, [ingestSeasonValue]);

  const onFreeze = useCallback(async () => {
    setFreezeBusy(true);
    setFreezeError(null);
    setFreezeResult(null);
    try {
      const dispatch = await freezeWeek(freezeSeasonValue, freezeWeekValue);
      setFreezeResult(dispatch);
    } catch (err: unknown) {
      setFreezeError(messageFor(err));
    } finally {
      setFreezeBusy(false);
    }
  }, [freezeSeasonValue, freezeWeekValue]);

  const ingestDisabled = ingestBusy || !Number.isFinite(ingestSeasonValue);
  const freezeDisabled =
    freezeBusy ||
    !Number.isFinite(freezeSeasonValue) ||
    !Number.isFinite(freezeWeekValue);

  return (
    <section
      data-testid="ingestion-panel"
      className="space-y-4 rounded-lg border border-gray-200 bg-white p-4"
    >
      <div>
        <h2 className="text-lg font-bold">Ingestion / Odds</h2>
        <p className="mt-0.5 text-sm text-gray-500">
          These controls dispatch background workers. Each returns a task id that
          runs asynchronously — the work is queued, not finished when you click.
        </p>
      </div>

      {/* Ingest season */}
      <div className="space-y-2 border-t border-gray-100 pt-4">
        <div className="flex flex-wrap items-end gap-3">
          <NumberField
            label="Season"
            value={ingestSeasonValue}
            width="w-28"
            onChange={setIngestSeasonValue}
          />
          <button
            type="button"
            disabled={ingestDisabled}
            onClick={() => void onIngest()}
            className={primaryButtonClass(ingestDisabled)}
          >
            Ingest season now
          </button>
          {ingestBusy && (
            <span className="text-xs text-gray-400">Dispatching…</span>
          )}
        </div>
        {ingestResult && (
          <p className="text-xs text-gray-600">
            Dispatched — task{" "}
            <code className="rounded bg-gray-100 px-1 py-0.5">
              {ingestResult.task_id}
            </code>{" "}
            is running in the background for season {ingestResult.season}.
          </p>
        )}
        {ingestError && <p className="text-xs text-red-600">{ingestError}</p>}
      </div>

      {/* Freeze week lines */}
      <div className="space-y-2 border-t border-gray-100 pt-4">
        <div className="flex flex-wrap items-end gap-3">
          <NumberField
            label="Season"
            value={freezeSeasonValue}
            width="w-28"
            onChange={setFreezeSeasonValue}
          />
          <NumberField
            label="Week"
            value={freezeWeekValue}
            width="w-24"
            onChange={setFreezeWeekValue}
          />
          <button
            type="button"
            disabled={freezeDisabled}
            onClick={() => void onFreeze()}
            className={primaryButtonClass(freezeDisabled)}
          >
            Freeze week lines now
          </button>
          {freezeBusy && (
            <span className="text-xs text-gray-400">Dispatching…</span>
          )}
        </div>
        {freezeResult && (
          <p className="text-xs text-gray-600">
            Dispatched — task{" "}
            <code className="rounded bg-gray-100 px-1 py-0.5">
              {freezeResult.task_id}
            </code>{" "}
            is running in the background for season {freezeResult.season}, week{" "}
            {freezeResult.week}.
          </p>
        )}
        {freezeError && <p className="text-xs text-red-600">{freezeError}</p>}
      </div>
    </section>
  );
}

// --------------------------------------------------------------------------- //
// Bot Personality panel (260627-xbb)
//
// One admin-only selector for the app-wide LLM chat voice. Loads the active id +
// available ids on mount; a <select> change POSTs the new id and re-merges the
// server-truth active_id (never an optimistic flip). Busy/error state mirrors the
// IngestionPanel's useState shape; a rejected 4xx (e.g. 409 unknown_personality)
// surfaces messageFor(err) inline. No new auth guard — the page is RequireAdmin-
// gated and the server enforces require_admin on both verbs.
// --------------------------------------------------------------------------- //

/** Humanize a personality id for the option label (e.g. stats_nerd -> "Stats Nerd"). */
function personalityLabel(id: string): string {
  return id
    .split("_")
    .map((part) => (part ? part[0].toUpperCase() + part.slice(1) : part))
    .join(" ");
}

function BotPersonalityPanel() {
  const [data, setData] = useState<BotPersonality | null>(null);
  const [status, setStatus] = useState<LoadStatus>("loading");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    getBotPersonality()
      .then((p) => {
        if (cancelled) return;
        setData(p);
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

  const onSelect = useCallback(async (id: string) => {
    setBusy(true);
    setError(null);
    try {
      const updated = await setBotPersonality(id);
      setData(updated); // server-truth merge (active_id + available_ids)
    } catch (err: unknown) {
      setError(messageFor(err));
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <section
      data-testid="bot-personality-panel"
      className="space-y-3 rounded-lg border border-gray-200 bg-white p-4"
    >
      <div>
        <h2 className="text-lg font-bold">Bot Personality</h2>
        <p className="mt-0.5 text-sm text-gray-500">
          Choose the chat bot's voice. The swap is live — it takes effect on the
          next event without a redeploy. The facts-first and leak-safety guardrails
          stay the same for every voice.
        </p>
      </div>

      {status === "loading" ? (
        <p className="text-sm text-gray-500">Loading…</p>
      ) : status === "error" ? (
        <p className="text-sm text-gray-600">
          {error ?? "Couldn't load the bot personality. Please try again later."}
        </p>
      ) : data ? (
        <div className="flex flex-wrap items-end gap-3 border-t border-gray-100 pt-3">
          <label className="text-sm">
            <span className="block text-xs font-medium text-gray-600">
              Active voice
            </span>
            <select
              value={data.active_id}
              disabled={busy}
              onChange={(e) => void onSelect(e.target.value)}
              className={[
                "mt-1 block w-64 rounded-md border border-gray-300 px-2 py-1.5 text-sm",
                busy ? "cursor-not-allowed opacity-50" : "",
              ].join(" ")}
            >
              {data.available_ids.map((id) => (
                <option key={id} value={id}>
                  {personalityLabel(id)}
                </option>
              ))}
            </select>
          </label>
          {busy && <span className="text-xs text-gray-400">Saving…</span>}
          {error && <p className="w-full text-xs text-red-600">{error}</p>}
        </div>
      ) : null}
    </section>
  );
}

/** A rounded pill badge (mirrors MyPicksPage SlotChip's bg-*-50/border-*-300/text-*-700 vocabulary). */
function Badge({
  label,
  tone,
}: {
  label: string;
  tone: "green" | "blue" | "gray";
}) {
  const tones: Record<typeof tone, string> = {
    green: "border-green-300 bg-green-50 text-green-700",
    blue: "border-blue-300 bg-blue-50 text-blue-700",
    gray: "border-gray-300 bg-gray-50 text-gray-500",
  };
  return (
    <span
      className={[
        "inline-block rounded-md border px-2 py-0.5 text-xs font-medium",
        tones[tone],
      ].join(" ")}
    >
      {label}
    </span>
  );
}

/** Per-row action button — disabled state reuses the cursor-not-allowed/opacity pattern. */
function RowButton({
  label,
  tone,
  disabled,
  onClick,
}: {
  label: string;
  tone: "neutral" | "danger";
  disabled: boolean;
  onClick: () => void;
}) {
  const tones: Record<typeof tone, string> = {
    neutral: "border-gray-300 bg-white text-gray-700 hover:border-blue-400",
    danger: "border-red-300 bg-white text-red-600 hover:border-red-500",
  };
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={[
        "rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
        tones[tone],
        disabled ? "cursor-not-allowed opacity-50" : "",
      ].join(" ")}
    >
      {label}
    </button>
  );
}

function UserRow({
  row,
  isSelf,
  busy,
  error,
  onToggleActive,
  onToggleAdmin,
  onDelete,
  onEditPicks,
}: {
  row: AdminUser;
  isSelf: boolean;
  busy: boolean;
  error: string | undefined;
  onToggleActive: (row: AdminUser) => void;
  onToggleAdmin: (row: AdminUser) => void;
  onDelete: (row: AdminUser) => void;
  onEditPicks: (row: AdminUser) => void;
}) {
  // Self-guards mirror the server: the current admin's own active-toggle,
  // admin-toggle, and delete are all disabled (cannot_act_on_self / last_admin).
  // A row mid-mutation disables its controls too (no double-submit).
  const controlsDisabled = isSelf || busy;

  return (
    <>
      <tr className="border-b border-gray-100 last:border-0">
        <td className="px-4 py-3 font-medium text-gray-800">
          {row.display_name}
          {isSelf && (
            <span className="ml-2 text-xs font-normal text-gray-400">(you)</span>
          )}
        </td>
        <td className="px-4 py-3 text-gray-600">
          {row.discord_id !== null ? "Discord" : "Web"}
        </td>
        <td className="px-4 py-3">
          <Badge
            label={row.is_active ? "Active" : "Inactive"}
            tone={row.is_active ? "green" : "gray"}
          />
        </td>
        <td className="px-4 py-3">
          <Badge
            label={row.is_admin ? "Admin" : "Player"}
            tone={row.is_admin ? "blue" : "gray"}
          />
        </td>
        <td className="px-4 py-3 text-gray-600">{friendlyDate(row.created_at)}</td>
        <td className="px-4 py-3 text-right text-gray-700">{row.pick_count}</td>
        <td className="px-4 py-3">
          <div className="flex flex-wrap items-center justify-end gap-2">
            {busy && <span className="text-xs text-gray-400">Saving…</span>}
            {/* Editing picks is allowed for ANY user INCLUDING self (admin may
               fix own picks off-window), so this is NOT gated by isSelf —
               only by the row's in-flight guard. */}
            <RowButton
              label="Edit picks"
              tone="neutral"
              disabled={busy}
              onClick={() => onEditPicks(row)}
            />
            <RowButton
              label={row.is_active ? "Deactivate" : "Reactivate"}
              tone="neutral"
              disabled={controlsDisabled}
              onClick={() => onToggleActive(row)}
            />
            <RowButton
              label={row.is_admin ? "Revoke admin" : "Grant admin"}
              tone="neutral"
              disabled={controlsDisabled}
              onClick={() => onToggleAdmin(row)}
            />
            <RowButton
              label="Delete"
              tone="danger"
              disabled={controlsDisabled}
              onClick={() => onDelete(row)}
            />
          </div>
        </td>
      </tr>
      {error && (
        <tr>
          <td colSpan={7} className="px-4 pb-3 text-xs text-red-600">
            {error}
          </td>
        </tr>
      )}
    </>
  );
}

// --------------------------------------------------------------------------- //
// Pick-override editor (QT-2)
//
// An admin-only, off-window per-user pick editor. Mirrors MyPicksPage's
// RosterTracker / GameCard / BetOption visual language but is driven by
// useAdminPickEditor (target user_id + admin-chosen season/week) and — crucially
// — does NOT freeze controls by game.locked: the whole point is the override
// bypasses the pick lock. Controls disable ONLY while a slot is saving. Roster
// integrity is enforced server-side and surfaces as inline 4xx.
// --------------------------------------------------------------------------- //

/** Format an ISO kickoff (mirrors MyPicksPage.friendlyKickoff); tolerate null/invalid. */
function friendlyKickoff(iso: string | null): string {
  if (!iso) return "Time TBD";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "Time TBD";
  return d.toLocaleString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

/** Resolve a team_id on a game to its abbreviation (or "—" if unknown). */
function abbrevFor(game: SlateGame, teamId: number | null): string {
  if (teamId === null) return "—";
  if (game.home_team.team_id === teamId) return game.home_team.abbreviation;
  if (game.away_team.team_id === teamId) return game.away_team.abbreviation;
  return "—";
}

/** A short human description of the persisted line for a game card header. */
function lineSummary(game: SlateGame): string {
  const parts: string[] = [];
  if (game.spread !== null && game.favorite_team_id !== null) {
    const fav = abbrevFor(game, game.favorite_team_id);
    const dog = abbrevFor(game, game.underdog_team_id);
    parts.push(`${fav} ${game.spread} over ${dog}`);
  }
  if (game.total !== null) {
    parts.push(`O/U ${game.total}`);
  }
  return parts.length ? parts.join(" · ") : "Line unavailable";
}

/** Which game (abbrevs) a filled slot is on, for the roster tracker. */
function slotGameLabel(
  picks: PicksBySlot,
  slate: SlateGame[],
  key: string,
): string | null {
  const pick = picks[key];
  if (!pick) return null;
  const game = slate.find((g) => g.game_id === pick.game_id);
  if (!game) return PICK_TYPE_LABEL[pick.pick_type];
  return `${game.away_team.abbreviation} @ ${game.home_team.abbreviation}`;
}

/** The off-window editor panel for one target user + season/week. */
function PickOverrideEditor({
  target,
  onClose,
}: {
  target: EditorTarget;
  onClose: () => void;
}) {
  // Local season/week so the admin can re-point the editor at any past/future
  // week (seeded from the target's initial season/week).
  const [season, setSeason] = useState<number>(target.season);
  const [week, setWeek] = useState<number>(target.week);

  const { status, slate, picks, saving, slotError, set, clear, grade } =
    useAdminPickEditor(target.id, season, week);

  return (
    <section
      data-testid="pick-override-editor"
      className="space-y-4 rounded-lg border border-blue-300 bg-white p-4"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold">
            Pick override · {target.display_name}
          </h2>
          <p className="mt-0.5 text-sm text-gray-500">
            Edit this player's roster for any week.
          </p>
        </div>
        <RowButton
          label="Close"
          tone="neutral"
          disabled={false}
          onClick={onClose}
        />
      </div>

      {/* REQUIRED off-window-override affordance: make it OBVIOUS this bypasses
         the normal pick lock while roster rules still apply. */}
      <div
        className="rounded-md border border-amber-300 bg-amber-50 px-4 py-2 text-sm text-amber-800"
        role="alert"
      >
        <span className="font-semibold">Admin override.</span> This editor
        bypasses the normal pick window/lock — you can add or change picks on
        locked or past games. Roster rules (one pick per base type, a single
        mortal lock, eligibility) are still enforced.
      </div>

      <div className="flex flex-wrap items-end gap-4">
        <label className="text-sm">
          <span className="block text-xs font-medium text-gray-600">Season</span>
          <input
            type="number"
            value={season}
            onChange={(e) => setSeason(Number(e.target.value))}
            className="mt-1 w-28 rounded-md border border-gray-300 px-2 py-1 text-sm"
          />
        </label>
        <label className="text-sm">
          <span className="block text-xs font-medium text-gray-600">Week</span>
          <input
            type="number"
            value={week}
            onChange={(e) => setWeek(Number(e.target.value))}
            className="mt-1 w-24 rounded-md border border-gray-300 px-2 py-1 text-sm"
          />
        </label>
      </div>

      {status === "loading" ? (
        <p className="text-gray-500">Loading roster…</p>
      ) : status === "error" ? (
        <p className="text-gray-600">
          Couldn't load this week for {target.display_name}. Check the season/week
          and try again.
        </p>
      ) : (
        <div className="space-y-4">
          <OverrideRosterTracker picks={picks} slate={slate} />

          <MiscOverridePanel
            slate={slate}
            picks={picks}
            saving={saving}
            slotError={slotError}
            onSet={set}
            onGrade={grade}
          />

          {slate.length === 0 ? (
            <p className="text-gray-500">No games are scheduled for this week.</p>
          ) : (
            <div className="space-y-4">
              {slate.map((game) => (
                <OverrideGameCard
                  key={game.game_id}
                  game={game}
                  picks={picks}
                  saving={saving}
                  slotError={slotError}
                  onSet={set}
                  onClear={clear}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

/**
 * The MISC sub-panel of the override editor. Two mutually-exclusive modes,
 * driven by whether the target has an existing MISC pick (slotKey("MISC",false)):
 *   - NO MISC pick → retroactive CREATE: a game <select> + text input + "Add MISC
 *     pick", calling onSet({pick_type:"MISC", misc_text}) (reuses the misc_text-
 *     aware setUserPick → admin_set_pick, window/lock bypassed).
 *   - HAS a MISC pick → GRADE: shows the text + state, then a Correct/Incorrect
 *     choice (REQUIRED before submit — mirrors the server misc_grade_must_decide
 *     so PENDING is never submittable) + a points int input + "Save grade",
 *     calling onGrade({result, points}).
 * No window/lock gating (consistent with the rest of the editor); the only
 * disable is the per-slot saving guard. 4xx surfaces inline via the MISC slot's
 * game-scoped errorKey.
 */
function MiscOverridePanel({
  slate,
  picks,
  saving,
  slotError,
  onSet,
  onGrade,
}: {
  slate: SlateGame[];
  picks: PicksBySlot;
  saving: Record<string, boolean>;
  slotError: Record<string, string>;
  onSet: (item: AdminPickSet) => void;
  onGrade: (body: AdminMiscGrade) => void;
}) {
  const miscPick = picks[slotKey("MISC", false)];
  const savingMisc = Boolean(saving[slotKey("MISC", false)]);

  // Create-mode local state (game + text).
  const [createGameId, setCreateGameId] = useState<number | null>(
    slate[0]?.game_id ?? null,
  );
  const [createText, setCreateText] = useState<string>("");

  // Grade-mode local state (decision + points). Result starts unset so the admin
  // must explicitly choose WIN/LOSS before Save grade enables.
  const [gradeResult, setGradeResult] = useState<"WIN" | "LOSS" | null>(null);
  const [gradePoints, setGradePoints] = useState<string>("0");

  // The inline error is scoped to the MISC slot's game (create uses the selected
  // game; grade uses the existing pick's game).
  const createErrKeyGame = createGameId ?? slate[0]?.game_id ?? 0;
  const createError = slotError[errorKey(createErrKeyGame, "MISC", false)];
  const gradeError = miscPick
    ? slotError[errorKey(miscPick.game_id, "MISC", false)]
    : undefined;

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <h3 className="text-sm font-semibold text-gray-700">Misc pick</h3>

      {!miscPick ? (
        // -------- Retroactive create --------
        <div className="mt-3 space-y-2">
          <p className="text-xs text-gray-500">
            This player has no misc pick for this week. Create one
            retroactively (window/lock bypassed), then grade it.
          </p>
          {slate.length === 0 ? (
            <p className="text-sm text-gray-400">
              No games are scheduled for this week.
            </p>
          ) : (
            <>
              <label className="block text-xs font-medium text-gray-600">
                Game
                <select
                  value={createGameId ?? ""}
                  disabled={savingMisc}
                  onChange={(e) => setCreateGameId(Number(e.target.value))}
                  className={[
                    "mt-1 block w-full rounded-md border border-gray-300 px-2 py-1.5 text-sm",
                    savingMisc ? "cursor-not-allowed opacity-50" : "",
                  ].join(" ")}
                >
                  {slate.map((g) => (
                    <option key={g.game_id} value={g.game_id}>
                      {g.away_team.abbreviation} @ {g.home_team.abbreviation}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block text-xs font-medium text-gray-600">
                Prediction
                <textarea
                  value={createText}
                  disabled={savingMisc}
                  onChange={(e) => setCreateText(e.target.value)}
                  rows={2}
                  placeholder="e.g. Mahomes passes for 400+ yards"
                  className={[
                    "mt-1 block w-full rounded-md border border-gray-300 px-2 py-1.5 text-sm",
                    savingMisc ? "cursor-not-allowed opacity-50" : "",
                  ].join(" ")}
                />
              </label>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  disabled={
                    savingMisc ||
                    createGameId === null ||
                    createText.trim().length === 0
                  }
                  onClick={() =>
                    createGameId !== null &&
                    onSet({
                      game_id: createGameId,
                      pick_type: "MISC",
                      is_mortal_lock: false,
                      misc_text: createText.trim(),
                    })
                  }
                  className={[
                    "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
                    "border-blue-600 bg-blue-600 text-white hover:bg-blue-700",
                    savingMisc ||
                    createGameId === null ||
                    createText.trim().length === 0
                      ? "cursor-not-allowed opacity-50"
                      : "",
                  ].join(" ")}
                >
                  Add Misc pick
                </button>
                {savingMisc && (
                  <span className="text-xs text-gray-400">Saving…</span>
                )}
              </div>
              {createError && (
                <p className="mt-1 text-xs text-red-600">{createError}</p>
              )}
            </>
          )}
        </div>
      ) : (
        // -------- Grade an existing MISC pick --------
        <div className="mt-3 space-y-3">
          <div>
            <p className="text-sm text-gray-800">{miscPick.misc_text}</p>
            <p className="mt-0.5 text-xs text-gray-500">
              {slotGameLabel(picks, slate, slotKey("MISC", false)) ??
                `Game #${miscPick.game_id}`}
              {" · "}
              <MiscResultText result={miscPick.result} points={miscPick.points} />
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium text-gray-600">Grade:</span>
            <button
              type="button"
              disabled={savingMisc}
              onClick={() => setGradeResult("WIN")}
              className={[
                "rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
                gradeResult === "WIN"
                  ? "border-green-600 bg-green-600 text-white"
                  : "border-gray-300 bg-white text-gray-700 hover:border-green-400",
                savingMisc ? "cursor-not-allowed opacity-50" : "",
              ].join(" ")}
            >
              Correct
            </button>
            <button
              type="button"
              disabled={savingMisc}
              onClick={() => setGradeResult("LOSS")}
              className={[
                "rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
                gradeResult === "LOSS"
                  ? "border-red-600 bg-red-600 text-white"
                  : "border-gray-300 bg-white text-gray-700 hover:border-red-400",
                savingMisc ? "cursor-not-allowed opacity-50" : "",
              ].join(" ")}
            >
              Incorrect
            </button>
            <label className="text-xs font-medium text-gray-600">
              Points
              <input
                type="number"
                step={1}
                value={gradePoints}
                disabled={savingMisc}
                onChange={(e) => setGradePoints(e.target.value)}
                className={[
                  "ml-1 w-20 rounded-md border border-gray-300 px-2 py-1 text-sm",
                  savingMisc ? "cursor-not-allowed opacity-50" : "",
                ].join(" ")}
              />
            </label>
            <button
              type="button"
              disabled={savingMisc || gradeResult === null}
              onClick={() =>
                gradeResult !== null &&
                onGrade({
                  result: gradeResult,
                  points: Math.trunc(Number(gradePoints) || 0),
                })
              }
              className={[
                "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
                "border-blue-600 bg-blue-600 text-white hover:bg-blue-700",
                savingMisc || gradeResult === null
                  ? "cursor-not-allowed opacity-50"
                  : "",
              ].join(" ")}
            >
              Save grade
            </button>
            {savingMisc && (
              <span className="text-xs text-gray-400">Saving…</span>
            )}
          </div>
          {gradeResult === null && (
            <p className="text-xs text-gray-400">
              Choose Correct or Incorrect to enable saving (a grade must decide
              the pick).
            </p>
          )}
          {gradeError && <p className="text-xs text-red-600">{gradeError}</p>}
        </div>
      )}
    </div>
  );
}

/** Human description of a MISC pick's current grade state. */
function MiscResultText({
  result,
  points,
}: {
  result: PickResult;
  points: number;
}) {
  if (result === "WIN") {
    return <span className="text-green-700">Correct · {points} pts</span>;
  }
  if (result === "LOSS") {
    return <span className="text-red-700">Incorrect · {points} pts</span>;
  }
  return <span className="text-gray-500">Pending</span>;
}

/** Compact at-a-glance summary of the 5 slots (4 base + the mortal lock). */
function OverrideRosterTracker({
  picks,
  slate,
}: {
  picks: PicksBySlot;
  slate: SlateGame[];
}) {
  const mortalKey = (() => {
    for (const pt of BASE_SLOTS) {
      const k = slotKey(pt, true);
      if (picks[k]) return k;
    }
    return null;
  })();

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <h3 className="text-sm font-semibold text-gray-700">Player roster</h3>
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-5">
        {BASE_SLOTS.map((pt) => {
          const key = slotKey(pt, false);
          return (
            <OverrideSlotChip
              key={key}
              label={PICK_TYPE_LABEL[pt]}
              filled={Boolean(picks[key])}
              detail={slotGameLabel(picks, slate, key)}
            />
          );
        })}
        <OverrideSlotChip
          label="Mortal Lock"
          filled={Boolean(mortalKey)}
          detail={mortalKey ? slotGameLabel(picks, slate, mortalKey) : null}
          accent
        />
      </div>
    </div>
  );
}

function OverrideSlotChip({
  label,
  filled,
  detail,
  accent,
}: {
  label: string;
  filled: boolean;
  detail: string | null;
  accent?: boolean;
}) {
  return (
    <div
      className={[
        "rounded-md border px-2.5 py-2 text-center",
        filled
          ? accent
            ? "border-blue-300 bg-blue-50"
            : "border-green-300 bg-green-50"
          : "border-dashed border-gray-300 bg-gray-50",
      ].join(" ")}
    >
      <div className="text-xs font-medium text-gray-700">{label}</div>
      <div
        className={[
          "mt-0.5 text-xs",
          filled ? "text-gray-600" : "text-gray-400",
        ].join(" ")}
      >
        {filled ? (detail ?? "filled") : "empty"}
      </div>
    </div>
  );
}

/** One slate game in the override editor: matchup/line + eligible bet options.
 * Shows a "locked (override)" badge on locked games but does NOT freeze them. */
function OverrideGameCard({
  game,
  picks,
  saving,
  slotError,
  onSet,
  onClear,
}: {
  game: SlateGame;
  picks: PicksBySlot;
  saving: Record<string, boolean>;
  slotError: Record<string, string>;
  onSet: (item: AdminPickSet) => void;
  onClear: (item: AdminPickSet) => void;
}) {
  const eligibleTypes = BASE_SLOTS.filter((pt) => game.eligibility[pt]);

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <div className="text-base font-semibold">
            {game.away_team.display_name} @ {game.home_team.display_name}
          </div>
          <div className="mt-0.5 text-xs text-gray-500">
            {friendlyKickoff(game.kickoff_at)} · {lineSummary(game)}
          </div>
        </div>
        {game.locked && (
          <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-700">
            locked (override)
          </span>
        )}
      </div>

      {eligibleTypes.length === 0 ? (
        <p className="mt-3 text-sm text-gray-400">
          No bet options are eligible for this game.
        </p>
      ) : (
        <div className="mt-3 space-y-2">
          {eligibleTypes.map((pt) => (
            <OverrideBetOption
              key={pt}
              game={game}
              pickType={pt}
              picks={picks}
              saving={saving}
              slotError={slotError}
              onSet={onSet}
              onClear={onClear}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/** A single eligible pick type on a game: base-pick + mortal-lock toggle + clear.
 * Controls disable ONLY while that slot is saving (NO window/lock gating). */
function OverrideBetOption({
  game,
  pickType,
  picks,
  saving,
  slotError,
  onSet,
  onClear,
}: {
  game: SlateGame;
  pickType: PickType;
  picks: PicksBySlot;
  saving: Record<string, boolean>;
  slotError: Record<string, string>;
  onSet: (item: AdminPickSet) => void;
  onClear: (item: AdminPickSet) => void;
}) {
  const baseKey = slotKey(pickType, false);
  const lockKey = slotKey(pickType, true);

  const baseSelected = picks[baseKey]?.game_id === game.game_id;
  const lockSelected = picks[lockKey]?.game_id === game.game_id;

  const baseSaving = Boolean(saving[baseKey]);
  const lockSaving = Boolean(saving[lockKey]);

  // Errors are game-scoped so a rejection shows only on THIS game's control.
  const baseError = slotError[errorKey(game.game_id, pickType, false)];
  const lockError = slotError[errorKey(game.game_id, pickType, true)];

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={baseSaving}
          onClick={() =>
            onSet({
              game_id: game.game_id,
              pick_type: pickType,
              is_mortal_lock: false,
            })
          }
          className={[
            "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
            baseSelected
              ? "border-blue-600 bg-blue-600 text-white"
              : "border-gray-300 bg-white text-gray-700 hover:border-blue-400",
            baseSaving ? "cursor-not-allowed opacity-50" : "",
          ].join(" ")}
        >
          {PICK_TYPE_LABEL[pickType]}
        </button>

        {baseSelected && (
          <button
            type="button"
            disabled={baseSaving}
            onClick={() =>
              onClear({
                game_id: game.game_id,
                pick_type: pickType,
                is_mortal_lock: false,
              })
            }
            title="Clear this pick"
            className={[
              "rounded-md border px-2 py-1 text-xs font-medium transition-colors",
              "border-red-300 bg-white text-red-600 hover:border-red-500",
              baseSaving ? "cursor-not-allowed opacity-50" : "",
            ].join(" ")}
          >
            Clear
          </button>
        )}

        <button
          type="button"
          disabled={lockSaving}
          onClick={() =>
            onSet({
              game_id: game.game_id,
              pick_type: pickType,
              is_mortal_lock: true,
            })
          }
          title="Designate this as the mortal lock"
          className={[
            "rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors",
            lockSelected
              ? "border-blue-600 bg-blue-50 text-blue-700"
              : "border-gray-300 bg-white text-gray-500 hover:border-blue-400",
            lockSaving ? "cursor-not-allowed opacity-50" : "",
          ].join(" ")}
        >
          {lockSelected ? "★ Mortal lock" : "☆ Mortal lock"}
        </button>

        {lockSelected && (
          <button
            type="button"
            disabled={lockSaving}
            onClick={() =>
              onClear({
                game_id: game.game_id,
                pick_type: pickType,
                is_mortal_lock: true,
              })
            }
            title="Remove the mortal lock"
            className={[
              "rounded-md border px-2 py-1 text-xs font-medium transition-colors",
              "border-red-300 bg-white text-red-600 hover:border-red-500",
              lockSaving ? "cursor-not-allowed opacity-50" : "",
            ].join(" ")}
          >
            ✕ Remove lock
          </button>
        )}

        {(baseSaving || lockSaving) && (
          <span className="text-xs text-gray-400">Saving…</span>
        )}
      </div>

      {baseError && <p className="mt-1 text-xs text-red-600">{baseError}</p>}
      {lockError && lockError !== baseError && (
        <p className="mt-1 text-xs text-red-600">{lockError}</p>
      )}
    </div>
  );
}
