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
  grantAdmin,
  listUsers,
  reactivateUser,
  revokeAdmin,
  type AdminUser,
} from "../lib/admin";
import { useAuth } from "../auth/useAuth";

type LoadStatus = "loading" | "ok" | "error";

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
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
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
}: {
  row: AdminUser;
  isSelf: boolean;
  busy: boolean;
  error: string | undefined;
  onToggleActive: (row: AdminUser) => void;
  onToggleAdmin: (row: AdminUser) => void;
  onDelete: (row: AdminUser) => void;
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
