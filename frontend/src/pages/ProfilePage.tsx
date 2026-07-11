/**
 * Profile / Settings page — the app's account home.
 *
 * Rendered inside RequireAuth -> AppShell (see router.tsx), so `user` is present
 * by the time this mounts; the null guard below is just a type narrowing. Shows
 * the avatar + display_name, the join date, and a Discord-vs-web origin line,
 * plus a self-serve password-change form (current / new / confirm).
 *
 * The password POST goes through the CSRF-aware api() client exactly like every
 * other state-changing call — the client attaches X-CSRF-Token on POST and uses
 * credentials:"include", so neither is set here. The password hash is NEVER read
 * or exposed: the form is always shown because every loginable account has a
 * hash (Discord users get a hashed random password at provision; seeded
 * accounts are hashed) and a null-hash account can't log in, so it never reaches
 * this route. Styling uses only the app's semantic theme tokens — no hardcoded
 * hex — matching LoginPage's token usage.
 */
import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";

import { useAuth } from "../auth/useAuth";
import Avatar from "../components/Avatar";
import { ApiError, api } from "../lib/api";
import { formatLocalDateTime } from "../lib/datetime";
import { PASSWORD_CHANGED_NOTICE } from "../lib/strings";

export default function ProfilePage() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // RequireAuth gates the route, so this is a type guard only.
  if (!user) {
    return null;
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);

    if (newPassword !== confirmPassword) {
      // Client-side check — do NOT hit the API when the confirmation mismatches.
      setError("New passwords do not match");
      return;
    }

    setSubmitting(true);
    try {
      await api("/api/auth/change-password", {
        method: "POST",
        body: JSON.stringify({
          current_password: currentPassword,
          new_password: newPassword,
        }),
      });
      // The server invalidates the current session on a password change, so the
      // SPA's session is now dead. Clear the fields, then redirect to /login with
      // a notice — navigate FIRST (route leaves RequireAuth) so logout()'s
      // setUser(null) can't trigger a RequireAuth redirect that would drop the
      // notice; logout() is fire-and-forget local-state cleanup.
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      navigate("/login", {
        replace: true,
        state: { notice: PASSWORD_CHANGED_NOTICE },
      });
      void logout();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setError("Current password is incorrect");
      } else {
        setError("Something went wrong. Please try again.");
      }
    } finally {
      setSubmitting(false);
    }
  }

  const originLabel = user.discord_id !== null ? "Discord account" : "Web account";

  return (
    <div className="mx-auto w-full max-w-lg px-4 py-6">
      {/* Account header */}
      <section className="mb-6 flex items-center gap-4 rounded-lg border border-border bg-surface p-5">
        <Avatar
          discordId={user.discord_id}
          avatarHash={user.discord_avatar_hash}
          displayName={user.display_name}
          size={64}
        />
        <div>
          <h1 className="text-xl font-bold text-fg">{user.display_name}</h1>
          <p className="text-sm text-fg-muted">
            Joined {formatLocalDateTime(user.created_at)}
          </p>
          <p className="text-sm text-fg-muted">{originLabel}</p>
        </div>
      </section>

      {/* Password change */}
      <section className="rounded-lg border border-border bg-surface p-5">
        <h2 className="mb-4 text-lg font-semibold text-fg">Change password</h2>

        {error && (
          <p className="mb-4 rounded bg-danger-bg px-3 py-2 text-sm text-danger-fg">
            {error}
          </p>
        )}

        <form onSubmit={handleSubmit}>
          <label className="mb-1 block text-sm font-medium text-fg-muted">
            Current password
          </label>
          <input
            type="password"
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
            autoComplete="current-password"
            required
            className="mb-4 w-full rounded border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none"
          />

          <label className="mb-1 block text-sm font-medium text-fg-muted">
            New password
          </label>
          <input
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            autoComplete="new-password"
            minLength={8}
            required
            className="mb-4 w-full rounded border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none"
          />

          <label className="mb-1 block text-sm font-medium text-fg-muted">
            Confirm new password
          </label>
          <input
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            autoComplete="new-password"
            minLength={8}
            required
            className="mb-6 w-full rounded border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none"
          />

          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded bg-accent-solid px-4 py-2 font-medium text-on-accent hover:bg-accent-solid-hover disabled:opacity-50"
          >
            {submitting ? "Saving…" : "Change password"}
          </button>
        </form>
      </section>
    </div>
  );
}
