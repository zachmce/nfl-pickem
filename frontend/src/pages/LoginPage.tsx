/**
 * Bare login page rendered OUTSIDE the shell (no Header/ContextBar).
 *
 * On mount it establishes the csrftoken cookie (GET /api/auth/csrf) so the cookie
 * exists for later unsafe requests. Submitting valid credentials POSTs to
 * /api/auth/login, refreshes the auth context, and navigates into the shell (/).
 * A 401 shows an inline error and stays on /login. The demo banner is rendered
 * above the card so the demo signal shows even pre-auth.
 */
import { useEffect, useState, type FormEvent } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { useAuth } from "../auth/useAuth";
import DemoBanner from "../components/DemoBanner";
import { ApiError, api, type UserRead } from "../lib/api";

export default function LoginPage() {
  const { refresh } = useAuth();
  const navigate = useNavigate();
  // Optional one-off notice carried in router state (e.g. redirected here after
  // a password change). Read defensively — state is unknown/null on a fresh nav.
  const notice = (useLocation().state as { notice?: string } | null)?.notice ?? null;

  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    // Establish the csrftoken cookie up front; failures are non-fatal.
    void api("/api/auth/csrf").catch(() => undefined);
  }, []);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await api<UserRead>("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ display_name: displayName, password }),
      });
      await refresh();
      navigate("/", { replace: true });
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setError("Invalid display name or password");
      } else {
        setError("Something went wrong. Please try again.");
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen bg-surface-raised">
      <DemoBanner />
      <div className="flex min-h-screen items-center justify-center px-4">
        <form
          onSubmit={handleSubmit}
          className="w-full max-w-sm rounded-lg border border-border bg-surface p-6 shadow-sm"
        >
          <h1 className="mb-6 text-center text-xl font-bold text-fg">
            🏈 NFL Pick'em
          </h1>

          {notice && (
            <p className="mb-4 rounded bg-success-bg px-3 py-2 text-sm text-success-fg">
              {notice}
            </p>
          )}

          {error && (
            <p className="mb-4 rounded bg-danger-bg px-3 py-2 text-sm text-danger-fg">
              {error}
            </p>
          )}

          <label className="mb-1 block text-sm font-medium text-fg-muted">
            Username
          </label>
          <input
            type="text"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            autoComplete="username"
            required
            className="mb-4 w-full rounded border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none"
          />

          <label className="mb-1 block text-sm font-medium text-fg-muted">
            Password
          </label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
            className="mb-6 w-full rounded border border-border px-3 py-2 text-sm focus:border-accent focus:outline-none"
          />

          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded bg-accent-solid px-4 py-2 font-medium text-on-accent hover:bg-accent-solid-hover disabled:opacity-50"
          >
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </div>
    </div>
  );
}
