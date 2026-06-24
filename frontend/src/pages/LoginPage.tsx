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
import { useNavigate } from "react-router-dom";

import { useAuth } from "../auth/useAuth";
import DemoBanner from "../components/DemoBanner";
import { ApiError, api, type UserRead } from "../lib/api";

export default function LoginPage() {
  const { refresh } = useAuth();
  const navigate = useNavigate();

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
    <div className="min-h-screen bg-gray-50">
      <DemoBanner />
      <div className="flex min-h-screen items-center justify-center px-4">
        <form
          onSubmit={handleSubmit}
          className="w-full max-w-sm rounded-lg border border-gray-200 bg-white p-6 shadow-sm"
        >
          <h1 className="mb-6 text-center text-xl font-bold text-gray-900">
            🏈 NFL Pick'em
          </h1>

          {error && (
            <p className="mb-4 rounded bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </p>
          )}

          <label className="mb-1 block text-sm font-medium text-gray-700">
            Display name
          </label>
          <input
            type="text"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            autoComplete="username"
            required
            className="mb-4 w-full rounded border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          />

          <label className="mb-1 block text-sm font-medium text-gray-700">
            Password
          </label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
            className="mb-6 w-full rounded border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          />

          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </div>
    </div>
  );
}
