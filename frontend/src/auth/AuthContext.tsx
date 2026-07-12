/**
 * Auth context: bootstraps the current user from GET /api/auth/me and exposes
 * {user, loading, refresh, logout} to the tree.
 *
 * The bootstrap NEVER throws into the React tree — a 401 (or any error) from
 * /api/auth/me is treated as "logged out" (user = null), which RequireAuth turns
 * into a redirect to /login. This is the standard SPA pattern.
 */
import { useCallback, useEffect, useState, type ReactNode } from "react";

import { api, type UserRead } from "../lib/api";
import { AuthContext } from "./auth-context";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserRead | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const me = await api<UserRead>("/api/auth/me");
      setUser(me);
    } catch {
      // 401 / network error -> treat as logged out; never throw to the tree.
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await api("/api/auth/logout", { method: "POST" });
    } catch {
      // Logout is best-effort; clear local state regardless.
    }
    setUser(null);
  }, []);

  // Bootstrap the user on mount. The fetch is inlined (rather than calling
  // refresh()) so every setState lives in a .then/.catch/.finally continuation
  // — not synchronously in the effect body (react-hooks/set-state-in-effect).
  // Behavior is identical to `void refresh()`: mount fetch, 401/error -> null,
  // loading -> false. `refresh` stays exposed on the context for post-login use.
  useEffect(() => {
    let cancelled = false;
    api<UserRead>("/api/auth/me")
      .then((me) => {
        if (!cancelled) setUser(me);
      })
      .catch(() => {
        // 401 / network error -> treat as logged out; never throw to the tree.
        if (!cancelled) setUser(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, refresh, logout }}>
      {children}
    </AuthContext.Provider>
  );
}
