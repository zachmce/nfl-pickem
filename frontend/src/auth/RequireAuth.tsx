/** Route guard: redirects unauthenticated users to /login; renders the shell otherwise. */
import { Navigate, Outlet } from "react-router-dom";

import { useAuth } from "./useAuth";

export default function RequireAuth() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center text-fg-muted">
        Loading…
      </div>
    );
  }

  return user ? <Outlet /> : <Navigate to="/login" replace />;
}
