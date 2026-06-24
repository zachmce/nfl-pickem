/** Route guard: admins pass; loaded non-admins are bounced to the index (/). */
import { Navigate, Outlet } from "react-router-dom";

import { useAuth } from "./useAuth";

export default function RequireAdmin() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center text-gray-500">
        Loading…
      </div>
    );
  }

  return user?.is_admin ? <Outlet /> : <Navigate to="/" replace />;
}
