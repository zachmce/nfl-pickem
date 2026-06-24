/** Hook to read the auth context; throws if used outside an AuthProvider. */
import { useContext } from "react";

import { AuthContext, type AuthState } from "./AuthContext";

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (ctx === null) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return ctx;
}
