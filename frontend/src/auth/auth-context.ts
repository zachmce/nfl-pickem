/**
 * Auth context object + its state type, split out of AuthContext.tsx so that the
 * .tsx exports only the AuthProvider component (react-refresh/only-export-components:
 * a .tsx that also exports a runtime value like createContext breaks Fast Refresh).
 */
import { createContext } from "react";

import type { UserRead } from "../lib/api";

export interface AuthState {
  user: UserRead | null;
  loading: boolean;
  refresh: () => Promise<void>;
  logout: () => Promise<void>;
}

export const AuthContext = createContext<AuthState | null>(null);
