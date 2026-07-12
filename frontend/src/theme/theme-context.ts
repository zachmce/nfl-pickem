/**
 * Theme context object + its types, split out of ThemeContext.tsx so that the
 * .tsx exports only the ThemeProvider component (react-refresh/only-export-components:
 * a .tsx that also exports a runtime value like createContext breaks Fast Refresh).
 */
import { createContext } from "react";

export type Theme = "light" | "dark" | "system";

export interface ThemeState {
  /** The user's stored choice (light | dark | system). */
  theme: Theme;
  /** The effective rendered mode after resolving 'system' against the OS. */
  resolved: "light" | "dark";
  setTheme: (theme: Theme) => void;
}

export const ThemeContext = createContext<ThemeState | null>(null);
