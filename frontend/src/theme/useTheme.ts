/** Hook to read the theme context; throws if used outside a ThemeProvider. */
import { useContext } from "react";

import { ThemeContext, type ThemeState } from "./theme-context";

export function useTheme(): ThemeState {
  const ctx = useContext(ThemeContext);
  if (ctx === null) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return ctx;
}
