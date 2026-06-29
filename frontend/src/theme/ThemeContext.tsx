/**
 * Theme context: tri-state light/dark/system theming persisted to localStorage.
 *
 * Mirrors the AuthContext/useAuth split. The provider initializes from
 * localStorage key 'theme' (defaulting to 'system' when unset/invalid), persists
 * every change, and applies the EFFECTIVE mode to <html> by toggling the `dark`
 * class. In 'system' mode it follows matchMedia('(prefers-color-scheme: dark)')
 * live (decision 2). The class-application rule matches the inline no-flash script
 * in index.html (same key, same rule) so there is no post-mount reflow.
 */
import {
  createContext,
  useCallback,
  useEffect,
  useState,
  type ReactNode,
} from "react";

export type Theme = "light" | "dark" | "system";

export interface ThemeState {
  /** The user's stored choice (light | dark | system). */
  theme: Theme;
  /** The effective rendered mode after resolving 'system' against the OS. */
  resolved: "light" | "dark";
  setTheme: (theme: Theme) => void;
}

export const ThemeContext = createContext<ThemeState | null>(null);

const STORAGE_KEY = "theme";

const DARK_QUERY = "(prefers-color-scheme: dark)";

function prefersDark(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia(DARK_QUERY).matches
  );
}

function readStored(): Theme {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    if (value === "light" || value === "dark" || value === "system") {
      return value;
    }
  } catch {
    /* storage unavailable — fall through to default */
  }
  return "system";
}

/** Resolve a stored choice to the concrete mode to render. */
function resolve(theme: Theme): "light" | "dark" {
  if (theme === "system") return prefersDark() ? "dark" : "light";
  return theme;
}

/** Apply (or clear) the `dark` class on <html> — same rule as the no-flash script. */
function applyClass(resolved: "light" | "dark") {
  document.documentElement.classList.toggle("dark", resolved === "dark");
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(readStored);
  const [resolved, setResolved] = useState<"light" | "dark">(() =>
    resolve(readStored()),
  );

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* persistence is best-effort */
    }
  }, []);

  // Apply the effective mode whenever the choice changes.
  useEffect(() => {
    const next = resolve(theme);
    setResolved(next);
    applyClass(next);
  }, [theme]);

  // In 'system' mode, follow live OS preference flips (decision 2).
  useEffect(() => {
    if (theme !== "system") return;
    const mql = window.matchMedia(DARK_QUERY);
    const onChange = () => {
      const next = prefersDark() ? "dark" : "light";
      setResolved(next);
      applyClass(next);
    };
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, resolved, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}
