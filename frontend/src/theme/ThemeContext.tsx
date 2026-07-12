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
  useSyncExternalStore,
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

/** Apply (or clear) the `dark` class on <html> — same rule as the no-flash script. */
function applyClass(resolved: "light" | "dark") {
  document.documentElement.classList.toggle("dark", resolved === "dark");
}

/** useSyncExternalStore subscribe: track live OS color-scheme flips (decision 2). */
function subscribeSystemPref(onChange: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  const mql = window.matchMedia(DARK_QUERY);
  mql.addEventListener("change", onChange);
  return () => mql.removeEventListener("change", onChange);
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(readStored);

  // Live OS preference as an external store: subscribe adds/removes the
  // matchMedia listener; getSnapshot reads the current match; the server
  // snapshot is `false`. This replaces the `resolved` state + its two effects,
  // so there is no setState inside an effect (react-hooks/set-state-in-effect).
  const systemPrefersDark = useSyncExternalStore(
    subscribeSystemPref,
    prefersDark,
    () => false,
  );

  // `resolved` is now DERIVED during render from the choice + the live OS
  // preference (same rule as the no-flash inline script) — system mode still
  // follows the OS live, because a preference flip re-snapshots the store.
  const resolved: "light" | "dark" =
    theme === "system" ? (systemPrefersDark ? "dark" : "light") : theme;

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* persistence is best-effort */
    }
  }, []);

  // The ONLY effect: write the derived mode to <html>. A DOM write in an effect
  // is the correct place (not flagged). Initial render computes the same
  // `resolved` the inline no-flash script already applied, so there is no
  // post-mount reflow/flash.
  useEffect(() => {
    applyClass(resolved);
  }, [resolved]);

  return (
    <ThemeContext.Provider value={{ theme, resolved, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}
