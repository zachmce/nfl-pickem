/**
 * Tri-state theme switcher: one button that cycles the stored choice
 * light -> dark -> system -> light on click and shows the icon for the CURRENT
 * choice (☀️ light / 🌙 dark / 💻 system). Lives inline in the header user menu
 * (between username and Sign out), so it also surfaces in the mobile menu.
 */
import type { Theme, ThemeState } from "./ThemeContext";
import { useTheme } from "./useTheme";

const NEXT: Record<Theme, Theme> = {
  light: "dark",
  dark: "system",
  system: "light",
};

const ICON: Record<Theme, string> = {
  light: "☀️",
  dark: "🌙",
  system: "💻",
};

const CHOICE_LABEL: Record<Theme, string> = {
  light: "Light",
  dark: "Dark",
  system: "System",
};

export default function ThemeSwitcher() {
  const { theme, setTheme }: ThemeState = useTheme();

  return (
    <button
      type="button"
      onClick={() => setTheme(NEXT[theme])}
      aria-label={`Theme: ${CHOICE_LABEL[theme]} — switch theme`}
      title={`Theme: ${CHOICE_LABEL[theme]} — switch theme`}
      className="rounded px-1.5 py-1 text-sm text-fg-muted hover:bg-surface-raised hover:text-fg"
    >
      <span aria-hidden="true">{ICON[theme]}</span>
    </button>
  );
}
