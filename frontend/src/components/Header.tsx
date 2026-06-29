/**
 * Main header bar: brand + nav + user menu, collapsing to a hamburger below md.
 *
 * The compact week-status chip stays visible at all widths (including when the
 * mobile nav is collapsed). The Admin nav link renders only for is_admin users —
 * a UX guard; the /admin route is independently protected by RequireAdmin and the
 * backend enforces is_admin on any admin endpoint.
 */
import { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";

import { useAuth } from "../auth/useAuth";
import ThemeSwitcher from "../theme/ThemeSwitcher";
import { WeekChip } from "./ContextBar";

interface NavItem {
  to: string;
  label: string;
  end?: boolean;
}

const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "My Picks", end: true },
  { to: "/standings", label: "Standings" },
  { to: "/weekly", label: "Weekly" },
  { to: "/calendar", label: "Calendar" },
  { to: "/rules", label: "Rules" },
];

function navLinkClass({ isActive }: { isActive: boolean }): string {
  return isActive
    ? "font-semibold text-accent"
    : "text-fg-muted hover:text-accent";
}

export default function Header() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);

  async function handleLogout() {
    await logout();
    navigate("/login");
  }

  const navLinks = (
    <>
      {NAV_ITEMS.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.end}
          className={navLinkClass}
          onClick={() => setOpen(false)}
        >
          {item.label}
        </NavLink>
      ))}
      {user?.is_admin && (
        <NavLink
          to="/admin"
          className={navLinkClass}
          onClick={() => setOpen(false)}
        >
          Admin
        </NavLink>
      )}
    </>
  );

  const userMenu = (
    <div className="flex items-center gap-3">
      {user && (
        <span className="text-sm text-fg-muted">{user.display_name}</span>
      )}
      <ThemeSwitcher />
      <button
        type="button"
        onClick={handleLogout}
        className="rounded border border-border bg-surface px-3 py-1 text-sm font-medium text-fg-muted hover:bg-surface-raised"
      >
        Sign out
      </button>
    </div>
  );

  return (
    <header className="border-b border-border bg-surface">
      <div className="mx-auto flex w-full max-w-5xl xl:max-w-7xl 2xl:max-w-[1700px] items-center justify-between px-4 sm:px-6 lg:px-8 py-3">
        {/* Brand */}
        <NavLink to="/" end className="text-lg font-bold text-fg">
          🏈 NFL Pick'em
        </NavLink>

        {/* Desktop nav (md+) */}
        <nav className="hidden items-center gap-6 text-sm md:flex">
          {navLinks}
        </nav>

        {/* Right side: week chip (always visible) + desktop user menu + hamburger */}
        <div className="flex items-center gap-3">
          <WeekChip />
          <div className="hidden md:block">{userMenu}</div>
          <button
            type="button"
            aria-label="Toggle navigation menu"
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
            className="rounded border border-border px-2 py-1 text-fg-muted md:hidden"
          >
            ☰
          </button>
        </div>
      </div>

      {/* Mobile collapsed menu (below md) */}
      {open && (
        <div className="border-t border-border px-4 py-3 md:hidden">
          <nav className="flex flex-col gap-3 text-sm">{navLinks}</nav>
          <div className="mt-3 border-t border-border pt-3">{userMenu}</div>
        </div>
      )}
    </header>
  );
}
