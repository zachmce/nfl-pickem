import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { MemoryRouter, Routes, Route, useLocation } from "react-router-dom";

// Partial mock: keep ApiError (and everything else) REAL, stub only api().
vi.mock("../lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api")>()),
  api: vi.fn(),
}));

import ProfilePage from "./ProfilePage";
import { api, ApiError, type UserRead } from "../lib/api";
import { AuthContext } from "../auth/AuthContext";
import { PASSWORD_CHANGED_NOTICE } from "../lib/strings";

// Minimal fake user — only the fields ProfilePage reads. Cast to satisfy the type.
const fakeUser = {
  display_name: "Tester",
  discord_id: null,
  discord_avatar_hash: null,
  created_at: "2026-01-01T00:00:00Z",
} as unknown as UserRead;

// Probe rendered at /login: surfaces the router-state notice so a successful
// redirect is asserted end-to-end by the notice text appearing.
function LoginProbe() {
  const notice = (useLocation().state as { notice?: string } | null)?.notice ?? null;
  return <div>{notice ?? "login-no-notice"}</div>;
}

function renderProfile(logout: () => Promise<void>) {
  return render(
    <AuthContext.Provider
      value={{ user: fakeUser, loading: false, refresh: vi.fn(), logout }}
    >
      <MemoryRouter initialEntries={["/profile"]}>
        <Routes>
          <Route path="/profile" element={<ProfilePage />} />
          <Route path="/login" element={<LoginProbe />} />
        </Routes>
      </MemoryRouter>
    </AuthContext.Provider>,
  );
}

// The three password inputs, in DOM order: current, new, confirm.
function fillPasswords(current: string, next: string, confirm: string) {
  const inputs = document.querySelectorAll<HTMLInputElement>(
    'input[type="password"]',
  );
  fireEvent.change(inputs[0], { target: { value: current } });
  fireEvent.change(inputs[1], { target: { value: next } });
  fireEvent.change(inputs[2], { target: { value: confirm } });
}

describe("ProfilePage password change", () => {
  beforeEach(() => vi.resetAllMocks());
  // globals:false means testing-library's auto-cleanup afterEach is NOT
  // registered — unmount between tests so renders don't accumulate in jsdom.
  afterEach(() => cleanup());

  it("on success clears auth and redirects to /login with a notice", async () => {
    const logout = vi.fn().mockResolvedValue(undefined);
    vi.mocked(api).mockResolvedValue(undefined);

    renderProfile(logout);
    fillPasswords("oldpass123", "newpass123", "newpass123");
    fireEvent.click(screen.getByRole("button", { name: "Change password" }));

    // Notice appearing proves navigation to /login carried the router state.
    expect(await screen.findByText(PASSWORD_CHANGED_NOTICE)).toBeTruthy();
    expect(logout).toHaveBeenCalled();
    expect(api).toHaveBeenCalledWith(
      "/api/auth/change-password",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("on mismatch shows inline error, does not call the API, and stays on Profile", () => {
    const logout = vi.fn().mockResolvedValue(undefined);

    renderProfile(logout);
    fillPasswords("oldpass123", "newpass123", "different123");
    fireEvent.click(screen.getByRole("button", { name: "Change password" }));

    expect(screen.getByText("New passwords do not match")).toBeTruthy();
    expect(api).not.toHaveBeenCalled();
    expect(logout).not.toHaveBeenCalled();
    // Still on Profile — the /login notice never rendered.
    expect(screen.queryByText(PASSWORD_CHANGED_NOTICE)).toBeNull();
  });

  it("on wrong current password shows inline error and stays on Profile", async () => {
    const logout = vi.fn().mockResolvedValue(undefined);
    vi.mocked(api).mockRejectedValue(new ApiError(401, "unauthorized"));

    renderProfile(logout);
    fillPasswords("wrongpass123", "newpass123", "newpass123");
    fireEvent.click(screen.getByRole("button", { name: "Change password" }));

    expect(await screen.findByText("Current password is incorrect")).toBeTruthy();
    expect(screen.queryByText(PASSWORD_CHANGED_NOTICE)).toBeNull();
    expect(logout).not.toHaveBeenCalled();
  });
});
