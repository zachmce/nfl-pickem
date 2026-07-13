import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { MemoryRouter, Routes, Route, Navigate } from "react-router-dom";

import HelpPage from "./HelpPage";

describe("HelpPage", () => {
  // globals:false means testing-library's auto-cleanup afterEach is NOT
  // registered — unmount between tests so renders don't accumulate in jsdom.
  afterEach(() => cleanup());

  it("renders the Help page directly with rules and bot content", () => {
    render(
      <MemoryRouter>
        <HelpPage />
      </MemoryRouter>,
    );

    // The h1 heading.
    expect(screen.getByText("Help")).toBeTruthy();
    // A bot example question (tolerant of surrounding markup).
    expect(
      screen.getByText(/who's gonna cover the Chiefs game/i),
    ).toBeTruthy();
    // A rules topic (exact match targets the "Scoring" accordion summary,
    // not the "(see Scoring)" cross-reference in another section).
    expect(screen.getByText("Scoring")).toBeTruthy();
    // The worked-example accordion (exact summary match).
    expect(screen.getByText("See it in action")).toBeTruthy();
    // Guard the worked-example numbers (regex — text sits inside a table cell).
    expect(screen.getByText(/27 \+ 20 = 47/)).toBeTruthy();
  });

  it("redirects /rules to /help", () => {
    render(
      <MemoryRouter initialEntries={["/rules"]}>
        <Routes>
          <Route path="/help" element={<HelpPage />} />
          <Route path="/rules" element={<Navigate to="/help" replace />} />
        </Routes>
      </MemoryRouter>,
    );

    // The redirect resolved to HelpPage — its heading is present.
    expect(screen.getByText("Help")).toBeTruthy();
  });
});
