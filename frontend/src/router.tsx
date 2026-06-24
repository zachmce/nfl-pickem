/**
 * Data-router route tree (react-router-dom v7).
 *
 * /login lives OUTSIDE the shell. Everything else is wrapped in RequireAuth ->
 * AppShell, with My Picks as the index and /admin additionally behind
 * RequireAdmin.
 */
import { createBrowserRouter } from "react-router-dom";

import RequireAdmin from "./auth/RequireAdmin";
import RequireAuth from "./auth/RequireAuth";
import AppShell from "./components/AppShell";
import AdminPage from "./pages/AdminPage";
import LoginPage from "./pages/LoginPage";
import MyPicksPage from "./pages/MyPicksPage";
import RulesPage from "./pages/RulesPage";
import StandingsPage from "./pages/StandingsPage";
import WeeklyPage from "./pages/WeeklyPage";

export const router = createBrowserRouter([
  {
    path: "/login",
    element: <LoginPage />,
  },
  {
    element: <RequireAuth />,
    children: [
      {
        element: <AppShell />,
        children: [
          { index: true, element: <MyPicksPage /> },
          { path: "standings", element: <StandingsPage /> },
          { path: "weekly", element: <WeeklyPage /> },
          { path: "rules", element: <RulesPage /> },
          {
            element: <RequireAdmin />,
            children: [{ path: "admin", element: <AdminPage /> }],
          },
        ],
      },
    ],
  },
]);
