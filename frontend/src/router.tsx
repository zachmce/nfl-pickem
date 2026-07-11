/**
 * Data-router route tree (react-router-dom v7).
 *
 * /login lives OUTSIDE the shell. Everything else is wrapped in RequireAuth ->
 * AppShell, with My Picks as the index and /admin additionally behind
 * RequireAdmin.
 */
import { createBrowserRouter, Navigate } from "react-router-dom";

import RequireAdmin from "./auth/RequireAdmin";
import RequireAuth from "./auth/RequireAuth";
import AppShell from "./components/AppShell";
import AdminPage from "./pages/AdminPage";
import CalendarPage from "./pages/CalendarPage";
import HelpPage from "./pages/HelpPage";
import LoginPage from "./pages/LoginPage";
import MyPicksPage from "./pages/MyPicksPage";
import ProfilePage from "./pages/ProfilePage";
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
          { path: "calendar", element: <CalendarPage /> },
          { path: "help", element: <HelpPage /> },
          { path: "rules", element: <Navigate to="/help" replace /> },
          { path: "profile", element: <ProfilePage /> },
          {
            element: <RequireAdmin />,
            children: [{ path: "admin", element: <AdminPage /> }],
          },
        ],
      },
    ],
  },
]);
