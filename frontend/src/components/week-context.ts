/**
 * Week context object + its state type, split out of WeekContext.tsx so that the
 * .tsx exports only the WeekProvider component (react-refresh/only-export-components:
 * a .tsx that also exports a runtime value like createContext breaks Fast Refresh).
 */
import { createContext } from "react";

import type { CurrentWeek } from "../lib/currentWeek";

export interface WeekState {
  data: CurrentWeek | null;
  status: "loading" | "ok" | "error";
}

export const WeekContext = createContext<WeekState | null>(null);
