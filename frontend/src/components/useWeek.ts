/** Hook to read the week context; throws if used outside a WeekProvider. */
import { useContext } from "react";

import { WeekContext, type WeekState } from "./WeekContext";

export function useWeek(): WeekState {
  const ctx = useContext(WeekContext);
  if (ctx === null) {
    throw new Error("useWeek must be used within a WeekProvider");
  }
  return ctx;
}
