/** Client config read from the unauthenticated GET /api/config (demo signal). */
import { api } from "./api";

export interface AppConfig {
  is_demo: boolean;
  season: number;
}

/** Fetch the public app config (demo flag + season). */
export function getConfig(): Promise<AppConfig> {
  return api<AppConfig>("/api/config");
}
