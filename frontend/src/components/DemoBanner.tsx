/**
 * Loud, full-width demo-mode banner.
 *
 * Fetches GET /api/config once and renders a high-contrast warning banner ONLY
 * when is_demo is true. On fetch error it fails safe (treats the app as
 * not-demo) and renders nothing — it must never crash its parent.
 */
import { useEffect, useState } from "react";

import { getConfig } from "../lib/config";

export default function DemoBanner() {
  const [isDemo, setIsDemo] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getConfig()
      .then((cfg) => {
        if (!cancelled) setIsDemo(cfg.is_demo);
      })
      .catch(() => {
        // Fail safe: a config-fetch failure must not block the app or imply demo.
        if (!cancelled) setIsDemo(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!isDemo) {
    return null;
  }

  return (
    <div className="w-full bg-warning-solid px-4 py-2 text-center text-sm font-bold text-on-warning">
      ⚠️ DEMO MODE — fake time-shifted 2025 season, NOT production data.
    </div>
  );
}
