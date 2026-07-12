// eslint.config.js — flat config; run via `eslint src`
import js from "@eslint/js";
import globals from "globals";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";

export default tseslint.config(
  // Global ignores. node_modules/.git are ignored by default in flat config;
  // dist is NOT — add it. public/theme-init.js is outside src, so `eslint src`
  // never touches it (no ignore entry needed for it).
  { ignores: ["dist"] },
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended, // NON type-checked (no parserOptions.project → fast)
      reactHooks.configs.flat.recommended, // rules-of-hooks + exhaustive-deps
      // NOTE: in eslint-plugin-react-refresh@0.5.3 the flat presets are OBJECTS,
      // not factory functions — `configs.vite` already wires the plugin + the
      // only-export-components rule (allowConstantExport: true). (RESEARCH.md's
      // `configs.vite()` call shape was for a different/newer minor; the installed
      // 0.5.3 exports an object, verified via the package's flat `configs`.)
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser, // src is browser code (document/fetch/window) — no-undef is ON in non-typed mode
    },
    rules: {
      // The VALUE gate this task exists for. `eslint-plugin-react-hooks`'s flat
      // `recommended` ships exhaustive-deps at "warn" by default (only rules-of-hooks
      // defaults to "error"), and `eslint src` fails only on errors — so without this
      // override a broken dependency array in the hand-rolled hooks (useWeekly/
      // useMyPicks/useStandings/useAdminPickEditor) would NOT fail CI. Promote it to
      // error. Currently clean (0 findings), so the gate is green today.
      "react-hooks/exhaustive-deps": "error",

      // ── react-hooks@7 residue triage (issue #112) ──
      // eslint-plugin-react-hooks@7's flat `recommended` set ships three newer,
      // React-Compiler-adjacent rules beyond exhaustive-deps. They were parked at WARN
      // when the linter was first added (real refactors with behavior risk, out of scope
      // for "add a linter"); issue #112 triages them to zero and promotes them back to
      // ERROR one group at a time so the gate stays green incrementally.
      //   - set-state-in-effect: RESOLVED (loading-state effects rewritten to the
      //     endorsed "adjust state during render on key change" pattern; auth bootstrap
      //     inlined into .then/.catch/.finally; theme `resolved` derived via
      //     useSyncExternalStore). Promoted to ERROR — a standing gate now.
      "react-hooks/set-state-in-effect": "error",
      //   - refs: RESOLVED (the two latest-value ref writes in useAdminPickEditor
      //     moved from render into effects; the refs are read only in click-fired
      //     callbacks, so effect-time writes are always current). ERROR now.
      "react-hooks/refs": "error",
      "react-refresh/only-export-components": "warn", // 3: context+provider co-located (HMR-only opinion)
    },
  },
);
