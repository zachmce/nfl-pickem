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

      // ── react-hooks@7 / react-refresh standing gates (issue #112 CLOSED) ──
      // eslint-plugin-react-hooks@7's flat `recommended` ships three newer,
      // React-Compiler-adjacent rules beyond exhaustive-deps. They were parked at
      // WARN when the linter was first added (real refactors with behavior risk);
      // issue #112 triaged all findings to zero — behavior-preserving effect/ref/
      // context refactors — and promoted every rule back to ERROR. They are now
      // standing gates: new violations fail CI, same as exhaustive-deps.
      "react-hooks/set-state-in-effect": "error",
      "react-hooks/refs": "error",
      "react-refresh/only-export-components": "error",
    },
  },
);
