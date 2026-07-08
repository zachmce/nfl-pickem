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
      // ── Residue split (see .planning/todos/pending/frontend-react-hooks-v7-residue.md) ──
      // This task's target was the `react-hooks/exhaustive-deps` bug class, which is
      // ALREADY clean (0 findings). eslint-plugin-react-hooks@7's flat `recommended`
      // set additionally ships three newer, React-Compiler-adjacent rules that fire 14x
      // across pre-existing, deliberate patterns (loading-state effects, latest-value
      // refs, context+provider co-location). Fixing them means real refactors with
      // behavior risk (deriving effect state, splitting context files, retiming ref
      // writes) — out of scope for "add a linter". Per the LOCKED decision (split the
      // large residue into a follow-up rather than balloon this task or mass-disable
      // inline), these three are downgraded to off HERE, documented, and tracked in the
      // follow-up todo above. The value rules — `exhaustive-deps` + rules-of-hooks —
      // stay ERROR and are the standing gate on future edits.
      "react-hooks/set-state-in-effect": "off", // 10 findings: setStatus("loading") before async fetch (correct, intentional)
      "react-hooks/refs": "off", // 2 findings: latest-value ref writes during render in useAdminPickEditor
      "react-refresh/only-export-components": "off", // 2 findings: Auth/Theme context+provider co-located (HMR-only opinion)
    },
  },
);
