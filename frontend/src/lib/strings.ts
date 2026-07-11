/**
 * Canonical loading / error / empty copy for the app's page-level states.
 *
 * One source of truth so the loading/error/empty voice never drifts between
 * pages (decision #6 of the site-consistency pass — locked). The shapes are:
 *   - Loading: "Loading {subject}…"
 *   - Error:   "Couldn't load {subject}. Please try again later."
 *   - Empty:   page-specific, but sourced here so wording stays consistent.
 *
 * Per-row / per-section micro-copy (e.g. a single user's "hidden until lock"
 * hint) is contextual, not a page-level state, and is intentionally NOT here.
 */

// --- Loading -------------------------------------------------------------- //
export const LOADING_MY_PICKS = "Loading your picks…";
export const LOADING_STANDINGS = "Loading the season scoreboard…";
export const LOADING_WEEKLY = "Loading this week's picks…";
export const LOADING_CALENDAR = "Loading the schedule…";
export const LOADING_WEEK_STATUS = "Loading week…";

// --- Error ---------------------------------------------------------------- //
export const ERROR_MY_PICKS =
  "Couldn't load this week's picks. Please try again later.";
export const ERROR_STANDINGS =
  "Couldn't load the standings. Please try again later.";
export const ERROR_WEEKLY =
  "Couldn't load the weekly results. Please try again later.";
export const ERROR_CALENDAR =
  "Couldn't load the calendar. Please try again later.";
/**
 * ContextBar's inline error stays terse — it is a one-line chip, not a page
 * block — but is sourced here so it no longer drifts from the page errors.
 */
export const ERROR_WEEK_STATUS = "Week status unavailable";

// --- Auth notices --------------------------------------------------------- //
/**
 * Shown on /login after a successful password change: the server invalidates the
 * current session, so the user must sign in again with the new password. Pinned
 * by a test — keep the exact wording.
 */
export const PASSWORD_CHANGED_NOTICE =
  "Password changed — sign in again with your new password.";

// --- Empty ---------------------------------------------------------------- //
export const EMPTY_STANDINGS =
  "No scores have been posted yet — the scoreboard will fill in once the season is underway.";
export const EMPTY_WEEKLY = "No picks have been made for this week yet.";
export const EMPTY_CALENDAR = "No games scheduled this month.";
