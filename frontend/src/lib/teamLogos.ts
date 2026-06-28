/**
 * Team logo URL helper — derive-and-hotlink, no storage, no data-model change.
 *
 * Source of truth: the ESPN "Family A" logo URL family, which is fully
 * derivable from the lowercase team abbreviation (no per-team GUID / stored
 * href needed). See `.planning/notes/team-logos.md` for the full strategy and
 * the two URL families (A = derivable, B = GUID-keyed premium treatments).
 *
 *   color URL:        https://a.espncdn.com/i/teamlogos/nfl/500/{abbr}.png
 *   dark variant:     https://a.espncdn.com/i/teamlogos/nfl/500-dark/{abbr}.png
 *
 * The light-vs-dark choice is encapsulated in ONE place (the `segment` below) so
 * a future theme pass can flip the whole app's logos with a one-line change.
 * Until then the default is "light"; nothing reads `variant` yet on its own.
 */

const BASE = "https://a.espncdn.com/i/teamlogos/nfl";

/**
 * Build the ESPN Family-A color logo URL for a team abbreviation.
 *
 * @param abbreviation team abbreviation (any case, e.g. "KC" or "sf")
 * @param variant      "light" (default) or "dark" — the theme seam
 * @returns the hotlinkable PNG URL with the abbreviation lowercased
 */
export function teamLogoUrl(
  abbreviation: string,
  variant: "light" | "dark" = "light",
): string {
  const abbr = abbreviation.toLowerCase();
  // Single place where the light/dark path segment is decided — a future theme
  // pass swaps the default here (or threads the active theme through) in one line.
  const segment = variant === "dark" ? "500-dark" : "500";
  return `${BASE}/${segment}/${abbr}.png`;
}
