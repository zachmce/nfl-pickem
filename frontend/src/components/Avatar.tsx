/**
 * One reusable avatar for every username surface (Header, Standings, Weekly,
 * Admin). Renders a circular Discord CDN image when the user has a Discord
 * identity (discordId + avatarHash), otherwise initials derived from the
 * display name. A broken/failed image NEVER renders: an onError handler swaps
 * to the initials fallback.
 *
 * Styling uses ONLY the app's semantic theme tokens (see src/index.css @theme):
 * the initials fallback is `bg-surface-raised text-fg-muted` with a
 * `border border-border`, so it stays legible after the dark-mode toggle. No
 * hardcoded hex, no arbitrary color values.
 */
import { useState } from "react";

export interface AvatarProps {
  /**
   * Discord snowflake id as a STRING (a 64-bit id loses precision as a JSON
   * number, corrupting the CDN URL); null for web-origin / non-Discord accounts.
   * Only ever interpolated into the CDN URL or null-checked — never arithmetic.
   */
  discordId: string | null;
  /** Discord avatar hash; null when the user has no custom avatar. */
  avatarHash: string | null;
  /** Display name — source of the initials fallback and the accessible label. */
  displayName: string;
  /** Pixel diameter of the (always-circular) avatar. Defaults to inline size. */
  size?: number;
}

/**
 * Initials from a display name: the first character of up to the first two
 * whitespace-separated words, uppercased ("bot_alpha" -> "B"; "Jane Doe" ->
 * "JD"). Falls back to a single neutral glyph when the name is empty.
 */
function initialsFor(displayName: string): string {
  const words = displayName.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) {
    return "?";
  }
  return words
    .slice(0, 2)
    .map((w) => w.charAt(0).toUpperCase())
    .join("");
}

export default function Avatar({
  discordId,
  avatarHash,
  displayName,
  size = 24,
}: AvatarProps) {
  // Flips to true once the CDN <img> fails to load, so a broken image is never
  // shown — we render the initials fallback instead.
  const [failed, setFailed] = useState(false);

  const dimension = { width: size, height: size } as const;
  const hasImage = discordId !== null && avatarHash !== null && !failed;

  if (hasImage) {
    // Discord CDN avatar URL (locked convention). We force a STATIC `.png`
    // render even when the hash is animated (the `a_` prefix): the CDN serves a
    // static frame at the `.png` path, which avoids autoplaying motion in dense
    // tables and keeps every avatar a single immutable static asset. `?size=64`
    // is a CDN-valid power-of-two render size and is sharp at our small inline
    // diameters (the element is downscaled via the inline width/height).
    const src = `https://cdn.discordapp.com/avatars/${discordId}/${avatarHash}.png?size=64`;
    return (
      <img
        src={src}
        alt={displayName}
        style={dimension}
        onError={() => setFailed(true)}
        className="shrink-0 rounded-full object-cover"
      />
    );
  }

  // Initials fallback — token-styled so it remains legible in light + dark.
  return (
    <span
      role="img"
      aria-label={displayName}
      style={dimension}
      className="inline-flex shrink-0 items-center justify-center rounded-full border border-border bg-surface-raised text-fg-muted font-medium leading-none"
    >
      <span style={{ fontSize: Math.max(10, Math.round(size * 0.42)) }}>
        {initialsFor(displayName)}
      </span>
    </span>
  );
}
