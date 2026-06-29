/**
 * Small CSRF-aware fetch wrapper for the SPA's cookie-auth API calls.
 *
 * The backend (see backend/app/csrf.py) uses double-submit-cookie CSRF: on unsafe
 * methods (POST/PUT/PATCH/DELETE) it requires the X-CSRF-Token header to match the
 * non-HttpOnly `csrftoken` cookie. This wrapper reads that cookie and echoes it on
 * unsafe methods only. All requests use credentials:"include" so the session +
 * csrftoken cookies flow.
 */

/** A user as returned by the backend (mirrors backend/app/schemas/auth.py UserRead). */
export interface UserRead {
  id: number;
  /** Discord snowflake id as a STRING (a 64-bit id loses precision as a JSON number). */
  discord_id: string | null;
  /** Discord avatar hash; null means no custom avatar (fall back to initials). No URL built here. */
  discord_avatar_hash: string | null;
  display_name: string;
  is_admin: boolean;
  is_active: boolean;
  /** Account join date as an ISO timestamp string (mirrors backend created_at). */
  created_at: string;
}

/** Thrown on a non-2xx response; carries the HTTP status so callers can branch (e.g. 401). */
export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

/** Read a single cookie value from document.cookie, or null if absent. */
function readCookie(name: string): string | null {
  const prefix = `${name}=`;
  for (const part of document.cookie.split(";")) {
    const trimmed = part.trim();
    if (trimmed.startsWith(prefix)) {
      return decodeURIComponent(trimmed.slice(prefix.length));
    }
  }
  return null;
}

/**
 * Fetch wrapper: always credentials:"include"; attaches X-CSRF-Token on unsafe
 * methods; throws ApiError (with status) on non-ok; returns parsed JSON on ok
 * (undefined for 204/empty bodies).
 */
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const method = (init?.method ?? "GET").toUpperCase();
  const headers = new Headers(init?.headers);

  if (init?.body !== undefined && init?.body !== null && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  if (!SAFE_METHODS.has(method)) {
    const csrf = readCookie("csrftoken");
    if (csrf) {
      headers.set("X-CSRF-Token", csrf);
    }
  }

  const res = await fetch(path, {
    ...init,
    method,
    headers,
    credentials: "include",
  });

  if (!res.ok) {
    let message = res.statusText;
    try {
      const body = await res.json();
      message = body?.error?.message ?? body?.detail ?? message;
    } catch {
      // Non-JSON error body — keep the status text.
    }
    throw new ApiError(res.status, message);
  }

  if (res.status === 204 || res.headers.get("Content-Length") === "0") {
    return undefined as T;
  }

  const text = await res.text();
  if (!text) {
    return undefined as T;
  }
  return JSON.parse(text) as T;
}
