import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import { api, ApiError } from "./api";

// Node (undici) provides global fetch/Response/Headers; jsdom provides the
// writable document.cookie jar. We stub fetch per-test and drive the cookie jar
// directly to exercise the CSRF-attach + response-parsing behavior of api().
beforeEach(() => vi.stubGlobal("fetch", vi.fn()));
afterEach(() => {
  vi.unstubAllGlobals();
  document.cookie = "csrftoken=; max-age=0";
});

const mockFetch = () => vi.mocked(fetch);

// Resolve a rejected api() call to its thrown error without failing the test.
const caught = (p: Promise<unknown>): Promise<unknown> =>
  p.then(
    () => null,
    (e: unknown) => e,
  );

describe("api()", () => {
  it("attaches X-CSRF-Token on unsafe methods only", async () => {
    document.cookie = "csrftoken=tok123";
    // A fresh Response per call — a Response body can only be read once.
    const ok = () =>
      new Response("{}", {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    mockFetch().mockResolvedValueOnce(ok()).mockResolvedValueOnce(ok());

    await api("/api/picks", { method: "POST", body: "{}" });
    await api("/api/weekly"); // GET is safe

    const postInit = mockFetch().mock.calls[0][1] as RequestInit;
    const getInit = mockFetch().mock.calls[1][1] as RequestInit;
    expect(new Headers(postInit.headers).get("X-CSRF-Token")).toBe("tok123");
    expect(new Headers(getInit.headers).get("X-CSRF-Token")).toBeNull();
  });

  it("does not attach X-CSRF-Token when the cookie is absent", async () => {
    mockFetch().mockResolvedValue(new Response(null, { status: 204 }));
    await api("/api/x", { method: "DELETE" });
    const init = mockFetch().mock.calls[0][1] as RequestInit;
    expect(new Headers(init.headers).get("X-CSRF-Token")).toBeNull();
  });

  it("returns undefined on a 204 response", async () => {
    mockFetch().mockResolvedValue(new Response(null, { status: 204 }));
    await expect(
      api("/api/x", { method: "DELETE" }),
    ).resolves.toBeUndefined();
  });

  it("returns undefined on an empty 200 body", async () => {
    mockFetch().mockResolvedValue(new Response("", { status: 200 }));
    await expect(api("/api/x")).resolves.toBeUndefined();
  });

  it("throws ApiError carrying the status and {error.message} envelope", async () => {
    mockFetch().mockResolvedValue(
      new Response(JSON.stringify({ error: { message: "nope" } }), {
        status: 403,
        headers: { "Content-Type": "application/json" },
      }),
    );
    const err = await caught(api("/api/x"));
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(403);
    expect((err as ApiError).message).toBe("nope");
  });

  it("falls back to {detail} then statusText for the error message", async () => {
    mockFetch().mockResolvedValue(
      new Response(JSON.stringify({ detail: "bad request" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }),
    );
    const err1 = await caught(api("/api/x"));
    expect((err1 as ApiError).status).toBe(400);
    expect((err1 as ApiError).message).toBe("bad request");

    mockFetch().mockResolvedValue(
      new Response("not json", { status: 500, statusText: "Server Error" }),
    );
    const err2 = await caught(api("/api/x"));
    expect((err2 as ApiError).status).toBe(500);
    expect((err2 as ApiError).message).toBe("Server Error");
  });
});
