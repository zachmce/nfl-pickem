/// <reference types="vitest/config" />
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// During dev the Vite server proxies /api to the FastAPI backend so the
// browser only ever talks to one origin (no CORS headaches in dev).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET ?? "http://backend:8000",
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: false, // explicit imports from "vitest" — keeps tsc -b + eslint no-undef clean
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
