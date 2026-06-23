import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During dev the Vite server proxies /api to the FastAPI backend so the
// browser only ever talks to one origin (no CORS headaches in dev).
export default defineConfig({
  plugins: [react()],
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
});
