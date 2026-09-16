import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API is served by FastAPI. In dev Vite proxies to it so the browser
// talks to one origin; in the container the built assets are served by
// FastAPI itself, so the same relative paths work without a proxy.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://localhost:8000", changeOrigin: true } },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
