import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    watch: {
      usePolling: true,
    },
    proxy: {
      "/api": {
        target: "http://localhost:8200",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8200",
        ws: true,
      },
      "/bridge-api": {
        target: "http://localhost:8201",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/bridge-api/, "/api"),
      },
      "/bridge-ws": {
        target: "ws://localhost:8201",
        ws: true,
        rewrite: (path) => path.replace(/^\/bridge-ws/, "/ws"),
      },
      "/voice-api": {
        target: "http://localhost:8202",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/voice-api/, "/api"),
      },
      "/voice-ws": {
        target: "ws://localhost:8202",
        ws: true,
        rewrite: (path) => path.replace(/^\/voice-ws/, "/ws"),
      },
    },
  },
});
