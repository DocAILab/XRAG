import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  base: '/static/',
  plugins: [vue()],
  build: {
    outDir: '../static',
    emptyOutDir: false,
    assetsDir: 'assets',
    assetsInlineLimit: 0,
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8765',
      '/health': 'http://127.0.0.1:8765',
    },
  },
});
