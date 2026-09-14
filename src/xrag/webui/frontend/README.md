# XRAG WebUI frontend

The frontend uses Vue 3 single-file components and Vite. FastAPI serves the
production build from `../static`.

```powershell
npm install
npm run build
```

For frontend-only development, start the Python WebUI on port 8765, then run:

```powershell
npm run dev
```

Vite proxies `/api` and `/health` to the Python server.
