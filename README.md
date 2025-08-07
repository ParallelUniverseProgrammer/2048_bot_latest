## 2048 Bot — Real‑Time Training & Visualization

> Production‑grade FastAPI + React system for training a Transformer with Mixture‑of‑Experts (MoE) to play 2048, with real‑time metrics, checkpoint playback, and a mobile‑first PWA frontend.

- Python 3.9+ (FastAPI, PyTorch)
- Node 18+ (Vite, React 18, Tailwind)
- Desktop launcher (console and GUI) with optional Cloudflare Tunnel and first‑run JSON config

### Current status (grounded in code)
- Backend serves training control endpoints, checkpoint management, and WebSocket streaming from `backend/main.py` using `TrainingManager`.
- Frontend provides four tabs: Training, Game, Checkpoints, Model Studio, implemented in `frontend/src` with Zustand stores and WebSocket hydration.
- Checkpoint playback is implemented server‑side (`CheckpointPlayback`) with live step streaming over WebSocket and polling fallback.
- PWA is configured via `vite-plugin-pwa` and service worker caching rules; mobile meta tags are present in `index.html`.
- Launcher (`launcher.py`) starts backend, frontend, and (optionally) a Cloudflare Tunnel; includes a landscape desktop GUI (CustomTkinter) and a modern console UI. First‑run creates `launcher.config.json`; precedence is CLI > JSON > internal defaults.
  - GUI highlights: landscape layout, bold service indicators (Backend / Frontend / Tunnel), single master progress bar, large QR panel, copyable URL. Smooth, non‑blocking animations.
  - Frontend URL ingestion uses environment variables (`VITE_BACKEND_URL`, `VITE_BACKEND_PORT`) instead of temporary Vite config injection.

---

### Table of contents
- Key features
- Architecture overview
- Quick start
- Usage guide
- API surface (selected)
- Troubleshooting
- Development notes

---

### Key features
- Real‑time training dashboard with loss/score charts, action distribution, expert usage, and derived KPIs streamed every episode via WebSocket with batching and rate limiting.
- Interactive game viewer with attention overlays and live action probabilities; checkpoint playback with pause/resume/stop and speed control.
- Checkpoint manager: list, stats, nickname edit, delete, resume‑training, and playback; interval/long‑run configuration endpoints.
- Mobile‑first PWA: touch‑friendly controls, double‑tap chart expansion, offline caching for static assets; adaptive networking for mobile/Safari.
- Model Studio (Week 1 scope shipped): drag‑and‑drop canvas (react‑konva), grid snapping, connections and validation API; compile/train stubs wired to backend routes.
- Desktop launcher: console UI with non‑scrolling animated status; optional desktop GUI with landscape layout, bold service indicators, single master progress bar, large QR, copyable URL; optional Cloudflare Tunnel. Robust dependency detection across Poetry/Node/npm.

---

### Architecture overview
- Backend: `FastAPI` app in `backend/main.py`
  - Training lifecycle: `TrainingManager` (async loop, metrics batching, checkpoint autosave)
  - WebSocket: connection health, batching, rate limiting, adaptive timeouts (`app/api/websocket_manager.py`)
  - Checkpoints: list/metadata, manual/interval saves, load/resume, playback (live stream + polling fallback)
  - Model Studio: design CRUD, validation, codegen stub, compile/train routes (`app/api/design_router.py`)
- Frontend: React + Zustand in `frontend/src`
  - Tabs: `TrainingDashboard`, `GameBoard`, `CheckpointManager`, `ModelStudioTab`
  - WebSocket client with adaptive reconnect and polling fallback (`utils/websocket.ts`)
  - PWA via Vite plugin; mobile meta + safe‑area and overscroll handling
- Launcher: `launcher.py` orchestrates services and (optionally) Cloudflare Tunnel; GUI is optional.

---

### Quick start

```bash
# 1) Clone
# (use your fork/path as appropriate)
# 2) Launch (console UI, first run writes launcher.config.json)
python launcher.py

# 3) Optional: GUI
python launcher.py --gui

# 4) Open frontend
# dev: http://localhost:5173  |  backend: http://localhost:8000
# production build is served from backend if you run a frontend build
```

Notes
- The launcher installs missing GUI deps as needed, starts FastAPI on port 8000 and Vite preview/dev on 5173 (+5174 HMR), and can start a Cloudflare Tunnel if available.
- Without GUI, the console UI shows animated progress and the access URLs; with GUI, a landscape window shows bold status indicators, a master progress bar, and a large QR with the access URL.
- A JSON config is generated on first run; edit it to change defaults. CLI flags take precedence at runtime.

Advanced flags
```bash
python launcher.py --dev                 # dev (hot reload)
python launcher.py --lan-only           # no tunnel
python launcher.py --tunnel-only        # prefer tunnel
python launcher.py --tunnel-type named  # named tunnel (requires cloudflared config)
python launcher.py --port 8000          # backend port
python launcher.py --no-qr              # disable QR focus
python launcher.py --config path/to/custom.config.json  # use custom JSON config
```

---

### Usage guide
- **Training**
  1) Open the Training tab and select model size (tiny/small/medium/large)
  2) Start training; WebSocket metrics populate charts and KPIs
  3) Pause/Resume/Stop; create a manual checkpoint anytime
- **Game**
  - View live board, attention overlay, action probabilities; during playback, use pause/resume/stop & speed control
- **Checkpoints**
  - Browse, search, rename, delete; resume training from a checkpoint or start playback
  - Configure autosave interval and long‑run behavior
- **Model Studio**
  - Week‑1 canvas: drag blocks, connect edges, validation endpoint, compile/train stubs

**Mobile experience**
- PWA meta + service worker, safe‑area padding, overscroll prevention, compact spacing, large touch targets
- WebSocket client adapts to Mobile Safari; automatic polling fallback and upgrade

---

### API surface (selected)
**Training**
- POST `/training/start` { model_size }
- POST `/training/pause` | `/training/resume` | `/training/stop`
- POST `/training/reset`
- GET  `/training/status` | `/training/config`
- GET  `/model/config` | POST `/model/config`

**Checkpoints**
- GET  `/checkpoints` | GET `/checkpoints/stats`
- POST `/checkpoints/refresh`
- GET  `/checkpoints/{id}` | POST `/checkpoints/{id}/nickname`
- DELETE `/checkpoints/{id}`
- POST `/checkpoints/{id}/load` (resume training)
- Playback: POST `/checkpoints/{id}/playback/start`, `.../pause`, `.../resume`, `.../stop`, `.../speed`, GET status/data/model

**WebSocket**
- `/ws` streams `training_update`, checkpoint playback steps, status updates, and heartbeats; batching and rate limiting enabled

---

### Troubleshooting
- WebSocket on mobile/Safari: client falls back to polling; ensure same LAN or use tunnel
- Port conflicts: the backend kills conflicting processes at startup; if needed, change ports via launcher flags
- GUI launcher deps: `customtkinter`, `pillow`, `psutil`, `pystray`; the launcher attempts installation
- Slow GPU/CPU: choose `tiny`/`small` model_size; trainer auto‑selects config based on VRAM

---

### Development notes
- Frontend dev server: `npm i && npm run dev` in `frontend/`
- Backend dev: `uvicorn backend.main:app --reload --port 8000` (or run via launcher `--dev`)
- PWA build: `npm run build` (served by backend if `frontend/dist` exists)
- Tests: use the project’s python test runners in `tests/` as needed

License: MIT
