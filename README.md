## 2048 Bot — Real‑Time Training & Visualization

> Production‑grade FastAPI + React system for training a Transformer with Mixture‑of‑Experts (MoE) to play 2048, with real‑time metrics, checkpoint playback, and a mobile‑first PWA frontend.

- Python 3.9+ (FastAPI, PyTorch)
- Node 18+ (Vite, React 18, Tailwind)
- Desktop launcher (console and GUI) with optional Cloudflare Tunnel and first‑run JSON config

### Current status (grounded in code)
- Backend serves training control endpoints, checkpoint management, and WebSocket streaming from `backend/main.py` using `TrainingManager`.
- Deterministic evaluation loop has been added: after synchronous PPO updates, the backend runs greedy (no‑exploration) evaluation over fixed seeds, computes robust metrics (median/percentiles, solve‑rates, max‑tile histogram), broadcasts `evaluation_metrics`, and checkpoints the best model by median score.
- PPO trainer has been upgraded for stability and learning speed:
  - Dual‑critic architecture (extrinsic and intrinsic value heads) with blended advantages for the policy and separate clipped value losses.
  - Synchronous rollout → update cadence driven by the manager.
  - Target‑KL early stop per update, value function clipping (Huber), and clip/entropy schedules.
  - Intrinsic signals are decoupled from extrinsic returns; load‑balancing remains an auxiliary loss only.
- Frontend tabs (mobile‑first PWA) are implemented in `frontend/src` with Zustand stores and a centralized WebSocket client:
  - `Controls` (`components/ControlsDashboard.tsx`) — start/pause/resume/stop/reset training, model size selection (icon updated to Sliders in `App.tsx`).
  - `Game` (`components/GameBoard.tsx`) — live board with attention overlays and playback controls.
  - `Metrics` (`components/TrainingDashboard.tsx`) — graph‑heavy dashboard with loss/score trends, evaluation median trend and max‑tile histogram, action distribution, and an MoE router usage trend with health KPIs (router entropy & active experts) and color‑coded statuses.
  - `Checkpoints` (`components/CheckpointManager.tsx`) — list/stats, search/sort/filter, rename, delete (with confirmation dialog), resume training, and playback; includes autosave interval + long‑run configuration.
  - `Model Studio` (`components/ModelStudioTab.tsx`) — canvas, validation, and stubs for compile/train.
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
- Real‑time training dashboard (Metrics) with:
  - Loss/score charts with smooth animations
  - Deterministic evaluation visuals: median score trend, max‑tile histogram, and solve‑rates
  - Action distribution
  - MoE routing diagnostics: expert usage trend (stacked‑like filled lines), router entropy and active‑expert KPIs
  - All metrics streamed via WebSocket with batching and rate limiting
- Interactive game viewer with attention overlays and live action probabilities; checkpoint playback with pause/resume/stop and speed control.
- Controls tab: start/pause/resume/stop/reset training and model size selection with tokenized, touch‑friendly UI (icon updated to Sliders).
- Checkpoint manager: list, stats, search/sort/filter, nickname edit, delete (gated with confirmation dialog), resume‑training, and playback; autosave interval + long‑run configuration.
- Mobile‑first PWA: touch‑friendly controls, double‑tap chart expansion, offline caching for static assets; adaptive networking for mobile/Safari.
- Model Studio (Week 1 scope shipped): drag‑and‑drop canvas (react‑konva), grid snapping, connections and validation API; compile/train stubs wired to backend routes.
- Desktop launcher: console UI with non‑scrolling animated status; optional desktop GUI with landscape layout, bold service indicators, single master progress bar, large QR, copyable URL; optional Cloudflare Tunnel. Robust dependency detection across Poetry/Node/npm.

Design system & UX (recent updates)
- Tokenized colors and roles across the app per `STYLE_GUIDE.md` (no ad‑hoc hues).
- Skeletons for loading (lists/cards) instead of blocking spinners.
- Reduced‑motion support via `prefers-reduced-motion` for transitions.
- Accessibility: proper roles on lists/regions, `aria-live` for errors, `aria-expanded`/`aria-controls` on expandable rows, and keyboard support for inline edits.

---

### Architecture overview
- Backend: `FastAPI` app in `backend/main.py`
  - Training lifecycle: `TrainingManager` (async loop, synchronous rollout→update, metrics batching, periodic deterministic evaluation, checkpoint autosave & best‑by‑median)
  - WebSocket: connection health, batching, rate limiting, adaptive timeouts (`app/api/websocket_manager.py`); broadcasts `training_update`, `evaluation_metrics`, status/batch messages
  - Checkpoints: list/metadata, manual/interval saves, load/resume, playback (live stream + polling fallback)
  - Model Studio: design CRUD, validation, codegen stub, compile/train routes (`app/api/design_router.py`)
- Frontend: React + Zustand in `frontend/src`
  - Tabs: `ControlsDashboard`, `GameBoard`, `TrainingDashboard` (Metrics), `CheckpointManager`, `ModelStudioTab` (managed in `App.tsx`).
  - Centralized WebSocket client with adaptive reconnect and polling fallback (`utils/websocket.ts`).
  - Checkpoints data layer extracted to `hooks/useCheckpoints.ts` (fetching, config, mutations); destructive actions use `components/ConfirmDialog.tsx`.
  - PWA via Vite plugin; mobile meta + safe‑area and overscroll handling.
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

- **Controls**
  1) Open the Controls tab and select model size (tiny/small/medium/large)
  2) Start training; WebSocket metrics populate the Metrics tab
  3) Pause/Resume/Stop/Reset; create a manual checkpoint anytime
- **Game**
  - View live board, attention overlay, action probabilities; during playback, use pause/resume/stop & speed control
- **Checkpoints**
  - Browse, search, rename, delete; resume training from a checkpoint or start playback
  - Configure autosave interval and long‑run behavior
- **Model Studio**
  - Week‑1 canvas: drag blocks, connect edges, validation endpoint, compile/train stubs
 - **Metrics**
   - Charts and KPIs for training; double‑tap to expand charts; respects reduced motion.

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
- `/ws` streams `training_update`, `evaluation_metrics`, checkpoint playback steps, status updates, and heartbeats; batching and rate limiting enabled

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
- PPO internals: dual‑critic returns and schedules live in `backend/app/training/ppo_trainer.py`; evaluation loop runs via `TrainingManager` after updates; best checkpointing uses median eval score.
- Tests: use the project’s python test runners in `tests/` as needed

License: MIT
