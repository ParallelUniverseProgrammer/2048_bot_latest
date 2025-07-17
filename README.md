# 🔢 2048 Bot: Real-Time AI Training & Visualization

> A high-performance, real-time platform for training a Mixture-of-Experts (MoE) Transformer to master the game of 2048.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![Node.js 18+](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-orange.svg)](https://github.com/krdge/2048_bot)

![2048 AI Training Dashboard](./screenshots/2048-ai-training-dashboard.png)

---

## ✨ What’s New?

• **Unified Metrics & Visuals** – the Training dashboard now concentrates all key analytics; the previous “Network” tab has been removed for clarity.

• **Progress-First UX** – global top-bar progress indicators replace floating pop-ups, providing unobtrusive feedback during training, checkpoint loading and playback.

• **Mobile Polish** – deeper device detection, Safari fallback logic and graceful connection recovery deliver a first-class PWA experience on iOS/Android.

• **Checkpoint Workflow** – revamped manager with search, inline rename, quick stats and rich loading states.


## 📋 Table of Contents
1. [Key Features](#key-features)
2. [Technology Stack](#technology-stack)
3. [Quick Start](#quick-start)
4. [Usage Guide](#usage-guide)
5. [Test Suite & Developer Experience](#test-suite--developer-experience)
6. [API](#api)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)
9. [Contributing](#contributing)
10. [License](#license)

---

## Key Features

• **Real-Time Training Dashboard** – loss & score charts, action distributions, expert usage and advanced KPIs, all updating every 1-2 s via WebSocket.

• **Interactive Game Viewer** – watch the agent play with attention overlays and live action probabilities.

• **Comprehensive Checkpoint Manager** – browse, rename, delete and resume checkpoints or start instant playback with animated loading states.

• **Adaptive PWA** – installs to mobile home-screen, works offline for checkpoint replay and includes robust connection fall-backs.

• **GPU-Aware Backend** – FastAPI + PyTorch PPO training engine that auto-scales model size to available VRAM, with CPU fallback.

---

## Technology Stack

| Layer | Tech |
|-------|------|
| Backend | **Python 3.9 / FastAPI / PyTorch** |
| RL | Custom PPO with MoE Transformer |
| Frontend | **React + TypeScript / Vite / Tailwind CSS / Framer Motion** |
| State Mgmt | Zustand (with persistence & WebSocket hydration) |
| Charts | Chart.js (mobile-optimised) |
| Testing | pytest, Jest & custom runners |
| Packaging | Poetry (Python) / npm (Node) |

---

## Quick Start

```bash
# Clone & enter the repo
$ git clone https://github.com/krdge/2048_bot_latest.git && cd 2048_bot_latest

# Fire up everything via the launcher (backend, frontend & ngrok/QR)
$ python launcher.py

# Open http://localhost:3000  (QR code printed for mobile)
```

The launcher installs dependencies, detects hardware (CUDA ↔︎ CPU) and starts both servers (with optional hot-reload, although iOS PWA support may break).

---

## Usage Guide

### Navigation
The interface is split into three tabs:

| Tab | Description |
|-----|-------------|
| **Training** | Real-time metrics, charts and training controls (start, pause, stop). |
| **Game** | Live board view for training or checkpoint playback with attention overlay and playback controls. |
| **Checkpoints** | Library of saved models with rename, search, resume-training and playback. |

### Starting a Training Session
1. Launch the backend & frontend (see *Quick Start*).
2. Navigate to **Training** and press **Start**. Select model size if prompted.
3. Monitor metrics or switch to **Game** to visually inspect gameplay.

### Playing a Checkpoint
1. Open **Checkpoints** and click *Watch* on any entry.
2. You’ll be auto-redirected to **Game** while the playback environment starts (progress bar at top).
3. Pause/resume, change speed or start a new game anytime.

---

## Test Suite & Developer Experience

Tests live under `tests/` and are grouped by domain (core, integration, performance, mobile, frontend…). Most integration tests spin up a mock backend automatically.

Run all tests:
```bash
python tests/runners/master_test_runner.py
```

Hot-reload is enabled out of the box (FastAPI `--reload`, Vite dev server). ESLint / Prettier (frontend) and Black / isort / flake8 (backend) keep the codebase tidy.

---

## API

### REST Endpoints (selection)
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/training/start` | Begin training |
| `POST` | `/api/training/stop` | Stop training |
| `POST` | `/api/training/pause` | Pause training |
| `GET`  | `/api/training/status` | Current status |
| `GET`  | `/api/checkpoints` | List checkpoints |
| `POST` | `/api/checkpoints/{id}/load` | Resume training from checkpoint |
| `POST` | `/checkpoints/{id}/playback/start` | Start playback |

### WebSocket Channels
* `training_update` – metrics every 1-2 s
* `game_state` – board + attention weights during training/playback

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| **WebSocket disconnects on mobile Safari** | iOS limitations | Fallback polling activates automatically; try Chrome or ensure same Wi-Fi LAN |
| **CUDA out-of-memory** | Model too large | The backend auto-switches to CPU; lower model size via dropdown before starting training |
| **Port already in use** | Previous run crashed | kill -9 $(lsof -ti:3000,8000) (or use Task Manager on Windows) |

More in `docs/TROUBLESHOOTING.md` (coming soon).

---

## FAQ

**Q:** *Do I need a GPU?*  
**A:** No. Training is slower but fully functional on CPU.

**Q:** *How long until the bot reaches 2048?*  
**A:** ~1–2 h on RTX 3070 for the large model, *significantly* longer on CPU.

**Q:** *Can I export checkpoints?*  
**A:** Yes – checkpoints are standard PyTorch `.pt` files saved under `backend/checkpoints/`.

---

## Contributing

1. Fork → feature branch → PR.  
2. Run `python launcher.py --dev` to spin up watch mode for both servers.  
3. Add/adjust tests.  
4. Follow the existing code style (Black, ESLint) and ensure the UI remains responsive on mobile.

---

## License

MIT © krdge and contributors 