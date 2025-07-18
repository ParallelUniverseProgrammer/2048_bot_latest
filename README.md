# 🔢 2048 Bot: Real-Time AI Training & Visualization

> A high-performance, real-time platform for training a Mixture-of-Experts (MoE) Transformer to master the game of 2048.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![Node.js 18+](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-orange.svg)](https://github.com/krdge/2048_bot)

![2048 AI Training Dashboard](./screenshots/2048-ai-training-dashboard.png)

---

## ✨ What's New?

• **🎨 Redesigned Training Dashboard** – Streamlined controls with smart button logic, compact charts, and unified metrics display. All functionality preserved in a mobile-optimized layout that fits perfectly on any screen size.

• **📱 Mobile-First Design** – Enhanced responsive layout with touch-friendly controls, double-tap chart expansion, and optimized spacing that works beautifully on phones, tablets, and desktops.

• **⚡ Performance Optimizations** – Reduced chart heights, tighter spacing, and efficient layout that maintains all visual richness while fitting comfortably within screen constraints.

• **🎯 Smart UI Logic** – Context-aware buttons that adapt based on training state, integrated status indicators, and streamlined workflow that reduces cognitive load.

• **Unified Metrics & Visuals** – the Training dashboard now concentrates all key analytics; the previous "Network" tab has been removed for clarity.

• **Progress-First UX** – global top-bar progress indicators replace floating pop-ups, providing unobtrusive feedback during training, checkpoint loading and playback.

• **Mobile Polish** – deeper device detection, Safari fallback logic and graceful connection recovery deliver a first-class PWA experience on iOS/Android.

• **Checkpoint Workflow** – revamped manager with search, inline rename, quick stats and rich loading states.

• **🚀 Background Service Roadmap** – comprehensive 5-phase plan for platform-agnostic installer that transforms this into a production-ready background service with automatic startup, while preserving all real-time functionality.

---

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

• **Real-Time Training Dashboard** – Redesigned with smart controls, compact charts, and unified metrics. Loss & score charts, action distributions, expert usage and advanced KPIs, all updating every 1-2 s via WebSocket.

• **Interactive Game Viewer** – watch the agent play with attention overlays and live action probabilities.

• **Comprehensive Checkpoint Manager** – browse, rename, delete and resume checkpoints or start instant playback with animated loading states.

• **Adaptive PWA** – installs to mobile home-screen, works offline for checkpoint replay and includes robust connection fall-backs. Optimized for touch interfaces with double-tap chart expansion.

• **GPU-Aware Backend** – FastAPI + PyTorch PPO training engine that auto-scales model size to available VRAM, with CPU fallback.

• **Mobile-First Design** – Responsive layout that adapts perfectly to any screen size, with touch-optimized controls and efficient use of screen real estate.

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

# Fire up everything via the launcher (backend, frontend & tunnel/QR)
$ python launcher.py

# Open http://localhost:8000  (QR code printed for mobile)
```

The launcher installs dependencies, detects hardware (CUDA ↔︎ CPU) and starts both servers with automatic Cloudflare Tunnel creation for internet access.

### Advanced Usage

The launcher supports multiple deployment modes:

```bash
# Development mode (LAN only, hot reload)
python launcher.py --dev

# LAN access only (no tunnel, faster startup)
python launcher.py --lan-only

# Tunnel access only (cloud-first deployment)
python launcher.py --tunnel-only

# Production mode (named tunnel, optimized build)
python launcher.py --production

# Custom configuration
python launcher.py --port 9000 --tunnel-type named --no-qr
```

See [LAUNCHER_README.md](./LAUNCHER_README.md) for complete usage instructions and all available options.

---

## Usage Guide

### Navigation
The interface is split into three tabs:

| Tab | Description |
|-----|-------------|
| **Training** | Redesigned real-time metrics, compact charts and smart training controls. Features context-aware buttons that adapt based on training state. |
| **Game** | Live board view for training or checkpoint playback with attention overlay and playback controls. |
| **Checkpoints** | Library of saved models with rename, search, resume-training and playback. |

### Starting a Training Session
1. Launch the backend & frontend (see *Quick Start*).
2. Navigate to **Training** and press **Start Training**. Select model size if prompted.
3. Monitor metrics or switch to **Game** to visually inspect gameplay.
4. Use the smart controls to pause, resume, or create manual checkpoints as needed.

### Mobile Experience
• **Touch-Optimized**: All controls are designed for touch interaction
• **Chart Expansion**: Double-tap any chart to view it in full-screen mode
• **Responsive Layout**: Automatically adapts to your device's screen size
• **PWA Installation**: Install directly to your home screen for offline access

### Remote Access
The launcher automatically creates a Cloudflare Tunnel for internet access:
- **Quick Tunnel**: Temporary HTTPS URL (no account required)
- **Named Tunnel**: Persistent HTTPS URL (requires Cloudflare account setup)
- **Mobile PWA**: Install directly from tunnel URL for offline access
- **QR Code**: Scan to access from any mobile device

### Playing a Checkpoint
1. Open **Checkpoints** and click *Watch* on any entry.
2. You'll be auto-redirected to **Game** while the playback environment starts (progress bar at top).
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

**Q:** *How does the mobile experience work?*  
**A:** The interface is fully responsive and touch-optimized. Double-tap charts to expand them, and all controls are designed for mobile interaction.

---

## 🚀 Development Roadmap

### UI/UX Improvements ✅ COMPLETED
We've successfully redesigned the Training Dashboard with a focus on mobile-first design and user experience:

• **Smart Controls** – Context-aware buttons that adapt based on training state
• **Compact Layout** – Optimized spacing and chart sizes that fit perfectly on mobile screens
• **Touch Optimization** – Double-tap chart expansion and touch-friendly controls
• **Responsive Design** – Seamless experience across all device sizes
• **Visual Polish** – Streamlined interface that reduces cognitive load while maintaining all functionality

### Remote Access Integration ✅ COMPLETED
We've successfully implemented **Cloudflare Tunnel integration** to transform this LAN-only development tool into an internet-reachable, HTTPS-secured service. This enables:

• **Internet Accessibility** – Access from anywhere via HTTPS with automatic QR code generation
• **Zero Configuration** – Automatic tunnel setup with Quick Tunnel fallback
• **Production Ready** – Named tunnels with auto-reconnect, monitoring, and service persistence
• **Mobile PWA Support** – Seamless installation and offline functionality across all devices

The launcher now supports multiple deployment modes including development, production, and cloud-first configurations. See [LAUNCHER_README.md](./LAUNCHER_README.md) for complete usage instructions.

### Background Service Installer (Planned)
We're working on a **platform-agnostic background service installer** that will transform this development tool into a production-ready service. This will enable:

• **Automatic Startup** – runs as a background service with system boot
• **Persistent Training** – survives system restarts and continues training sessions
• **Service Management** – easy install/uninstall with comprehensive monitoring
• **Cross-Platform** – works seamlessly on Windows, macOS, and Linux

See [BACKGROUND_SERVICE_ROADMAP.md](./BACKGROUND_SERVICE_ROADMAP.md) for the complete 5-phase development plan.

### Future Vision: Model Studio Tab
Our long-term vision includes a **graphical Model Studio** where users can:
• Visually design novel transformer architectures
• Experiment with different MoE configurations in real-time
• Test model performance instantly
• Save and share custom model designs

This will leverage our existing checkpoint system and WebSocket infrastructure to provide a powerful visual AI model development environment.

## Contributing

1. Fork → feature branch → PR.  
2. Run `python launcher.py --dev` to spin up watch mode for both servers.  
3. Add/adjust tests.  
4. Follow the existing code style (Black, ESLint) and ensure the UI remains responsive on mobile.
5. See [STYLE_GUIDE.md](./STYLE_GUIDE.md) for detailed design and development guidelines.

---

## License

MIT © krdge and contributors 