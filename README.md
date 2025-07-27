# <img src="./project_icon.png" alt="2048 Bot Icon" width="32" height="32" style="vertical-align: middle; margin-right: 8px;"> 2048 Bot: Real-Time AI Training & Visualization

> A high-performance, real-time platform for training a Mixture-of-Experts (MoE) Transformer to master the game of 2048. Features a unified design system with mobile-first responsive layout, consistent component patterns, and smooth animations powered by Framer Motion.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![Node.js 18+](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-orange.svg)](https://github.com/krdge/2048_bot)

![2048 AI Training Dashboard](./screenshots/2048-ai-training-dashboard.png)

---

## ✨ What's New?

• **🖥️ Desktop GUI Launcher** – **NEW!** Launch with `python launcher.py --gui` for a modern desktop window with progress tracking, status indicators, and service management. Features smooth animations, one-click URL copying, and real-time status monitoring.

• **🎨 Enhanced Console UI** – **IMPROVED!** Ultra-smooth 60fps animations with precise timing, non-blocking architecture, and professional visual effects. Features double-buffered rendering, content change detection, and graceful terminal compatibility.

• **🎯 Unified Design System** – Comprehensive style guide implementation with consistent component patterns, standardized animations, and unified color palette. All components now follow the same layout structure, button patterns, and visual hierarchy for a cohesive experience.

• **🎯 Component Pattern Standardization** – CheckpointManager, GameBoard, and TrainingDashboard now share identical layout architecture with consistent card-glass styling, standardized spacing, and unified animation system using Framer Motion.

• **🎨 Redesigned Training Dashboard** – Streamlined controls with smart button logic, compact charts, and unified metrics display. All functionality preserved in a mobile-optimized layout that fits perfectly on any screen size.

• **📱 Mobile-First Design** – Enhanced responsive layout with touch-friendly controls, double-tap chart expansion, and optimized spacing that works beautifully on phones, tablets, and desktops.

• **⚡ Performance Optimizations** – Reduced chart heights, tighter spacing, and efficient layout that maintains all visual richness while fitting comfortably within screen constraints.

• **🎯 Smart UI Logic** – Context-aware buttons that adapt based on training state, integrated status indicators, and streamlined workflow that reduces cognitive load.

• **Unified Metrics & Visuals** – the Training dashboard now concentrates all key analytics; the previous "Network" tab has been removed for clarity.

• **Progress-First UX** – global top-bar progress indicators replace floating pop-ups, providing unobtrusive feedback during training, checkpoint loading and playback.

• **Mobile Polish** – deeper device detection, Safari fallback logic and graceful connection recovery deliver a first-class PWA experience on iOS/Android.

• **Checkpoint Workflow** – revamped manager with search, inline rename, quick stats and rich loading states.

• **🚀 Background Service Roadmap** – comprehensive 5-phase plan for platform-agnostic installer that transforms this into a production-ready background service with automatic startup, while preserving all real-time functionality.

• **🎨 Modern Launcher UI** – redesigned console interface with smooth progress animations, non-scrolling display, and QR-focused experience that makes mobile access effortless.

• **🎯 Model Studio Canvas** – Complete drag-and-drop canvas implementation with react-konva. Features visual block design, grid snapping, touch gestures (pinch-zoom, pan), connection visualization, and mobile-first interface with 70%+ canvas space allocation. Enhanced error handling and responsive UX.

• **🧪 Test Suite Compliance Overhaul** – Complete standardization of the test suite with TestLogger integration across all 68 test files. All major compliance issues resolved (0 major issues remaining), improved maintainability with proper main() functions and exit codes, and enhanced error handling capabilities.

• **🚀 Backend Decorator Implementation** – **COMPLETED** ✅ Comprehensive backend management system with automatic decorator application. Successfully reduced major compliance issues from 66 to 3 files (97% improvement), implemented `@requires_real_backend` and `@requires_mock_backend` decorators across all test files, and created automated utility scripts for decorator management and validation.

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

• **Desktop GUI Launcher** – Modern desktop window with progress tracking, status indicators, and service management for seamless operation.

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
| Launcher | **CustomTkinter / Console UI / System Tray** |

---

## Quick Start

```bash
# Clone & enter the repo
$ git clone https://github.com/krdge/2048_bot_latest.git && cd 2048_bot_latest

# Fire up everything via the launcher (backend, frontend & tunnel/QR)
$ python launcher.py

# Or launch with desktop GUI window
$ python launcher.py --gui

# Open http://localhost:8000  (QR code displayed prominently in terminal)
```

The launcher installs dependencies, detects hardware (CUDA ↔︎ CPU) and starts both servers with automatic Cloudflare Tunnel creation for internet access. The new console interface features smooth progress animations and prominently displays the QR code for mobile access.

### Advanced Usage

The launcher supports multiple deployment modes:

```bash
# Development mode (LAN only, hot reload)
python launcher.py --dev

# Desktop GUI mode with system tray
python launcher.py --gui

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
The interface is split into four tabs:

| Tab | Description |
|-----|-------------|
| **Training** | Redesigned real-time metrics, compact charts and smart training controls. Features context-aware buttons that adapt based on training state. |
| **Game** | Live board view for training or checkpoint playback with attention overlay and playback controls. |
| **Checkpoints** | Library of saved models with rename, search, resume-training and playback. |
| **Model Studio** 🎯 | Visual model designer with complete drag-and-drop canvas, grid snapping, touch gestures, and real-time validation. Create custom transformer architectures and experiment with MoE configurations with enhanced error handling and mobile optimization. |

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

✅ **Test Suite Compliance Overhaul Complete** – All major compliance issues have been resolved with comprehensive TestLogger integration across the entire test suite.

✅ **Backend Decorator Implementation Complete** – Comprehensive backend management system with automatic decorator application and validation.

⚠️ **Warning**: The backend parity test (`tests/integration/test_backend_parity.py`) is a comprehensive test that validates API consistency between real and mock backends. It takes 10-15 minutes to run and should only be used for thorough validation, not regular testing.

The test suite now features:
• **Standardized Logging** – Consistent TestLogger usage across all 68 test files
• **Zero Major Issues** – All compliance violations resolved (down from 66 major issues to just 3)
• **Automated Backend Management** – `@requires_real_backend` and `@requires_mock_backend` decorators automatically handle backend startup/teardown
• **Improved Maintainability** – Proper main() functions with return values and exit codes
• **Enhanced Error Handling** – Standardized error reporting and debugging capabilities
• **Comprehensive Coverage** – Tests grouped by domain (core, integration, performance, mobile, frontend)
• **Compliance Checker** – Automated tool to ensure all test files follow standards
• **Backend Decorator Utilities** – Automated scripts for decorator application, legacy cleanup, and validation

### Running Tests

**Master Test Runner** (Recommended):
```bash
python tests/runners/master_test_runner.py --level core
```

**Individual Test Categories**:
```bash
# Core functionality tests
python tests/core/test_checkpoint_loading.py

# Integration tests
python tests/integration/test_complete_games.py

# Backend parity test (comprehensive - takes 10-15 minutes)
python tests/integration/test_backend_parity.py

# Frontend tests
python tests/frontend/test_automation.py

# Performance tests
python tests/performance/test_performance.py
```

**Compliance Checker**:
```bash
python tests/compliance_checker.py
```

### Pre-Commit Checklist

**Before committing any test changes, always run:**

1. **Compliance Checker** - Ensure your test files follow standards:
   ```bash
   python tests/compliance_checker.py
   ```

2. **Master Test Runner** - Verify your changes work correctly:
   ```bash
   python tests/runners/master_test_runner.py --level core
   ```

This ensures all test files maintain consistency and functionality before being committed to the repository.

### Test Architecture

The test suite follows a structured approach with:
• **TestLogger** – Standardized logging with colors, formatting, and structured output
• **BackendTester** – Common backend API testing functionality
• **GameTester** – Game playback testing utilities
• **PlaybackTester** – Live playback and control testing

All tests use consistent message prefixes (OK:, ERROR:, WARNING:, INFO:) for easy parsing and visual scanning.

### Backend Parity Testing

The backend parity test (`tests/integration/test_backend_parity.py`) is a comprehensive validation tool that ensures the mock backend accurately reflects the real backend's API behavior. It tests:

• **API Endpoint Consistency** – All endpoints available on both backends
• **Response Format Matching** – Identical JSON structure and data types
• **Error Handling Parity** – Same error responses and status codes
• **WebSocket Functionality** – WebSocket connections and message handling
• **Performance Characteristics** – Response time comparisons within acceptable ranges

**TODO: Backend Parity Cleanup**
- The parity test currently identifies several API inconsistencies between real and mock backends
- These need to be addressed to ensure test reliability and accurate mock backend behavior
- Priority: Fix response format mismatches, missing endpoints, and error handling differences
- Run `python tests/integration/test_backend_parity.py` to see current parity status

### Developer Experience

Hot-reload is enabled out of the box (FastAPI `--reload`, Vite dev server). ESLint / Prettier (frontend) and Black / isort / flake8 (backend) keep the codebase tidy.

See [tests/README.md](./tests/README.md) for comprehensive testing documentation and guidelines.

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
| **GUI launcher not working** | Missing dependencies | Run `pip install customtkinter pillow` to install GUI dependencies |

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

**Q:** *What's the difference between console and GUI launcher?*  
**A:** Console launcher provides ultra-smooth animations and QR-focused display. GUI launcher offers a desktop window with system tray integration, one-click URL copying, and service management controls.

---

## 🚀 Development Roadmap

### Desktop GUI Launcher ✅ **COMPLETED**
We've successfully implemented a **modern desktop GUI launcher** that transforms the development experience:

• **Desktop Window Interface** – Launch with `python launcher.py --gui` for a professional desktop window
• **Progress Tracking** – Real-time progress bars and status indicators with smooth animations
• **Service Management** – One-click stop/restart services and view logs directly from the GUI
• **URL Management** – One-click copying of access URLs with visual feedback
• **Status Monitoring** – Real-time status indicators for backend, frontend, and tunnel services
• **Cross-Platform** – Works seamlessly on Windows, macOS, and Linux with native appearance

**Note**: System tray integration is implemented but may have compatibility issues on some platforms. The desktop window provides full functionality for service management and monitoring.

### Enhanced Console UI ✅ **COMPLETED**
We've completely redesigned the console interface to provide a **professional-grade experience**:

• **Ultra-Smooth 60fps Animations** – Precise timing with `time.perf_counter()` for microsecond accuracy
• **Non-Blocking Architecture** – All heavy operations run in background threads, UI never freezes
• **Double-Buffered Rendering** – Eliminates screen flickering and visual artifacts
• **Content Change Detection** – Hash-based diffing for efficient updates
• **Professional Visual Effects** – Smooth easing, typing effects, pulse animations
• **Terminal Compatibility** – Graceful fallbacks for Unicode/ASCII and color support

### UI/UX Improvements ✅ COMPLETED
We've successfully redesigned the Training Dashboard with a focus on mobile-first design and user experience:

• **Smart Controls** – Context-aware buttons that adapt based on training state
• **Compact Layout** – Optimized spacing and chart sizes that fit perfectly on mobile screens
• **Touch Optimization** – Double-tap chart expansion and touch-friendly controls
• **Responsive Design** – Seamless experience across all device sizes
• **Visual Polish** – Streamlined interface that reduces cognitive load while maintaining all functionality

### Launcher UI Redesign ✅ COMPLETED
We've completely redesigned the launcher console interface to provide a modern, non-scrolling experience:

• **Smooth Progress Animations** – Beautiful animated spinners and progress bars with high refresh rate
• **QR-Focused Display** – Prominent QR code display in the center of the terminal
• **Non-Scrolling Interface** – Clean, single-screen updates that don't clutter the console
• **Mobile-First Experience** – QR code is the primary focus, making mobile access effortless
• **Development Mode** – `--dev` flag provides traditional verbose output for debugging

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

### Model Studio Tab 🎯 **WEEK 1 COMPLETE**
We've successfully implemented a **graphical Model Studio** that transforms this into a visual "Scratch for Machine Learning" environment. Users can now:
• Visually design novel transformer architectures with drag-and-drop blocks and grid snapping
• Experiment with different MoE configurations in real-time with touch gestures
• Test model performance instantly with one-click compilation and training
• See visual connections between components with real-time validation feedback
• Use the interface seamlessly on mobile devices with optimized touch interactions

**✅ Current Progress (Week 1 Complete):**
• **Backend Foundation**: Complete API with dynamic model generation, validation, and training integration
• **Frontend Foundation**: Enhanced mobile-first UI with improved error handling and responsive controls
• **Canvas System**: Complete drag-and-drop interface with react-konva, touch gestures, and connection visualization
• **Mobile Optimization**: Touch-friendly interface with 70%+ canvas space allocation, Safari overscroll prevention
• **Enhanced UX**: Comprehensive error handling, status indicators, and animated metrics display

**🎯 Next Phase (Week 2):** Real-time validation enhancements with floating banners and visual feedback

**📋 Full Roadmap**: See [MODEL_STUDIO_ROADMAP.md](./MODEL_STUDIO_ROADMAP.md) for our comprehensive 10-phase development plan and current implementation status.

## Contributing

1. Fork → feature branch → PR.  
2. Run `python launcher.py --dev` to spin up watch mode for both servers.  
3. Add/adjust tests.  
4. Follow the existing code style (Black, ESLint) and ensure the UI remains responsive on mobile.
5. See [STYLE_GUIDE.md](./STYLE_GUIDE.md) for detailed design and development guidelines.

---

## License

MIT © krdge and contributors 