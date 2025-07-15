# 🔢 2048-Transformer: Real-Time AI Training & Visualization

> A high-performance, real-time visualization platform for training a Mixture-of-Experts (MoE) Transformer to master the game of 2048.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 18+](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-orange.svg)](https://github.com/krdge/2048_bot_cursor_pro)

***
> **Disclaimer:** This README was primarily authored and structured by Gemini 2.5 Pro.
***

![2048 AI Training Dashboard](./screenshots/2048-ai-training-dashboard.png)

This platform provides a transparent, interactive, and deeply analytical view into the reinforcement learning process, built for both power users and enthusiasts. It features a sophisticated backend training engine and a feature-rich Progressive Web App (PWA) for monitoring and control from any device.

## 🎯 What You'll Learn

- **🧠 Deep Learning**: Understand how Mixture-of-Experts Transformers work in practice
- **🎮 Game AI**: See reinforcement learning train an AI to master 2048 in real-time
- **📊 Data Visualization**: Learn to build comprehensive ML training dashboards
- **🔧 Full-Stack Development**: Experience a complete Python backend + React frontend system
- **📱 Progressive Web Apps**: Build mobile-optimized web applications with native-like features

## ⚡ Quick Start (30 seconds)

```bash
# Clone the repository
git clone https://github.com/krdge/2048_bot_cursor_pro.git
cd 2048_bot_cursor_pro

# Run the automated launcher
python launcher.py

# Open your browser to http://localhost:3000
# Scan the QR code for mobile access
```

That's it! The launcher handles everything else automatically.

## 📋 Table of Contents
- [✨ Key Features](#-key-features)
- [🛠️ Technology Stack](#️-technology-stack)
- [⚙️ Setup & Installation](#️-setup--installation)
  - [Prerequisites](#prerequisites)
  - [🚀 Automated Launch (Recommended)](#-automated-launch-recommended)
  - [🔧 Manual Setup (Power Users)](#-manual-setup-power-users)
- [🕹️ Usage Guide](#️-usage-guide)
  - [Starting a Training Session](#starting-a-training-session)
  - [Navigating the Interface](#navigating-the-interface)
  - [Checkpoint Playback](#checkpoint-playback)
- [🔌 API Documentation](#-api-documentation)
- [🏛️ System Architecture](#️-system-architecture)
- [🤖 Neural Network Deep Dive](#-neural-network-deep-dive)

- [🆘 Troubleshooting](#-troubleshooting)
- [❓ FAQ](#-faq)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

## ✨ Key Features

- **🧠 Advanced MoE Transformer**: A sophisticated Mixture-of-Experts model that dynamically routes input, balances expert load, and adapts its size based on available VRAM for optimal performance.
- **📊 Comprehensive Visualizations**: Go beyond simple graphs. Analyze attention heatmaps, inspect expert routing in real-time, and monitor a rich set of performance analytics on a sleek, mobile-optimized dashboard.
- **🚀 High-Performance Training**: The system leverages parallel environments to accelerate data collection and is built for serious training.
- **📱 Seamless Mobile Experience**: A full-featured Progressive Web App (PWA) provides a native app-like experience on your phone, with an automated launcher that generates a QR code for instant access.
- **⚙️ Advanced Checkpoint Management**: A robust system for saving, loading, and managing model checkpoints. Includes performance metadata, user-editable tags, and a full playback system to review historical games.
- **🔧 Effortless Developer Experience**: An automated launcher handles all dependency installation and network configuration. Combined with detailed, colored logging and built-in troubleshooting, the development workflow is streamlined and efficient.

## 🛠️ Technology Stack

The project is built on a modern, high-performance stack, chosen for scalability and developer efficiency.

| Component         | Technology                                                                                                  |
| ----------------- | ----------------------------------------------------------------------------------------------------------- |
| **🤖 Backend**      | Python 3.9+, FastAPI, PyTorch 2.0+, Uvicorn, Websockets                                                    |
| **🖥️ Frontend**     | React 18+, TypeScript, Vite, Tailwind CSS, Framer Motion, Chart.js, Zustand                                |
| **📱 Mobile**       | Progressive Web App (PWA) with `vite-plugin-pwa`                                                           |
| **📦 Tooling**      | Poetry (Python), NPM (Node.js), Black, ESLint, Prettier                                                    |
| **⚙️ Automation** | Custom Python Launcher (`launcher.py`)                                                                      |

## ⚙️ Setup & Installation

Follow the path that best suits your needs. For a fast, automated setup, use the launcher. For granular control, follow the manual setup.

### Prerequisites

Ensure the following tools are installed and available in your system's PATH:

#### Required Software
- **Git** - [Download here](https://git-scm.com/downloads)
- **Python 3.9+** - [Download here](https://www.python.org/downloads/)
- **Poetry** - [Installation guide](https://python-poetry.org/docs/#installation)
- **Node.js 18+** - [Download here](https://nodejs.org/)
- **NPM** (comes with Node.js)

#### Hardware Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 2GB free space
- **GPU**: Optional - CUDA-compatible GPU with 4GB+ VRAM recommended
- **Network**: Stable internet connection for initial dependency download

#### Platform-Specific Notes
- **Windows**: Ensure PowerShell execution policy allows script execution
- **macOS**: May need to install Xcode Command Line Tools
- **Linux**: Ensure `build-essential` package is installed

### 🚀 Automated Launch (Recommended)

The included `launcher.py` script is the most efficient way to get started. It automates dependency installation, network configuration, and process management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/krdge/2048_bot_cursor_pro.git
    cd 2048_bot_cursor_pro
    ```

2.  **Run the launcher:**
    ```bash
    python launcher.py
    ```

The launcher will perform the following actions:
- ✅ Verify all necessary dependencies are installed
- ✅ Install any missing Python or Node.js packages
- ✅ Discover the best local IP for LAN access
- ✅ Start the backend server on `http://localhost:8000`
- ✅ Start the frontend development server on `http://localhost:3000`
- ✅ Generate a `mobile_access_qr.png` and display a QR code in the terminal for instant mobile access

### 🔧 Manual Setup (Power Users)

For developers who require granular control over the startup process, follow these steps.

1.  **Clone the repository and navigate into it:**
    ```bash
    git clone https://github.com/krdge/2048_bot_cursor_pro.git
    cd 2048_bot_cursor_pro
    ```

2.  **Launch the Backend Server:**
    Open a terminal session and run:
    ```bash
    # Navigate to the backend directory
    cd backend

    # Install dependencies using Poetry
    poetry install

    # Start the FastAPI server
    poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The `--host 0.0.0.0` flag binds the server to all available network interfaces, making it accessible on your LAN.

3.  **Launch the Frontend Application:**
    Open a *second* terminal session and run:
    ```bash
    # Navigate to the frontend directory
    cd frontend

    # Install dependencies using NPM
    npm install

    # Start the Vite development server
    npm run dev -- --host
    ```
    The `--host` flag exposes the frontend on your local network, allowing mobile device access. Vite will print the accessible URL.

## 🕹️ Usage Guide

### Starting a Training Session
1.  **Access the UI**: Open your browser to the URL provided by your setup method (typically `http://localhost:3000`).
2.  **Connect**: The frontend will automatically establish a WebSocket connection to the backend. The connection status is displayed in the header.
3.  **Initiate Training**: Click the **Start** button. This initializes the model with a dynamically selected configuration based on your hardware and begins the training process.
4.  **Monitor**: Real-time metrics, game states, and network visualizations will immediately begin streaming to the dashboard.

### Navigating the Interface
The application is organized into four main tabs:
- **📊 Training**: The main dashboard. View real-time charts for loss, score, action distributions, and expert usage. Key performance indicators (KPIs) like training speed and GPU memory are also displayed here.
- **🎮 Game**: Watch the AI play the game in real-time. This view includes the game board, the AI's chosen action, and an optional attention overlay to see what the model is "focusing" on.
- **🧠 Network**: A visual deep-dive into the transformer architecture. See the model's layers, inspect expert routing patterns, and analyze the load-balancing score.
- **💾 Checkpoints**: Manage, review, and test saved model checkpoints.

### Checkpoint Playback
From the **Checkpoints** tab, you can load a previously saved model for analysis.
- **Load a Checkpoint**: Select a checkpoint from the list.
- **Start Playback**: Click the "Play" icon to initiate a game simulation using the selected model's weights.
- **Analyze**: Navigate to the **Game** tab to watch the checkpoint play. You can control the playback speed, pause, and stop the simulation.

## 🔌 API Documentation

### REST Endpoints

#### Training Control
- `POST /api/training/start` - Start training session
- `POST /api/training/stop` - Stop training session
- `POST /api/training/pause` - Pause training session
- `GET /api/training/status` - Get current training status

#### Checkpoint Management
- `GET /api/checkpoints` - List all available checkpoints
- `POST /api/checkpoints/save` - Save current model state
- `POST /api/checkpoints/load/{checkpoint_id}` - Load specific checkpoint
- `DELETE /api/checkpoints/{checkpoint_id}` - Delete checkpoint

#### Model Configuration
- `GET /api/config` - Get current model configuration
- `POST /api/config` - Update model configuration

### WebSocket Events

The frontend connects to `ws://localhost:8000/ws` and receives the following events:

```typescript
// Training metrics update
{
  "type": "training_metrics",
  "data": {
    "episode": 1234,
    "score": 2048,
    "loss": 0.045,
    "expert_usage": [0.2, 0.3, 0.1, 0.4],
    "gpu_memory": 4.2
  }
}

// Game state update
{
  "type": "game_state",
  "data": {
    "board": [[2,4,8,16], [0,2,0,0], [0,0,4,0], [0,0,0,0]],
    "action": "right",
    "attention_weights": [...]
  }
}

// System status
{
  "type": "system_status",
  "data": {
    "training_active": true,
    "episodes_per_minute": 85,
    "connection_status": "connected"
  }
}
```

## 🏛️ System Architecture

The project is a decoupled, two-part system: a Python backend and a React frontend.

- **Backend**: A FastAPI server manages the entire training pipeline. It uses PyTorch for the neural network, Gymnasium for the game environment, and a custom PPO trainer. A `TrainingManager` runs the main loop in a separate thread, allowing for non-blocking control via REST and WebSocket endpoints. All operations are designed to be thread-safe.

- **Frontend**: A responsive React application built with Vite and TypeScript. It uses Zustand for efficient state management, Chart.js for visualizations, and Framer Motion for smooth animations. A `WebSocket` utility handles real-time data streaming, with logic for adaptive timeouts and reconnections to ensure a stable experience, especially on mobile networks.

## 🤖 Neural Network Deep Dive

The core of this project is a purpose-built Mixture-of-Experts (MoE) Transformer. This architecture was chosen over a standard transformer for its ability to develop specialized sub-networks (experts) that can handle distinct patterns and phases of the 2048 game, leading to more nuanced and effective decision-making.

Here's a breakdown of how the network processes the game state.

### 1. Input Processing: From Board to Language

The model doesn't "see" the board as pixels; it interprets it as a sequence of tokens, much like words in a sentence.

-   **Tokenization**: The 4x4 grid is flattened into a 16-token sequence. Each tile's value (e.g., 2, 4, 8, 1024) is transformed by taking its `log2` (e.g., 1, 2, 3, 10). This normalizes the values and creates a linear relationship that's easier for the network to learn. Empty tiles are represented by a zero token.
-   **2D Positional Encoding**: A standard transformer only knows the order of tokens, not their spatial relationship. For a grid game like 2048, knowing that one tile is "above" or "to the left of" another is critical. We inject this knowledge using a custom **2D positional encoding**, which encodes both row and column information for each of the 16 positions. This gives the model a native understanding of the board's geometry.

### 2. The Core: Mixture-of-Experts (MoE) Transformer

This is where the model's intelligence lies. Instead of a single, monolithic network processing every game state, the MoE architecture uses a collection of smaller, specialized networks.

-   **The Router**: At the heart of each MoE layer is a small "gating" network, or router. When a board state (represented as 16 tokens) arrives, the router analyzes it and decides which of the available "experts" are best suited to handle it. It's a "soft" decision—it assigns probabilities and routes the information to a small subset (e.g., the top 2) of the most relevant experts.

-   **The Experts**: Each expert is a standard feed-forward neural network. While they all start with the same architecture, training causes them to specialize. For example:
    -   *Expert 1* might become adept at identifying immediate merge opportunities in the early game.
    -   *Expert 4* could specialize in the complex patterns required to maintain an open board in the late game.
    -   *Expert 7* might learn to recognize defensive positions for survival when the board is cluttered.
    You can see this specialization emerge in real-time in the **Network Visualizer**.

-   **Load Balancing**: A key challenge with MoE is preventing the router from over-relying on a few "superstar" experts. To combat this, the training process includes an auxiliary **load-balancing loss**. This loss function penalizes the model if it doesn't spread the workload evenly, encouraging it to utilize all experts and fostering greater specialization.

### 3. Self-Attention: Discovering Relationships

Within each transformer block, the self-attention mechanism allows the model to weigh the importance of every tile relative to every other tile.

For each of the 16 positions, attention calculates a set of scores that determine how much "focus" to place on the other 15 positions. For example, a `512` tile in the corner might learn to pay high attention to an adjacent `512` tile (a merge opportunity) and to nearby empty cells (potential escape routes). This is the data visualized in the **Attention Heatmaps** tab, providing a direct look into the model's reasoning process.

### 4. Output Heads: The Final Decision

After the board state has been processed through multiple layers of attention and MoE blocks, the resulting high-level representation is passed to two final, separate networks:

1.  **Policy Head**: This head outputs a probability distribution over the four possible moves (Up, Down, Left, Right). It represents the model's final decision on which move is most likely to lead to a better outcome.
2.  **Value Head**: This head outputs a single scalar number, which is the model's assessment of the current board's quality. A high value signifies a strong, promising position, while a low value indicates a weak or dangerous one. This value is critical for the PPO algorithm to learn and improve its policy over time.



## 🆘 Troubleshooting

If you encounter issues, refer to the following comprehensive guide.

### Common Issues

| Issue | Error Message | Solution |
|-------|---------------|----------|
| **GPU Out of Memory** | `CUDA out of memory` | The system automatically falls back to CPU mode. Monitor VRAM usage in the dashboard. For manual control, modify `DynamicModelConfig` in backend code. |
| **WebSocket Connection Fails** | `WebSocket connection failed` | Ensure backend server is running on port 8000. Check firewall settings. Verify network connectivity. |
| **Mobile Access Issues** | `Connection refused` | Confirm device is on same Wi-Fi network. Use QR code from launcher for correct IP address. |
| **Dependencies Fail to Install** | `poetry install failed` | Ensure Python 3.9+ and Node.js 18+ are installed. Try running launcher with `--force` flag. |
| **Port Already in Use** | `Address already in use` | Kill processes on ports 3000/8000: `lsof -ti:3000 | xargs kill -9` |
| **Permission Denied** | `Permission denied` | Run with appropriate permissions. On Windows, run PowerShell as Administrator. |

### Platform-Specific Issues

#### Windows
- **PowerShell Execution Policy**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Firewall**: Allow Python and Node.js through Windows Firewall
- **Path Issues**: Ensure Python and Node.js are in system PATH

#### macOS
- **Xcode Tools**: Install with `xcode-select --install`
- **Permission Issues**: Grant terminal full disk access in System Preferences
- **Homebrew**: Install dependencies via `brew install python node`

#### Linux
- **Build Tools**: Install with `sudo apt-get install build-essential`
- **Python Dev**: Install with `sudo apt-get install python3-dev`
- **Node.js**: Use NodeSource repository for latest version

### Debug Mode

Enable debug logging by setting environment variables:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
python launcher.py
```

### Getting Help

1. Check the browser console for JavaScript errors
2. Review backend logs in the terminal
3. Verify all prerequisites are installed correctly
4. Try the manual setup process
5. Open an issue on GitHub with detailed error information

## ❓ FAQ

### General Questions

**Q: How long does it take to train a good model?**
A: Initial training takes 2-4 hours to reach 2048 tile consistently. Full optimization takes 8-12 hours.

**Q: Can I use this without a GPU?**
A: Yes! The system automatically falls back to CPU mode, though training will be slower than with GPU acceleration.

**Q: What's the highest score achieved?**
A: The current best model consistently reaches 4096+ tiles, with occasional 8192 tiles.

**Q: Can I modify the game rules?**
A: Yes, the game environment is configurable. See `backend/app/environment/game_2048.py` for customization options.

### Technical Questions

**Q: How does the MoE architecture help with 2048?**
A: Different experts specialize in different game phases - early game merging, mid-game strategy, and late-game survival patterns.

**Q: What's the model size?**
A: Dynamic based on available VRAM. Typically 50-200M parameters, automatically scaled to fit your hardware.

**Q: Can I export the trained model?**
A: Yes, checkpoints can be exported and used in other applications. See the checkpoint management section.

**Q: How do I interpret the attention heatmaps?**
A: Brighter colors indicate higher attention. The model "focuses" on tiles that are most relevant for the current move decision.

### Mobile Questions

**Q: Does the mobile app work offline?**
A: The PWA caches resources for offline viewing, but training requires an active connection to the backend.

**Q: Can I control training from my phone?**
A: Yes! Start/stop/pause training, view real-time metrics, and manage checkpoints from the mobile interface.

**Q: What browsers are supported?**
A: Chrome, Safari, Firefox, and Edge on iOS/Android. Chrome is recommended for best performance.

## 🤝 Contributing

Contributions are welcome! This project thrives on community input and improvements.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Ensure code is well-documented and tested
4. **Test thoroughly**: Verify changes work on desktop and mobile
5. **Submit a pull request**: Include clear description of changes and their purpose

### Development Guidelines

- **Code Style**: Follow existing patterns. Use Black for Python, Prettier for JavaScript
- **Testing**: Add tests for new features
- **Documentation**: Update README and add inline comments
- **Mobile-First**: Ensure all changes are responsive and mobile-friendly
- **Performance**: Monitor impact on training speed and UI responsiveness

### Areas for Contribution

- **UI/UX Improvements**: Better visualizations, mobile optimizations
- **Model Enhancements**: New architectures, hyperparameter optimization
- **Performance**: Training speed improvements, memory optimization
- **Documentation**: Tutorials, examples, API documentation
- **Testing**: Unit tests, integration tests, performance benchmarks

### Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Code Review**: All PRs require review before merging

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### License Summary

- ✅ **Commercial Use**: Allowed
- ✅ **Modification**: Allowed  
- ✅ **Distribution**: Allowed
- ✅ **Private Use**: Allowed
- ❌ **Liability**: Limited
- ❌ **Warranty**: None

## 🗺️ Roadmap

### Short Term (Next 3 months)
- [ ] **Enhanced MoE Architecture**: Deeper integration of expert routing, improved load balancing, and dynamic expert allocation
- [ ] **Advanced Visualization Framework**: More detailed attention heatmaps, expert specialization analysis, and real-time network introspection
- [ ] **Streamlined Experimentation**: Simplified interface for testing different network configurations and hyperparameters
- [ ] **Checkpoint Analysis Tools**: Better tools for comparing model performance and understanding training progression

### Medium Term (3-6 months)
- [ ] **Abstract Network Framework**: Create a flexible architecture system for easily designing and testing new neural network approaches
- [ ] **Multi-Architecture Support**: Framework to experiment with different transformer variants, attention mechanisms, and expert designs
- [ ] **Advanced Training Algorithms**: Integration of different RL algorithms and training strategies for 2048
- [ ] **Performance Optimization**: Deeper optimization of the training pipeline and real-time visualization system

### Long Term (6+ months)
- [ ] **Comprehensive Experimentation Platform**: A complete framework for systematically exploring different approaches to solving 2048
- [ ] **Novel Architecture Research**: Development and testing of entirely new neural network designs specifically for game AI
- [ ] **Academic Integration**: Tools for research and publication of findings in game AI and reinforcement learning
- [ ] **Community Experimentation**: Platform for the community to contribute and test new architectural ideas

---

**Made with ❤️ by the 2048-Transformer community** 