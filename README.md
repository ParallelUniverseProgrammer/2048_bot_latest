# 🚀 2048-Transformer: Real-Time AI Training & Visualization

> A high-performance, real-time visualization platform for training a Mixture-of-Experts (MoE) Transformer to master the game of 2048.

***
> **Disclaimer:** This README was primarily authored and structured by Gemini 2.5 Pro.
***

![2048 AI Training Dashboard](https://raw.githubusercontent.com/krdge/2048_bot_cursor_pro/main/screenshots/2048-ai-training-dashboard.png)

This platform provides a transparent, interactive, and deeply analytical view into the reinforcement learning process, built for both power users and enthusiasts. It features a sophisticated backend training engine and a feature-rich Progressive Web App (PWA) for monitoring and control from any device.

## Table of Contents
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
- [🏛️ System Architecture](#️-system-architecture)
- [🤖 Neural Network Deep Dive](#-neural-network-deep-dive)
- [📈 Performance](#-performance)
- [🆘 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

## ✨ Key Features

- **🧠 Advanced MoE Transformer**: A sophisticated Mixture-of-Experts model that dynamically routes input, balances expert load, and adapts its size based on available VRAM for optimal performance.
- **📊 Comprehensive Visualizations**: Go beyond simple graphs. Analyze attention heatmaps, inspect expert routing in real-time, and monitor a rich set of performance analytics on a sleek, mobile-optimized dashboard.
- **🚀 High-Performance Training**: The system leverages parallel environments to accelerate data collection, targeting over 100 episodes per minute. It's built for serious training.
- **📱 Seamless Mobile Experience**: A full-featured Progressive Web App (PWA) provides a native app-like experience on your phone, with an automated launcher that generates a QR code for instant access.
- **⚙️ Advanced Checkpoint Management**: A robust system for saving, loading, and managing model checkpoints. Includes performance metadata, user-editable tags, and a full playback system to review historical games.
- **🔧 Effortless Developer Experience**: An automated launcher handles all dependency installation and network configuration. Combined with detailed, colored logging and built-in troubleshooting, the development workflow is streamlined and efficient.

## 🛠️ Technology Stack

The project is built on a modern, high-performance stack, chosen for scalability and developer efficiency.

| Component         | Technology                                                                                                  |
| ----------------- | ----------------------------------------------------------------------------------------------------------- |
| **🤖 Backend**      | Python, FastAPI, PyTorch, Uvicorn, Websockets                                                               |
| **🖥️ Frontend**     | React, TypeScript, Vite, Tailwind CSS, Framer Motion, Chart.js, Zustand                                     |
| **📱 Mobile**       | Progressive Web App (PWA) with `vite-plugin-pwa`                                                            |
| **📦 Tooling**      | Poetry (Python), NPM (Node.js), Black, ESLint, Prettier                                                     |
| **⚙️ Automation** | Custom Python Launcher (`launcher.py`)                                                                      |

## ⚙️ Setup & Installation

Follow the path that best suits your needs. For a fast, automated setup, use the launcher. For granular control, follow the manual setup.

### Prerequisites

Ensure the following tools are installed and available in your system's PATH:
- **Python 3.9+**
- **Poetry** (for Python package management)
- **Node.js 18+**
- **NPM** (comes with Node.js)
- **CUDA-compatible GPU** (Optional): The system will automatically fall back to a CPU-optimized configuration if a GPU is not detected.

### 🚀 Automated Launch (Recommended)

The included `launcher.py` script is the most efficient way to get started. It automates dependency installation, network configuration, and process management.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd 2048_bot_cursor_pro
    ```

2.  **Run the launcher:**
    ```bash
    python launcher.py
    ```

The launcher will perform the following actions:
- ✅ Verify all necessary dependencies are installed.
- ✅ Install any missing Python or Node.js packages.
- ✅ Discover the best local IP for LAN access.
- ✅ Start the backend server on `http://localhost:8000`.
- ✅ Start the frontend development server on `http://localhost:3000`.
- ✅ Generate a `mobile_access_qr.png` and display a QR code in the terminal for instant mobile access.

### 🔧 Manual Setup (Power Users)

For developers who require granular control over the startup process, follow these steps.

1.  **Clone the repository and navigate into it.**

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

## 🏛️ System Architecture

The project is a decoupled, two-part system: a Python backend and a React frontend.

- **Backend**: A FastAPI server manages the entire training pipeline. It uses PyTorch for the neural network, Gymnasium for the game environment, and a custom PPO trainer. A `TrainingManager` runs the main loop in a separate thread, allowing for non-blocking control via REST and WebSocket endpoints. All operations are designed to be thread-safe.

- **Frontend**: A responsive React application built with Vite and TypeScript. It uses Zustand for efficient state management, Chart.js for visualizations, and Framer Motion for smooth animations. A `WebSocket` utility handles real-time data streaming, with logic for adaptive timeouts and reconnections to ensure a stable experience, especially on mobile networks.

## 🤖 Neural Network Deep Dive

The core of this project is a purpose-built Mixture-of-Experts (MoE) Transformer. This architecture was chosen over a standard transformer for its ability to develop specialized sub-networks (experts) that can handle distinct patterns and phases of the 2048 game, leading to more nuanced and effective decision-making.

Here’s a breakdown of how the network processes the game state.

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

## 📈 Performance

The system is engineered for high-throughput training and a fluid user experience.

| Metric                        | Target / Feature                                            |
| ----------------------------- | ----------------------------------------------------------- |
| **Training Speed**            | **100+ episodes/minute** via parallel environments.         |
| **GPU Memory**                | **<7GB VRAM** on an RTX 3070 Ti, with dynamic model sizing.   |
| **UI Updates**                | **1-2 seconds** via optimized WebSocket messages.           |
| **Animation Framerate**       | **60 FPS** for all UI animations and transitions.           |
| **Mobile Load Time**          | **<3 seconds** on a standard connection, enabled by PWA caching. |

## 🆘 Troubleshooting

If you encounter issues, refer to the following guide.

| Issue                          | Solution                                                                                                                                                                                                                                              |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GPU Out of Memory**          | The system automatically falls back to a CPU-optimized configuration. You can monitor VRAM usage in the dashboard. For manual control, modify the `DynamicModelConfig` in the backend code.                                                          |
| **WebSocket Connection Fails** | Ensure the backend server is running. The launcher script handles network discovery and firewall configurations. If running manually, ensure your firewall allows traffic on port `8000`. Check the browser's developer console for error messages. |
| **Mobile Access Issues**       | Confirm your mobile device is on the same Wi-Fi network as the host computer. Use the QR code generated by the launcher for the most reliable connection, as it embeds the correct local IP address.                                                |
| **Dependencies Fail to Install** | Ensure you are using the correct versions of Python (3.9+) and Node.js (18+). The automated launcher (`launcher.py`) handles all dependency installation and is the recommended first step for troubleshooting.                               |

## 🤝 Contributing

Contributions are welcome. Please follow this process:
1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/my-amazing-feature`).
3.  Implement your changes. Ensure code is well-documented and tested.
4.  Verify that your changes are responsive and mobile-friendly.
5.  Submit a pull request with a clear description of the changes and their purpose.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 