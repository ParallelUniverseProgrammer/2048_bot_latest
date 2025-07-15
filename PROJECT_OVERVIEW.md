# 2048 Transformer Training Visualization

## Project Vision
A real-time visualization system for training a transformer-based neural network to play 2048, featuring:
- Live training progress visualization
- Real-time bot gameplay viewing
- Mobile and desktop responsive design
- GPU-accelerated training with automatic resource management
- System-agnostic deployment (CPU fallback, VRAM-aware)
- Advanced MoE architecture with dynamic routing and load balancing
- Parallel environment training for accelerated learning
- Comprehensive checkpoint management with playback capabilities

## Core Components

### 1. Neural Network Architecture
- **Model**: Transformer-based architecture optimized for 2048 game state
- **Input**: 4x4 game board state (16 values) with 2D positional encoding
- **Output**: Action probabilities (up, down, left, right) with value estimation
- **Training**: Reinforcement learning with PPO algorithm and auxiliary tasks
- **MoE Design**: Mixture of Experts with top-k routing, capacity factors, and load balancing
- **Dynamic Sizing**: Automatic configuration based on available VRAM (2GB-8GB+ support)

### 2. Training System
- **Backend**: Python-based training server with FastAPI and WebSocket support
- **GPU Management**: Automatic CUDA detection with VRAM monitoring and dynamic batch sizing
- **Resource Management**: Dynamic batch sizing, model checkpointing, and memory optimization
- **Progress Tracking**: Real-time metrics, model state serialization, and performance analytics
- **Parallel Training**: Multiple concurrent game environments for accelerated training
- **Advanced Metrics**: Training speed, game length statistics, wall clock timing, and efficiency monitoring

### 3. Visualization Frontend
- **Framework**: Modern React/TypeScript stack with real-time WebSocket updates
- **Real-time Updates**: WebSocket connection for live training data with mobile optimization
- **Game Viewer**: Interactive 2048 board showing bot decisions with playback controls
- **Training Graphs**: Loss curves, score progression, action distributions, learning rate schedules
- **Network Visualizer**: Expert usage visualization, attention heatmaps, MoE routing patterns
- **Responsive Design**: Mobile-first approach with device detection and adaptive layouts
- **Advanced Controls**: Playback speed control, pause/resume, new game generation

### 4. System Architecture
- **Server**: FastAPI with WebSocket support and comprehensive API endpoints
- **Client**: Progressive Web App (PWA) with offline capability and native app experience
- **Communication**: WebSocket for real-time data, REST for configuration and control
- **Error Handling**: Comprehensive logging, graceful degradation, and automatic recovery
- **Network Discovery**: Automatic LAN detection and QR code generation for mobile access
- **Thread Safety**: Concurrent training and WebSocket operations with proper locking

## Development Environment

### Platform Requirements
- **Primary Development**: PowerShell (Windows 10/11)
- **Linux Tools**: WSL with Debian for Linux-specific software
- **Package Management**: Poetry for Python, npm for Node.js
- **Console Commands**: All commands should be run in PowerShell unless WSL is specifically required
- **Automated Setup**: Launcher script with dependency management and network configuration

## Technical Requirements

### Performance
- GPU acceleration when available (CUDA/PyTorch) with automatic fallback
- Automatic VRAM management with dynamic model sizing
- CPU fallback for training with optimized configurations
- Efficient model serialization for checkpoints with metadata
- Parallel environment training for 100+ episodes per minute target
- Memory optimization and leak prevention

### User Experience
- Real-time training visualization with 1-2 second update frequency
- Interactive game playback controls with speed adjustment
- Mobile-responsive design with touch-optimized interfaces
- Offline capability for viewing saved checkpoints
- Native app-like PWA experience with iOS-style design
- Adaptive UI based on device capabilities and screen size

### Development Experience
- Clear error messages and comprehensive logging with colored output
- Hot-reload for development with automatic dependency installation
- Comprehensive documentation with troubleshooting guides
- Easy deployment and setup with automated launcher
- Network troubleshooting and firewall assistance
- Cross-platform compatibility and testing

## Project Requirements

### Training & Model
- **Algorithm**: PPO (Proximal Policy Optimization) with advanced hyperparameters
- **Architecture**: Large, efficient transformer with sparse/MoE design and load balancing
- **Update Frequency**: Visual feedback every 1-2 seconds maximum
- **Training Control**: Start/pause/stop with checkpoint resumption and reset capability
- **GPU Target**: RTX 3070 Ti (8GB VRAM) with dynamic sizing and CPU fallback
- **Parallel Environments**: Multiple concurrent game environments for accelerated training
- **Advanced Metrics**: Training speed, efficiency monitoring, and performance analytics
- **Auxiliary Tasks**: Next tile prediction, board evaluation, and merge opportunity detection

### Visualizations (All Required)
1. **Training Dashboard** (Priority #1)
   - Real-time loss curves with mobile-optimized display
   - Score progression with trend analysis
   - Action distributions with probability visualization
   - Learning rate schedules with adaptive display
   - Training speed and efficiency metrics
   - Game length statistics and performance indicators

2. **Game Replay Viewer**
   - Live bot gameplay with real-time board updates
   - Historical game replays with step-by-step playback
   - Action highlighting and decision explanations
   - Playback speed control and pause/resume functionality
   - New game generation from checkpoints
   - Attention overlay for transformer focus visualization

3. **Attention Heatmaps**
   - Transformer attention visualization with 4x4 grid display
   - Board position focus analysis with intensity mapping
   - Real-time attention weight updates during training
   - Mobile-optimized heatmap display
   - Interactive attention pattern exploration

4. **Network Architecture Viewer**
   - Visual representation of model structure with layer visualization
   - Dynamic sizing decisions with configuration display
   - MoE routing visualization with expert usage patterns
   - Load balancing metrics and efficiency indicators
   - Real-time expert activation and specialization display
   - Performance metrics and memory usage visualization

### Mobile Experience
- **PWA Design**: Native app-like experience with iOS-style interface
- **Aesthetics**: Mobile and desktop share the same visual style with adaptive layouts
- **Full Feature Access**: All features available on desktop are also accessible on mobile (game replay, attention maps, checkpoints, etc.)
- **No Scrolling Requirement**: The interface must fit on a mobile screen without requiring vertical scrolling
- **Compact Graphs**: All graphs and charts must be simplified and compact (e.g., trendlines instead of raw data, minimal axes/labels)
- **Touch-Friendly**: Controls and navigation must be easily usable on mobile devices
- **App Icon**: Use the provided image as the app icon for PWA and mobile home screen
- **Mobile Optimization**: Adaptive timeouts, connection pooling, and device-specific optimizations
- **Offline Capability**: View saved checkpoints and training history without network connection

### Data Management
- **Checkpoint Library**: Server-side storage with comprehensive user management
- **Naming**: Auto-generated with user-editable nicknames and tagging system
- **Metadata**: Hardcoded cutoff dates, training parameters, and performance metrics
- **Resumption**: Load any checkpoint and continue training with derivative naming
- **Derivatives**: New checkpoints get derived names from originals with inheritance tracking
- **Playback System**: Full checkpoint playback with game simulation and analysis
- **Performance Analytics**: Training speed, efficiency, and model performance tracking

### Advanced Features
- **Network Discovery**: Automatic LAN detection and QR code generation for mobile access
- **Thread Safety**: Concurrent training and WebSocket operations with proper locking
- **Error Recovery**: Graceful degradation and automatic recovery mechanisms
- **Performance Monitoring**: Real-time GPU usage, memory tracking, and efficiency metrics
- **Adaptive UI**: Device detection and screen size optimization
- **Comprehensive Logging**: Rich console output with colored status messages
- **Network Troubleshooting**: Built-in connectivity testing and firewall assistance

### Technical Preferences
- **Installation**: Direct (Anaconda acceptable for dependencies) with automated setup
- **No Docker**: Prefer native setup with cross-platform compatibility
- **No Extensibility**: Focus on 2048 only with optimized architecture
- **Goal**: Experimentation with excellent gameplay performance and comprehensive visualization
- **Launcher Integration**: Automated development environment setup and management 