# 2048 Transformer Training Visualization

## Project Vision
A real-time visualization system for training a transformer-based neural network to play 2048, featuring:
- Live training progress visualization
- Real-time bot gameplay viewing
- Mobile and desktop responsive design
- GPU-accelerated training with automatic resource management
- System-agnostic deployment (CPU fallback, VRAM-aware)

## Core Components

### 1. Neural Network Architecture
- **Model**: Transformer-based architecture optimized for 2048 game state
- **Input**: 4x4 game board state (16 values)
- **Output**: Action probabilities (up, down, left, right)
- **Training**: Reinforcement learning with policy gradient methods

### 2. Training System
- **Backend**: Python-based training server
- **GPU Management**: Automatic CUDA detection with VRAM monitoring
- **Resource Management**: Dynamic batch sizing and model checkpointing
- **Progress Tracking**: Real-time metrics and model state serialization

### 3. Visualization Frontend
- **Framework**: Modern web stack (React/Vue + WebSocket)
- **Real-time Updates**: WebSocket connection for live training data
- **Game Viewer**: Interactive 2048 board showing bot decisions
- **Training Graphs**: Loss curves, score progression, action distributions
- **Responsive Design**: Mobile-first approach with desktop enhancements

### 4. System Architecture
- **Server**: FastAPI/Flask with WebSocket support
- **Client**: Progressive Web App (PWA) capabilities
- **Communication**: WebSocket for real-time data, REST for configuration
- **Error Handling**: Comprehensive logging with descriptive console messages

## Development Environment

### Platform Requirements
- **Primary Development**: PowerShell (Windows 10/11)
- **Linux Tools**: WSL with Debian for Linux-specific software
- **Package Management**: Poetry for Python, npm for Node.js
- **Console Commands**: All commands should be run in PowerShell unless WSL is specifically required

## Technical Requirements

### Performance
- GPU acceleration when available (CUDA/PyTorch)
- Automatic VRAM management
- CPU fallback for training
- Efficient model serialization for checkpoints

### User Experience
- Real-time training visualization
- Interactive game playback controls
- Mobile-responsive design
- Offline capability for viewing saved checkpoints

### Development Experience
- Clear error messages and logging
- Hot-reload for development
- Comprehensive documentation
- Easy deployment and setup

## Project Requirements

### Training & Model
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Architecture**: Large, efficient transformer with sparse/MoE design
- **Update Frequency**: Visual feedback every 1-2 seconds maximum
- **Training Control**: Start/pause/stop with checkpoint resumption
- **GPU Target**: RTX 3070 Ti (8GB VRAM) with dynamic sizing

### Visualizations (All Required)
1. **Training Graphs** (Priority #1)
   - Real-time loss curves
   - Score progression
   - Action distributions
   - Learning rate schedules
2. **Game Replay Viewer**
   - Live bot gameplay
   - Historical game replays
   - Action highlighting
3. **Attention Heatmaps**
   - Transformer attention visualization
   - Board position focus analysis
4. **Network Architecture Viewer**
   - Visual representation of model structure
   - Dynamic sizing decisions
   - MoE routing visualization

### Mobile Experience
- **PWA Design**: Native app-like experience
- **Aesthetics**: Mobile and desktop share the same visual style
- **Full Feature Access**: All features available on desktop are also accessible on mobile (game replay, attention maps, checkpoints, etc.)
- **No Scrolling Requirement**: The interface must fit on a mobile screen without requiring vertical scrolling
- **Compact Graphs**: All graphs and charts must be simplified and compact (e.g., trendlines instead of raw data, minimal axes/labels)
- **Touch-Friendly**: Controls and navigation must be easily usable on mobile devices
- **App Icon**: Use the provided image as the app icon for PWA and mobile home screen (see implementation notes)

### Data Management
- **Checkpoint Library**: Server-side storage with user management
- **Naming**: Auto-generated with user-editable nicknames
- **Metadata**: Hardcoded cutoff dates, training parameters
- **Resumption**: Load any checkpoint and continue training
- **Derivatives**: New checkpoints get derived names from originals

### Technical Preferences
- **Installation**: Direct (Anaconda acceptable for dependencies)
- **No Docker**: Prefer native setup
- **No Extensibility**: Focus on 2048 only
- **Goal**: Experimentation with good gameplay performance 