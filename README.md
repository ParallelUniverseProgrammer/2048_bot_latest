# 2048 Transformer Training

A real-time visualization system for training a MoE (Mixture of Experts) transformer-based neural network to play 2048. Features live training progress visualization, interactive game replay, attention heatmaps, and expert usage monitoring.

## Features

### 🧠 **MoE Transformer Architecture**
- **Dynamic routing** based on input complexity
- **Load balancing** to prevent expert starvation
- **Sparse computation** for efficiency
- **Adaptive sizing** based on available VRAM

### 📊 **Real-time Visualizations**
- **Training Dashboard** with live metrics
- **Network Architecture Viewer** showing expert usage and attention patterns
- **Game Replay** with live 2048 board and action history
- **Attention Heatmaps** for transformer focus analysis

### 🎮 **Training System**
- **PPO Algorithm** with policy gradient optimization
- **Reward shaping** for better learning
- **Checkpoint management** with resumption capability
- **GPU acceleration** with automatic VRAM management

### 📱 **Mobile PWA**
- **iOS-style design** with native app feel
- **Offline capability** for viewing saved checkpoints
- **Responsive design** for all screen sizes
- **Simplified mobile interface** for training monitoring

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- CUDA-compatible GPU (optional, will fall back to CPU)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd 2048_bot_cursor_pro
   ```

2. **Install Python dependencies**
   ```bash
   cd backend
   poetry install
   ```

3. **Install Node.js dependencies**
   ```bash
   cd ../frontend
   npm install
   ```

### Running the Application

1. **Start the backend server**
   ```bash
   cd backend
   poetry run python main.py
   ```
   The server will start on `http://localhost:8000`

2. **Start the frontend development server**
   ```bash
   cd frontend
   npm run dev
   ```
   The frontend will start on `http://localhost:3000`

3. **Open your browser**
   Navigate to `http://localhost:3000` to access the training dashboard

## Usage

### Starting Training

1. **Connect to the server** - The frontend will automatically attempt to connect
2. **Click "Start Training"** - This will initialize the model and begin training
3. **Monitor progress** - Watch real-time updates of training metrics
4. **Control training** - Use pause/resume/stop buttons as needed

### Visualizations

#### Training Dashboard
- **Connection status** and training controls
- **Live metrics** including episode count, scores, and rewards
- **GPU memory usage** and model parameters
- **Recent rewards chart** showing training progress

#### Network Architecture Viewer
- **Expert usage visualization** showing which experts are active
- **Attention heatmaps** displaying transformer focus patterns
- **Real-time updates** as the model processes game states

#### Game Replay
- **Live 2048 board** showing current game state
- **Action history** displaying recent bot decisions
- **Tile animations** and color-coded values

### Checkpoint Management

- **Automatic saving** every 100 episodes
- **Manual saving** via API endpoints
- **Checkpoint library** with metadata
- **Training resumption** from any checkpoint

## Architecture

### Backend (Python/FastAPI)
```
backend/
├── app/
│   ├── environment/     # 2048 game environment
│   ├── models/         # MoE transformer implementation
│   ├── training/       # PPO trainer and training loop
│   ├── utils/          # GPU manager and utilities
│   └── api/           # FastAPI server and WebSocket
├── checkpoints/       # Saved model checkpoints
└── main.py           # Server entry point
```

### Frontend (React/TypeScript)
```
frontend/
├── src/
│   ├── components/    # React components
│   ├── stores/        # Zustand state management
│   └── utils/         # Utility functions
├── public/           # Static assets
└── package.json      # Dependencies
```

## Model Architecture

### MoE Transformer Design
- **Input**: 4x4 board as log2 values with positional encoding
- **Layers**: 4-8 transformer layers with MoE routing
- **Experts**: 4-8 general-purpose experts with load balancing
- **Output**: Policy head (actions), value head (state evaluation), auxiliary heads

### Dynamic Sizing
- **Small**: 4 layers, 4 experts, 256 dims (2GB VRAM)
- **Medium**: 6 layers, 6 experts, 384 dims (4GB VRAM)
- **Large**: 8 layers, 8 experts, 512 dims (6GB VRAM)

### Training Features
- **PPO algorithm** with clipped objective
- **Layer-wise learning rates** for different components
- **Progressive auxiliary task weighting**
- **Mixed precision training** for efficiency

## API Endpoints

### REST API
- `GET /` - Server status
- `GET /health` - Health check
- `GET /model/config` - Model configuration
- `POST /training/start` - Start training
- `POST /training/stop` - Stop training
- `POST /training/pause` - Pause training
- `POST /training/resume` - Resume training
- `GET /checkpoints` - List checkpoints
- `POST /checkpoints/save` - Save checkpoint
- `POST /checkpoints/load/{id}` - Load checkpoint

### WebSocket
- `ws://localhost:8000/ws` - Real-time training updates

## Development

### Backend Development
```bash
cd backend
poetry install
poetry run python main.py
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

### Code Quality
```bash
# Backend
cd backend
poetry run black .
poetry run isort .
poetry run flake8

# Frontend
cd frontend
npm run lint
```

## Performance

### Training Speed
- **Target**: 100+ episodes per minute
- **Update frequency**: Every 1-2 seconds
- **Memory usage**: <7GB VRAM on RTX 3070 Ti

### Visualization Performance
- **60fps animations** for smooth experience
- **Real-time updates** via WebSocket
- **Mobile optimization** for responsive design

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**
   - The system will automatically fall back to CPU training
   - Check available VRAM with `nvidia-smi`
   - Reduce model size in GPU manager configuration

2. **WebSocket Connection Issues**
   - Ensure backend server is running on port 8000
   - Check firewall settings
   - Verify CORS configuration

3. **Training Not Starting**
   - Check GPU availability and CUDA installation
   - Verify all dependencies are installed
   - Check server logs for error messages

### Logs
- **Backend logs**: `training_server.log`
- **Frontend logs**: Browser developer console
- **Training metrics**: Stored in checkpoint metadata

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the original 2048 game
- Built with modern AI/ML technologies
- Designed for educational and research purposes 