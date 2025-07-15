# Technical Architecture

## System Overview

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   Frontend      │ ◄──────────────► │   Backend       │
│   (React/Vue)   │                 │   (FastAPI)     │
└─────────────────┘                 └─────────────────┘
         │                                   │
         │                                   │
         ▼                                   ▼
┌─────────────────┐                 ┌─────────────────┐
│   PWA Cache     │                 │   Training      │
│   (Offline)     │                 │   Engine        │
└─────────────────┘                 └─────────────────┘
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │   GPU Manager   │
                                     │   (CUDA/CPU)    │
                                     └─────────────────┘
```

## Technology Stack Recommendations

### Backend (Python)
- **Framework**: FastAPI (async, WebSocket support, auto-docs)
- **ML Framework**: PyTorch (GPU acceleration, transformer support)
- **RL Library**: Stable Baselines3 or custom implementation
- **WebSocket**: FastAPI WebSocket endpoints
- **GPU Management**: `torch.cuda` with VRAM monitoring
- **Logging**: Structured logging with rich console output

### Frontend (JavaScript/TypeScript)
- **Framework**: React with TypeScript (mature ecosystem)
- **State Management**: Zustand or Redux Toolkit
- **Charts**: Chart.js or D3.js for training visualizations
- **WebSocket**: Native WebSocket API or Socket.io
- **Styling**: Tailwind CSS (responsive, utility-first)
- **PWA**: Service workers for offline capability

### Development Tools
- **Package Manager**: Poetry (Python) / npm/yarn (Node.js)
- **Linting**: Black, isort, flake8 (Python) / ESLint (JS)
- **Testing**: pytest (Python) / Jest (JS)
- **Hot Reload**: FastAPI reload / Vite (React)

## Key Implementation Decisions

### 1. Model Architecture
```python
# MoE-based transformer structure for efficiency
class GameTransformer(nn.Module):
    def __init__(self, d_model=512, n_heads=16, n_layers=8, n_experts=8, top_k=2):
        # Input: 4x4 board → 16 tokens with position encoding
        # MoE layers with sparse routing
        # Dynamic sizing based on VRAM availability
        # Output: 4 action probabilities with value head
```

### 2. Training Strategy
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Environment**: Custom 2048 gym environment
- **Reward**: Score-based with survival bonus and efficiency rewards
- **Batch Size**: Dynamic based on available VRAM (target: 8GB RTX 3070 Ti)
- **Update Frequency**: Every 1-2 seconds for visual feedback
- **Checkpointing**: Every N episodes with full metadata
- **MoE Routing**: Sparse expert selection with load balancing

### 3. Real-time Communication
```python
# WebSocket message structure (1-2 second updates)
{
    "type": "training_update",
    "episode": 1234,
    "score": 2048,
    "loss": 0.123,
    "policy_loss": 0.045,
    "value_loss": 0.078,
    "entropy": 0.234,
    "learning_rate": 0.0003,
    "actions": [0.25, 0.3, 0.2, 0.25],
    "board_state": [[2,4,8,16], ...],
    "attention_weights": [...],  # For heatmaps
    "expert_usage": [0.12, 0.08, 0.15, ...],  # MoE routing
    "gpu_memory": 6.2,  # GB used
    "model_params": 45.6  # Million parameters
}
```

### 4. Resource Management
- **VRAM Monitoring**: `torch.cuda.memory_allocated()`
- **Dynamic Batching**: Reduce batch size if OOM
- **Model Checkpointing**: Save/load with metadata
- **CPU Fallback**: Automatic detection and switching

## Development Phases

### Phase 1: Core Infrastructure
1. Basic 2048 game environment
2. Simple transformer model
3. FastAPI backend with WebSocket
4. Basic React frontend

### Phase 2: Training System
1. RL training loop
2. GPU management
3. Real-time metrics
4. Model checkpointing

### Phase 3: Visualization
1. Training graphs
2. Game replay viewer
3. Mobile responsiveness
4. PWA features

### Phase 4: Polish
1. Error handling
2. Performance optimization
3. Documentation
4. Deployment setup

## Questions for Technical Decisions

1. **Model Complexity**: Should we start with a simple transformer or go straight to a more sophisticated architecture?

2. **Training Frequency**: How often should we send updates to the frontend? Every episode, every N steps?

3. **Data Storage**: Do we need a database for experiment tracking, or is file-based storage sufficient?

4. **Deployment**: Docker containers, or direct installation? Any specific cloud platform preferences?

5. **Mobile Performance**: Should we implement different visualization fidelity for mobile vs desktop?

6. **Security**: Any authentication needed, or is this purely local development?

7. **Extensibility**: Should we design for easy addition of other games or RL algorithms?

8. **Monitoring**: What level of system monitoring do you want? GPU usage, training metrics, etc.? 