# Technical Architecture

## System Overview

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   Frontend      │ ◄──────────────► │   Backend       │
│   (React/TS)    │                 │   (FastAPI)     │
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
                                              │
                                              ▼
                                     ┌─────────────────┐
                                     │ Parallel Env    │
                                     │   Manager       │
                                     └─────────────────┘
```

## Technology Stack Recommendations

### Backend (Python)
- **Framework**: FastAPI (async, WebSocket support, auto-docs, comprehensive API)
- **ML Framework**: PyTorch (GPU acceleration, transformer support, dynamic sizing)
- **RL Library**: Custom PPO implementation with advanced hyperparameters
- **WebSocket**: FastAPI WebSocket endpoints with mobile optimization
- **GPU Management**: `torch.cuda` with VRAM monitoring and dynamic configuration
- **Logging**: Structured logging with rich console output and colored status
- **Thread Safety**: Proper locking for concurrent operations
- **Network Discovery**: Automatic LAN detection and QR code generation

### Frontend (JavaScript/TypeScript)
- **Framework**: React with TypeScript (mature ecosystem, type safety)
- **State Management**: Zustand with persistence and real-time updates
- **Charts**: Chart.js with mobile-optimized display and adaptive layouts
- **WebSocket**: Native WebSocket API with adaptive timeouts, exponential backoff, circuit breaker patterns, and polling fallback
- **Styling**: Tailwind CSS (responsive, utility-first, device detection)
- **PWA**: Service workers for offline capability and native app experience
- **Animations**: Framer Motion for smooth 60fps transitions
- **Device Detection**: Advanced mobile/desktop detection with adaptive UI
- **Loading States**: Enhanced progress tracking with step-by-step feedback and estimated completion times

### Development Tools
- **Package Manager**: Poetry (Python) / npm/yarn (Node.js)
- **Linting**: Black, isort, flake8 (Python) / ESLint (JS)
- **Testing**: pytest (Python) / Jest (JS)
- **Hot Reload**: FastAPI reload / Vite (React)
- **Launcher**: Automated setup with dependency management
- **Network Tools**: QR code generation and connectivity testing

## Key Implementation Decisions

### 1. Advanced Model Architecture
```python
# MoE-based transformer structure with load balancing and dynamic sizing
class GameTransformer(nn.Module):
    def __init__(self, d_model=512, n_heads=16, n_layers=8, n_experts=8, top_k=2):
        # Input: 4x4 board → 16 tokens with 2D positional encoding
        # MoE layers with sparse routing, capacity factors, and load balancing
        # Dynamic sizing based on VRAM availability (2GB-8GB+ support)
        # Output: 4 action probabilities with value head and auxiliary tasks
        # Thread-safe operations with proper memory management
```

### 2. Advanced Training Strategy
- **Algorithm**: PPO (Proximal Policy Optimization) with advanced hyperparameters
- **Environment**: Custom 2048 gym environment with parallel execution
- **Reward**: Score-based with survival bonus, efficiency rewards, and auxiliary tasks
- **Batch Size**: Dynamic based on available VRAM with memory optimization
- **Update Frequency**: Every 1-2 seconds for visual feedback with mobile optimization
- **Checkpointing**: Every N episodes with comprehensive metadata and performance tracking
- **MoE Routing**: Sparse expert selection with load balancing and auxiliary loss
- **Parallel Training**: Multiple concurrent environments for accelerated learning

### 3. Real-time Communication with Mobile Optimization
```python
# WebSocket message structure with adaptive timeouts and mobile optimization
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
    "attention_weights": [...],  # For heatmaps with 4x4 grid display
    "expert_usage": [0.12, 0.08, 0.15, ...],  # MoE routing with load balancing
    "gpu_memory": 6.2,  # GB used with dynamic monitoring
    "model_params": 45.6,  # Million parameters with efficiency metrics
    "training_speed": 65.5,  # Episodes per minute
    "game_length_stats": {...},  # Performance analytics
    "wall_clock_elapsed": 1277.5,  # Training duration tracking
}
```

**Enhanced Features:**
- **Adaptive Timeouts**: Device-specific timeout management with mobile optimization
- **Exponential Backoff**: Intelligent reconnection with increasing delays
- **Circuit Breaker**: Prevents cascading failures with automatic recovery
- **Polling Fallback**: Graceful degradation to HTTP polling when WebSocket fails
- **Connection Health Monitoring**: Real-time connection quality assessment
- **Mobile Network Optimization**: Adaptive behavior for different network conditions

### 4. Advanced Resource Management
- **VRAM Monitoring**: `torch.cuda.memory_allocated()` with dynamic adjustment
- **Dynamic Batching**: Reduce batch size if OOM with memory optimization
- **Model Checkpointing**: Save/load with comprehensive metadata and performance tracking
- **CPU Fallback**: Automatic detection and switching with optimized configurations
- **Memory Management**: Leak prevention and efficient tensor operations
- **Parallel Processing**: Multiple environments with thread-safe operations

### 5. Mobile PWA Architecture
```javascript
// Progressive Web App with native app experience
{
          "name": "2048 Bot Training",
  "short_name": "2048 AI",
  "display": "standalone",
  "orientation": "portrait",
  "theme_color": "#3b82f6",
  "background_color": "#0f172a",
  "icons": [...],  // Multiple sizes for different devices
  "service_worker": {
    "offline_capability": true,
    "checkpoint_viewing": true,
    "caching_strategy": "CacheFirst"
  }
}
```

### 6. Enhanced Loading State Management
```typescript
// Comprehensive loading state interface with progress tracking
interface LoadingStates {
  isTrainingStarting: boolean
  isPlaybackStarting: boolean
  isNewGameStarting: boolean
  isTrainingStopping: boolean
  isTrainingResetting: boolean
  loadingMessage: string | null
  loadingProgress: number // 0-100
  loadingStep: string | null // Current step description
  loadingSteps: string[] // All steps for this operation
  currentStepIndex: number // Current step index
  estimatedTimeRemaining: number | null // Estimated seconds remaining
  startTime: number | null // When the operation started
  progressInterval: number | null // Interval ID for progress simulation
}
```

**Features:**
- **Progress Tracking**: Real-time progress bars with smooth animations
- **Step-by-Step Feedback**: Detailed descriptions of current operations
- **Estimated Completion**: Time remaining estimates for long operations
- **Visual Indicators**: Progress bars, step indicators, and completion status
- **Operation-Specific Steps**: Tailored step sequences for different operations
- **Graceful Completion**: Smooth transition from loading to active state

## Development Phases

### Phase 1: Core Infrastructure
1. Advanced 2048 game environment with parallel execution
2. Sophisticated MoE transformer model with load balancing
3. FastAPI backend with WebSocket and comprehensive API
4. React/TypeScript frontend with PWA support

### Phase 2: Advanced Training System
1. PPO training loop with parallel environments
2. GPU management with dynamic sizing and optimization
3. Real-time metrics with mobile optimization
4. Model checkpointing with comprehensive metadata

### Phase 3: Comprehensive Visualization
1. Advanced training graphs with mobile optimization
2. Sophisticated game replay viewer with playback controls
3. Network architecture viewer with expert usage patterns
4. Attention heatmaps with 4x4 grid display

### Phase 4: Mobile PWA & Advanced UI
1. Native app-like PWA experience with iOS-style design
2. Device detection and adaptive layouts
3. Touch-friendly controls with framer-motion animations
4. Network discovery and QR code generation

### Phase 5: Advanced Checkpoint Management
1. Comprehensive checkpoint system with playback capability
2. Performance analytics and training speed tracking
3. Historical comparison and trend analysis
4. Export capabilities for training data

### Phase 6: Performance Optimization & Polish
1. Advanced error handling with graceful degradation
2. Memory optimization and leak prevention
3. Cross-platform compatibility and testing
4. Comprehensive documentation and troubleshooting

## Advanced Features

### 1. Network Discovery & Mobile Access
```python
# Automatic LAN detection and QR code generation
class NetworkManager:
    def discover_local_ip(self):
        # Find best local IP for LAN access
        # Generate QR code for mobile access
        # Configure CORS for secure cross-origin requests
        # Set up WebSocket connections with mobile optimization
```

### 2. Thread-Safe Operations
```python
# Concurrent training and WebSocket operations
class ThreadSafeManager:
    def __init__(self):
        self._buffer_lock = threading.Lock()
        self._training_lock = threading.Lock()
    
    def safe_operation(self):
        with self._buffer_lock:
            # Thread-safe buffer operations
            pass
```

### 3. Advanced Device Detection
```typescript
// Intelligent device detection with adaptive UI
interface DeviceInfo {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  orientation: 'portrait' | 'landscape';
  screenWidth: number;
  screenHeight: number;
}

function getDisplayMode(): DisplayMode {
  // Mobile devices always use mobile mode
  // Tablets use mobile mode in portrait, desktop in landscape
  // Desktop devices always use desktop mode
}
```

### 4. Performance Monitoring
```python
# Real-time performance metrics and efficiency tracking
class PerformanceMonitor:
    def track_metrics(self):
        return {
            "training_speed": episodes_per_minute,
            "gpu_utilization": gpu_usage_percentage,
            "memory_efficiency": memory_usage_optimization,
            "load_balancing": expert_usage_variance,
            "wall_clock_time": elapsed_training_time
        }
```

## Questions for Technical Decisions

1. **Model Complexity**: Should we implement auxiliary tasks (next tile prediction, board evaluation) for improved performance?

2. **Training Frequency**: How should we optimize update frequency for different device types (mobile vs desktop)?

3. **Data Storage**: Should we implement database storage for experiment tracking, or is file-based storage sufficient?

4. **Deployment**: Should we implement Docker containers for easier deployment, or maintain direct installation?

5. **Mobile Performance**: How should we implement different visualization fidelity for mobile vs desktop?

6. **Security**: Should we implement authentication for multi-user environments, or maintain local-only access?

7. **Extensibility**: Should we design for easy addition of other games or RL algorithms?

8. **Monitoring**: What level of system monitoring and performance analytics should we implement?

9. **Offline Capability**: How comprehensive should the offline PWA functionality be?

10. **Network Optimization**: How should we handle network discovery and mobile connectivity issues? 