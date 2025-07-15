# Network Architecture Specification

## Core Design Principles

### Sparsity-First Approach
- **Dynamic routing** based on input complexity and game stage
- **Load balancing** to prevent expert underutilization
- **Efficient computation** with minimal parameter usage
- **Scalable design** from small to large based on VRAM

## Model Architecture

### Input Representation
```python
# Board state encoding
board_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
log2_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Input: 4x4 board → 16 tokens
# Each token: log2(value) + relative_position_embedding
# Empty cells: special token (value 0)
```

### Position Encoding
```python
# Relative positional encoding
# Each cell gets embedding based on its (row, col) position
# Relative distances between cells for attention mechanism
# 2D sinusoidal encoding for spatial relationships
```

### MoE Layer Design
```python
class DynamicMoELayer(nn.Module):
    def __init__(self, d_model, n_experts, min_experts=2, max_experts=8):
        # Dynamic expert selection based on input complexity
        # Load balancing to ensure even expert utilization
        # Sparse routing with top-k selection (k varies by complexity)
        
    def forward(self, x, routing_weights=None):
        # Calculate input complexity score
        # Select appropriate number of experts (2-8)
        # Route tokens to experts with load balancing
        # Combine expert outputs
```

### Expert Specialization
```python
class GameExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        # General-purpose experts that learn specialization
        # Each expert can handle any game pattern
        # Model learns to route based on input characteristics
        # No hard-coded specialization - emergent from training
```

### Multi-Head Attention
```python
class GameAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        # Standard multi-head attention
        # Relative positional encoding
        # Attention weights for visualization
        # Efficient implementation for 16-token sequences
```

## Output Heads

### 1. Policy Head (Action Selection)
```python
class PolicyHead(nn.Module):
    def __init__(self, d_model):
        # Attention-based policy network
        # Output: 4 action probabilities (up, down, left, right)
        # Softmax over actions
        # Temperature scaling for exploration
```

### 2. Value Head (State Evaluation)
```python
class ValueHead(nn.Module):
    def __init__(self, d_model):
        # Attention-based value estimation
        # Output: scalar value for current state
        # Used for advantage calculation in PPO
```

### 3. Auxiliary Heads
```python
class AuxiliaryHeads(nn.Module):
    def __init__(self, d_model):
        # Next tile prediction (2, 4 probabilities)
        # Board evaluation (win probability, game stage)
        # Merge opportunity detection
        # Space utilization score
```

## Dynamic Sizing Strategy

### VRAM-Based Scaling
```python
class DynamicModelConfig:
    def __init__(self, available_vram_gb):
        # Small config (2-4GB): 4 layers, 4 experts, 256 dims
        # Medium config (4-6GB): 6 layers, 6 experts, 384 dims  
        # Large config (6-8GB): 8 layers, 8 experts, 512 dims
        # Start small, scale up if VRAM available
```

### Scaling Parameters
- **Layers**: 4 → 6 → 8
- **Experts**: 4 → 6 → 8
- **Embedding dim**: 256 → 384 → 512
- **Attention heads**: 8 → 12 → 16
- **FFN dim**: 1024 → 1536 → 2048

## Training Optimizations

### Efficiency Techniques
```python
# Mixed precision training (FP16)
# Gradient checkpointing for memory efficiency
# Gradient accumulation for larger effective batches
# Layer-wise learning rates (deeper layers learn slower)
# Expert load balancing loss
```

### Regularization
```python
# Dropout: 0.1 for attention, 0.2 for FFN
# Layer normalization: pre-norm architecture
# Weight decay: 1e-4 for stability
# Expert diversity loss: encourage different expert behaviors
# Attention dropout: 0.1 for regularization
```

## Dynamic Routing Algorithm

### Complexity Detection
```python
def calculate_complexity(board_state):
    # Count non-empty cells
    # Measure value distribution
    # Detect game stage (early/mid/late)
    # Calculate entropy of board state
    # Return complexity score (0-1)
```

### Expert Selection
```python
def select_experts(complexity_score, n_available_experts):
    # Low complexity: 2-3 experts
    # Medium complexity: 4-6 experts  
    # High complexity: 6-8 experts
    # Load balancing: track expert usage
    # Prevent expert starvation
```

### Load Balancing
```python
def balance_expert_usage(expert_usage_history, routing_weights):
    # Track expert utilization over time
    # Penalize underutilized experts
    # Adjust routing weights for balance
    # Maintain routing quality while ensuring fairness
```

## Visualization Architecture

### Network Viewer
```python
class NetworkVisualizer:
    def __init__(self):
        # Beautiful, aesthetic visualization
        # Grounded in actual model structure
        # Real-time expert usage display
        # Attention pattern visualization
        # Gradient flow visualization
        # Interactive exploration
```

### Visual Elements
- **Expert nodes**: Size based on utilization, color by specialization
- **Connection lines**: Thickness based on attention weights
- **Attention heatmaps**: 4x4 grid showing focus patterns
- **Routing decisions**: Real-time expert selection visualization
- **Training metrics**: Loss curves, accuracy, efficiency

### Aesthetic Priorities
- **Smooth animations**: 60fps transitions
- **Color schemes**: Consistent, accessible palettes
- **Layout**: Clean, uncluttered design
- **Interactivity**: Hover effects, zoom, pan
- **Mobile optimization**: Responsive design

## Implementation Details

### Memory Management
```python
# VRAM monitoring and dynamic adjustment
# Efficient tensor operations
# Gradient accumulation for large batches
# Checkpointing for long training runs
# CPU fallback when GPU memory exhausted
```

### Performance Targets
- **Training speed**: 100+ episodes per minute
- **Memory usage**: <7GB VRAM on RTX 3070 Ti
- **Update frequency**: Every 1-2 seconds
- **Visualization lag**: <100ms

### Scalability
- **Start small**: 4 layers, 4 experts, 256 dims
- **Scale up**: Based on available VRAM
- **Fallback**: CPU training if needed
- **Efficiency**: Optimize for 2048-specific patterns 