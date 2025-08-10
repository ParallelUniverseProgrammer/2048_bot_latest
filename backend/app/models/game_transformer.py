"""
Game Transformer with Mixture of Experts for 2048
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any
from .model_config import ModelConfig

class PositionalEncoding(nn.Module):
    """2D positional encoding for the 4x4 board"""
    
    def __init__(self, d_model: int, max_len: int = 16):
        super().__init__()
        
        # Create position embeddings for 4x4 grid
        pe = torch.zeros(max_len, d_model)
        
        for pos in range(max_len):
            # Convert linear position to 2D coordinates
            row = pos // 4
            col = pos % 4
            
            # Use different frequencies for row and column
            for i in range(0, d_model, 4):
                # Row encoding
                pe[pos, i] = math.sin(row / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(row / (10000 ** (i / d_model)))
                
                # Column encoding
                if i + 2 < d_model:
                    pe[pos, i + 2] = math.sin(col / (10000 ** ((i + 2) / d_model)))
                if i + 3 < d_model:
                    pe[pos, i + 3] = math.cos(col / (10000 ** ((i + 3) / d_model)))
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """Multi-head attention with relative position encoding"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        # Ensure input is a regular autograd tensor (not an inference tensor)
        x = x.clone()
        
        # Generate Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Store attention weights for visualization (average across heads)
        self.attention_weights = attention.mean(dim=1).detach()
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(context)

class Expert(nn.Module):
    """Individual expert in the MoE layer"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.gelu(self.w1(x))))

class MoELayer(nn.Module):
    """Mixture of Experts layer with load balancing"""
    
    def __init__(self, 
                 d_model: int, 
                 n_experts: int, 
                 d_ff: int, 
                 top_k: int = 2, 
                 dropout: float = 0.1,
                 capacity_factor: float = 1.25,
                 noise_std: float = 0.01):
        """Mixture-of-Experts layer with:
        1. Capacity factor trick – limits tokens per expert to keep routing balanced.
        2. Load-balancing auxiliary loss stored in ``self.lb_loss`` for the optimiser.
        3. Tiny Gaussian noise added to router logits for exploration.
        """
        super().__init__()

        self.n_experts = n_experts
        self.top_k = top_k
        self.d_model = d_model

        # ENHANCED: Adaptive routing hyper-parameters for all model sizes
        # Scale capacity factor by model size to prevent starvation
        if n_experts <= 4:
            self.capacity_factor = 2.0  # More capacity for tiny models to prevent starvation
            self.noise_std = 0.05  # Higher noise for better exploration in tiny models
        elif n_experts <= 6:
            self.capacity_factor = 1.8  # Enhanced capacity for medium models
            self.noise_std = 0.03  # Moderate noise for medium models
        else:
            self.capacity_factor = 1.6  # Enhanced capacity for large models
            self.noise_std = 0.02  # Lower noise for large models but still present

        # Router network
        self.router = nn.Linear(d_model, n_experts)

        # Expert networks
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(n_experts)
        ])

        # Tracking expert usage (for UI & debugging)
        self.register_buffer('expert_usage', torch.zeros(n_experts))
        self.usage_decay = 0.99

        # Stored per-forward values (visualisation + loss)
        self.current_expert_usage = None
        self.lb_loss: Optional[torch.Tensor] = None
        # Store router importance and entropy from last forward for monitoring
        self.last_importance: Optional[torch.Tensor] = None
        self.last_entropy: Optional[torch.Tensor] = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        x_flat = x.view(-1, d_model)
        
        # Route tokens to experts (with tiny Gaussian noise for exploration)
        router_logits = self.router(x_flat)
        if self.noise_std > 0.0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std

        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Track routing concentration statistics for sparsity monitoring (best-effort)
        try:
            max_prob = router_probs.max(dim=-1).values
            self.avg_top1_prob = max_prob.mean().detach()
            if self.n_experts >= 2:
                top2_vals, _ = torch.topk(router_probs, k=min(2, self.n_experts), dim=-1)
                if top2_vals.size(-1) >= 2:
                    self.avg_top2_prob = top2_vals[..., 1].mean().detach()
                else:
                    self.avg_top2_prob = torch.tensor(0.0, device=router_probs.device)
            else:
                self.avg_top2_prob = torch.tensor(0.0, device=router_probs.device)
            self.avg_concentration = (self.avg_top1_prob - self.avg_top2_prob).detach()
        except Exception:
            self.avg_top1_prob = None
            self.avg_top2_prob = None
            self.avg_concentration = None

        # Compute capacity (number of tokens each expert can receive)
        tokens_total = x_flat.size(0)
        capacity = math.ceil(tokens_total * self.capacity_factor / self.n_experts)
        
        # Update expert usage for load balancing
        with torch.no_grad():
            usage = torch.zeros(self.n_experts, device=x.device)
            for i in range(self.n_experts):
                usage[i] = (top_k_indices == i).float().sum()
            usage = usage / usage.sum()
            # Keep expert usage on device; buffer is a registered buffer
            self.expert_usage = self.usage_decay * self.expert_usage + (1 - self.usage_decay) * usage
            # Store device tensor for visualization/metrics; callers can .tolist() as needed
            self.current_expert_usage = usage
        
        # Apply experts with capacity limiting
        # Ensure output uses the same dtype as input activations to avoid AMP dtype mismatches
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # Tokens (indices) that selected this expert in any of their top-k routes
            expert_token_mask = (top_k_indices == i).any(dim=-1)
            token_indices = torch.nonzero(expert_token_mask, as_tuple=False).squeeze(-1)

            if token_indices.numel() == 0:
                continue

            # Enforce capacity limit
            if token_indices.numel() > capacity:
                token_indices = token_indices[:capacity]

            # Run through expert
            # Clone slice to ensure it's a regular autograd tensor (not an inference tensor)
            expert_input = x_flat[token_indices].clone()
            expert_output = expert(expert_input)
            # Align dtypes for safe "index_put" under autocast (expert may be fp16 while output is fp32)
            if expert_output.dtype != output.dtype:
                expert_output = expert_output.to(output.dtype)

            # Simple assignment (weight ≈1) – keep implementation lightweight for 16-token boards
            output[token_indices] = expert_output
        
        # ---- ENHANCED: Load-balancing auxiliary loss for all model sizes ----
        # Use more sophisticated loss function for better expert utilization
        
        # 1. Switch-style importance loss (original) and monitoring signals
        importance = router_probs.mean(dim=0)  # (n_experts,)
        # Store importance and entropy for external monitoring
        self.last_importance = importance.detach()
        self.last_entropy = (-importance * torch.log(importance + 1e-8)).sum().detach()
        switch_loss = (importance * importance).sum() * self.n_experts
        
        # 2. ENHANCED: Entropy-based diversity loss
        # Encourage higher entropy in expert selection
        expert_entropy = -torch.sum(importance * torch.log(importance + 1e-8))
        max_entropy = torch.log(torch.tensor(float(self.n_experts), device=x.device))
        normalized_entropy = expert_entropy / max_entropy
        diversity_loss = 1.0 - normalized_entropy
        
        # 3. ENHANCED: Starvation penalty with model-size scaling
        # Scale starvation threshold by model size
        if self.n_experts <= 4:
            min_usage_threshold = 1.0 / (self.n_experts * 4)  # 25% of uniform distribution
            starvation_weight = 0.5
        elif self.n_experts <= 6:
            min_usage_threshold = 1.0 / (self.n_experts * 3)  # 33% of uniform distribution
            starvation_weight = 0.7
        else:
            min_usage_threshold = 1.0 / (self.n_experts * 2.5)  # 40% of uniform distribution
            starvation_weight = 0.9
        
        starvation_mask = importance < min_usage_threshold
        starvation_penalty = torch.sum(starvation_mask.float()) * starvation_weight
        
        # 4. ENHANCED: Balance quality loss with progressive scaling
        # Penalize deviation from uniform distribution
        uniform_target = 1.0 / self.n_experts
        balance_loss = torch.mean(torch.abs(importance - uniform_target)) / uniform_target
        
        # 5. NEW: Expert diversity enforcement
        # Ensure at least a minimum number of experts are actively used
        active_experts = torch.sum((importance > uniform_target * 0.1).float())
        min_active_experts = max(2, self.n_experts // 2)  # At least half the experts should be active
        diversity_enforcement = torch.relu(min_active_experts - active_experts) * 0.3
        
        # Combine losses with adaptive weights for all model sizes
        if self.n_experts <= 4:
            # Strong load balancing for tiny models
            self.lb_loss = (switch_loss * 0.3 + 
                           diversity_loss * 0.4 + 
                           starvation_penalty * 0.2 + 
                           balance_loss * 0.1)
        elif self.n_experts <= 6:
            # Enhanced load balancing for medium models
            self.lb_loss = (switch_loss * 0.25 + 
                           diversity_loss * 0.35 + 
                           starvation_penalty * 0.25 + 
                           balance_loss * 0.1 + 
                           diversity_enforcement * 0.05)
        else:
            # Strongest load balancing for large models
            self.lb_loss = (switch_loss * 0.2 + 
                           diversity_loss * 0.3 + 
                           starvation_penalty * 0.3 + 
                           balance_loss * 0.1 + 
                           diversity_enforcement * 0.1)

        return output.view(batch_size, seq_len, d_model)

class TransformerBlock(nn.Module):
    """Transformer block with attention and MoE"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(
            config.d_model, 
            config.n_heads, 
            config.attention_dropout
        )
        self.moe = MoELayer(
            config.d_model,
            config.n_experts,
            config.d_ff,
            config.top_k,
            config.dropout
        )
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        moe_out = self.moe(self.norm2(x))
        x = x + self.dropout(moe_out)
        
        return x

class GameTransformer(nn.Module):
    """Main transformer model for 2048 game"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # Input embedding for board values (0-16 for powers of 2)
        self.value_embedding = nn.Embedding(17, config.d_model)
        self.position_encoding = PositionalEncoding(config.d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output heads
        # Direction-aware policy head: combines row/column aggregation with pooled token state
        self.policy_norm = nn.LayerNorm(config.d_model)
        self.policy_mlp = nn.Sequential(
            nn.Linear(config.d_model, max(64, config.d_model // 2)),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        # Row/column summarizers to bias logits toward spatial structure of 2048
        hidden_out = max(64, config.d_model // 2)
        self.row_head = nn.Linear(hidden_out, 2)   # up/down bias
        self.col_head = nn.Linear(hidden_out, 2)   # left/right bias
        self.policy_out = nn.Linear(hidden_out, 4) # residual logits
        
        # Extrinsic value head (for game score)
        self.value_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1)  # Single value output
        )

        # Intrinsic value head (for novelty/intrinsic rewards)
        self.intrinsic_value_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Store attention weights for visualization
        self.attention_weights = None
        self.expert_usage = None
        # Store auxiliary load-balancing loss from last forward pass
        self.latest_lb_loss: Optional[torch.Tensor] = None
        # Store router signals from last MoE layer
        self.router_importance: Optional[torch.Tensor] = None
        self.router_entropy: Optional[torch.Tensor] = None
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def encode_board(self, board: torch.Tensor) -> torch.Tensor:
        """Encode 4x4 board to log2 values"""
        # Convert board values to log2 encoding
        # 0 -> 0, 2 -> 1, 4 -> 2, 8 -> 3, etc.
        encoded = torch.zeros_like(board, dtype=torch.long)
        
        # Handle non-zero values
        mask = board > 0
        if mask.any():
            # Use safe log2 calculation
            log_values = torch.log2(board[mask].float()).long()
            encoded[mask] = log_values
        
        return encoded.clamp(0, 16)  # Clamp to valid range
    
    def forward(self, board: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            board: (batch_size, 4, 4) tensor of board values
            
        Returns:
            policy_logits: (batch_size, 4) action logits
            value_ext: (batch_size, 1) extrinsic state value
            value_int: (batch_size, 1) intrinsic state value
        """
        batch_size = board.size(0)
        
        # Encode board values
        encoded_board = self.encode_board(board)
        
        # Flatten to sequence: (batch_size, 16)
        board_seq = encoded_board.view(batch_size, 16)
        
        # Embed and add positional encoding
        x = self.value_embedding(board_seq)
        x = self.position_encoding(x)
        
        total_lb_loss = x.new_tensor(0.0)
        for layer in self.layers:
            x = layer(x)
            # Accumulate load-balancing loss from MoE layer
            if hasattr(layer, 'moe') and getattr(layer.moe, 'lb_loss', None) is not None:
                total_lb_loss = total_lb_loss + layer.moe.lb_loss

        # Store summed auxiliary loss for external access (e.g. optimiser)
        self.latest_lb_loss = total_lb_loss
        
        # Store attention weights from last layer for visualization
        if self.layers:
            last_attention = self.layers[-1].attention.attention_weights
            if last_attention is not None:
                # Average across batch and heads, then reshape to 4x4
                # Shape: [batch, heads, seq_len, seq_len] -> [seq_len, seq_len] -> [4, 4]
                self.attention_weights = last_attention.mean(dim=(0, 1)).view(4, 4)
        
        # Store expert usage from last layer
        if self.layers:
            last_moe = self.layers[-1].moe.current_expert_usage
            if last_moe is not None:
                self.expert_usage = last_moe

        # Store router importance/entropy from last layer
        if self.layers:
            last_moe_layer = self.layers[-1].moe
            if getattr(last_moe_layer, 'last_importance', None) is not None:
                self.router_importance = last_moe_layer.last_importance
            if getattr(last_moe_layer, 'last_entropy', None) is not None:
                self.router_entropy = last_moe_layer.last_entropy
        
        # Global average pooling for final representation
        pooled = x.mean(dim=1)  # (batch_size, d_model)

        # Direction-aware policy computation
        h = self.policy_mlp(self.policy_norm(pooled))  # (B, H)
        # Split logits into directional components
        row_logits = self.row_head(h)  # (B, 2) -> [up, down]
        col_logits = self.col_head(h)  # (B, 2) -> [left, right]
        residual_logits = self.policy_out(h)  # (B, 4)
        # Assemble in action order [up, down, left, right]
        policy_logits = torch.stack(
            [row_logits[:, 0], row_logits[:, 1], col_logits[:, 0], col_logits[:, 1]], dim=-1
        ) + residual_logits
        value_ext = self.value_head(pooled)
        value_int = self.intrinsic_value_head(pooled)

        return policy_logits, value_ext, value_int
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights for visualization"""
        return self.attention_weights
    
    def get_expert_usage(self) -> Optional[torch.Tensor]:
        """Get expert usage for visualization"""
        return self.expert_usage

    def get_router_importance(self) -> Optional[torch.Tensor]:
        """Get per-expert importance distribution from router."""
        return self.router_importance

    def get_router_entropy(self) -> Optional[torch.Tensor]:
        """Get router entropy (scalar) from last forward."""
        return self.router_entropy
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_memory_usage(self) -> float:
        """Get GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0 

    @torch.no_grad()
    def count_parameters(self) -> int:
        """Count parameters with no autograd overhead (kept for compatibility)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)