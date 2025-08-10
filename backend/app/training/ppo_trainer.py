"""
PPO (Proximal Policy Optimization) trainer for 2048 game
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
import time
import threading
import os
import json
import asyncio
from datetime import datetime
import hashlib
import statistics

from app.models import GameTransformer, DynamicModelConfig
from app.environment.gym_2048_env import Gym2048Env
from app.utils.action_selection import select_action_from_logits_with_validation
from torch.cuda.amp import GradScaler

class TimingLogger:
    """Comprehensive timing logger for performance diagnostics"""
    
    def __init__(self, log_file: str = "training_timing.log"):
        self.log_file = log_file
        self.timings = {}
        self.current_operation = None
        self.operation_start_time = None
        self.lock = threading.Lock()
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        
        # Write header to log file
        with open(self.log_file, 'w') as f:
            f.write(f"=== Training Timing Log - Started at {datetime.now().isoformat()} ===\n")
            f.write("timestamp,operation,phase,duration_ms,details\n")
    
    def start_operation(self, operation: str, phase: str = "main", details: str = ""):
        """Start timing an operation"""
        with self.lock:
            self.current_operation = operation
            self.operation_start_time = time.perf_counter()
            self._log_event("START", operation, phase, 0, details)
    
    def end_operation(self, operation: str, phase: str = "main", details: str = ""):
        """End timing an operation and log the duration"""
        with self.lock:
            if self.current_operation == operation and self.operation_start_time is not None:
                duration_ms = (time.perf_counter() - self.operation_start_time) * 1000
                self._log_event("END", operation, phase, duration_ms, details)
                
                # Store timing for summary
                key = f"{operation}_{phase}"
                if key not in self.timings:
                    self.timings[key] = []
                self.timings[key].append(duration_ms)
                
                self.current_operation = None
                self.operation_start_time = None
    
    def log_event(self, operation: str, phase: str = "main", duration_ms: float = 0, details: str = ""):
        """Log a single timing event"""
        with self.lock:
            self._log_event("EVENT", operation, phase, duration_ms, details)
    
    def _log_event(self, event_type: str, operation: str, phase: str, duration_ms: float, details: str):
        """Internal method to write to log file"""
        # Re-enabled: write timing logs for diagnostics
        timestamp = datetime.now().isoformat()
        log_line = f"{timestamp},{event_type}_{operation},{phase},{duration_ms:.2f},{details}\n"
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_line)
        except Exception as e:
            print(f"Warning: Could not write to timing log: {e}")
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing summary statistics"""
        summary = {}
        
        with self.lock:
            for key, durations in self.timings.items():
                if durations:
                    summary[key] = {
                        'count': len(durations),
                        'total_ms': sum(durations),
                        'avg_ms': sum(durations) / len(durations),
                        'min_ms': min(durations),
                        'max_ms': max(durations),
                        'median_ms': sorted(durations)[len(durations)//2]
                    }
        
        return summary
    
    def write_summary(self, filename: str = "timing_summary.json"):
        """Write timing summary to JSON file"""
        summary = self.get_summary()
        
        # Add metadata
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'log_file': self.log_file,
            'summary': summary
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(summary_data, f, indent=2)
            print(f"Timing summary written to {filename}")
        except Exception as e:
            print(f"Warning: Could not write timing summary: {e}")

class PPOTrainer:
    """PPO trainer for the GameTransformer model"""
    
    def __init__(self, config=None, learning_rate: float = 3e-5, device=None, *, validate_moves: bool = False, compile_model: bool = False):
        """Initialize PPO trainer"""
        
        # Initialize timing logger
        self.timing_logger = TimingLogger("logs/ppo_training_timing.log")
        self.timing_logger.start_operation("trainer_init", "setup", f"config={config}, lr={learning_rate}")
        
        # Model configuration
        if config is None:
            self.timing_logger.start_operation("config_selection", "setup")
            config = DynamicModelConfig.select_config()  # Use automatic VRAM detection
            self.timing_logger.end_operation("config_selection", "setup", f"selected_config={config}")
        
        self.config = config
        self.device = device or DynamicModelConfig.get_device()
        
        # Initialize model
        self.timing_logger.start_operation("model_creation", "setup", f"device={self.device}")
        # Enable high-precision TF32 matmul on Ampere+ for speed
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        self.model = GameTransformer(config).to(self.device)
        # Optionally compile model for throughput (off by default). Enable by setting
        # ENABLE_TORCH_COMPILE=1 in the environment or passing compile_model=True.
        try:
            enable_compile = compile_model or (os.getenv("ENABLE_TORCH_COMPILE", "0") == "1")
        except Exception:
            enable_compile = compile_model
        if enable_compile:
            try:
                # Best-effort: suppress errors and fall back to eager on failure
                try:
                    import torch._dynamo as _dynamo  # type: ignore
                    _dynamo.config.suppress_errors = True
                except Exception:
                    pass
                self.model = torch.compile(self.model, mode="max-autotune")  # type: ignore[attr-defined]
            except Exception:
                pass
        self.timing_logger.end_operation("model_creation", "setup", f"model_params={self.model.count_parameters():,}")
        
        # Initialize optimizer
        self.timing_logger.start_operation("optimizer_setup", "setup")
        # Allow caller to pass a profile-aware default LR; if not provided, try derive one
        try:
            if learning_rate == 3e-5:  # legacy default in signature
                # Use module-level DynamicModelConfig to avoid shadowing
                learning_rate = float(DynamicModelConfig.get_default_learning_rate(config))
        except Exception:
            pass
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)
        # Cosine decay scheduler for smoother optimisation
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100_000, eta_min=1e-6
        )
        self.timing_logger.end_operation("optimizer_setup", "setup")
        
        # PPO hyperparameters
        # Profile-aware baselines; annealed over intrinsic_weight_decay_updates
        profile = getattr(self.config, "model_size", None) or "base"
        self.profile = profile
        # Clipping and target KL
        self.clip_epsilon = 0.2
        self.clip_epsilon_final = 0.1
        self.target_kl = 0.02 if profile != "expert" else 0.03
        # Value loss
        self.value_loss_coef = 0.5
        # Entropy schedule: reduce exploration on tiny models, moderate on base, slightly lower on expert
        if profile == "lightning":
            self.entropy_coef = 0.01
            self.entropy_coef_final = 0.002
        elif profile == "expert":
            self.entropy_coef = 0.02
            self.entropy_coef_final = 0.004
        else:  # base
            self.entropy_coef = 0.015
            self.entropy_coef_final = 0.003
        # Auxiliary load-balancing loss kept but small for stability
        if profile == "lightning":
            self.lb_loss_coef = 0.0  # ablate for tiny to reduce interference
        elif profile == "expert":
            self.lb_loss_coef = 0.002
        else:
            self.lb_loss_coef = 0.001
        self.max_grad_norm = 0.5 if profile != "lightning" else 0.4

        # Adaptive schedules
        self._clip_scale = 1.0
        self._entropy_scale = 1.0
        
        # Load balancing reward parameters — remove from intrinsic stream on small models
        if profile == "lightning":
            self.lb_reward_coef = 0.0
        elif profile == "base":
            self.lb_reward_coef = 0.1
        else:  # expert
            self.lb_reward_coef = 0.05
        # ENHANCED: Scale critical threshold by model size to prevent starvation
        base_threshold = 0.15
        if self.config.n_experts <= 4:
            # Tiny/small models: use current threshold
            self.lb_critical_threshold = max(base_threshold, 1.0 / (self.config.n_experts * 2))
        elif self.config.n_experts <= 6:
            # Medium models: higher threshold to prevent starvation
            self.lb_critical_threshold = max(base_threshold, 1.0 / (self.config.n_experts * 1.5))
        else:
            # Large models: even higher threshold
            self.lb_critical_threshold = max(base_threshold, 1.0 / (self.config.n_experts * 1.2))
        
        # ENHANCED: Scale early training boost by model size
        if profile == "lightning":
            self.lb_early_training_boost = 0.0
            self.lb_episode_threshold = 0
        elif profile == "base":
            self.lb_early_training_boost = 0.5
            self.lb_episode_threshold = 1500
        else:
            self.lb_early_training_boost = 1.0
            self.lb_episode_threshold = 2000
        
        # ENHANCED: Improved adaptive factor that scales properly with model size
        # Instead of reducing strength for larger models, maintain or increase it
        if profile == "lightning":
            self.lb_adaptive_factor = 1.0
        elif profile == "base":
            self.lb_adaptive_factor = 1.3
        else:
            self.lb_adaptive_factor = 1.5
        
        # NEW: Progressive load balancing intensity
        # Increase load balancing strength over time for larger models
        self.lb_progressive_factor = 1.0
        self.lb_progressive_episodes = 1000  # Start progressive scaling after 1000 episodes
        
        # NEW: Expert recovery tracking
        self.expert_recovery_history = deque(maxlen=50)  # Track expert recovery patterns
        self.starved_experts_tracker = {}  # Track which experts are consistently starved
        
        # New: Expert diversity tracking
        self.expert_usage_history = deque(maxlen=100)  # Track recent usage patterns
        self.lb_diversity_penalty = 0.2  # Penalty for low diversity
        self.ppo_epochs = 4
        # Use more aggressive batch sizing when ample VRAM is available
        self.batch_size = DynamicModelConfig.get_batch_size(config)
        # Small stability tweaks per profile
        if profile == "lightning":
            self.ppo_epochs = 4  # give tiny models a bit more learning per update
        elif profile == "expert":
            self.ppo_epochs = 3  # reduce per-update step to avoid large KL jumps

        # ---- NEW: lock for thread-safe experience buffer operations ----
        self._buffer_lock = threading.Lock()
        # NEW: serialize policy updates to avoid concurrent optimizer steps on shared trainer
        self._update_lock = threading.Lock()

        # Mixed precision configuration with safe fallback
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

        # Action validation toggle (avoid expensive deep copies by default)
        self.validate_moves = validate_moves

        # Novelty-driven exploration settings (intrinsic signal)
        self._novelty_counts = {}
        self._novelty_order = deque(maxlen=200_000)
        self.novelty_bonus_coef = 0.1
        self.novelty_min = 0.0
        self.novelty_max = 1.0
        self._stagnation_steps = 0
        self._stagnation_threshold_steps = 10
        # Precompute novelty lookup for 0..65536 (tile values)
        try:
            self._novelty_lookup = np.zeros(65537, dtype=np.uint8)
            for v in range(1, 65537):
                # For powers of two typical in 2048: exponent is bit_length - 1
                self._novelty_lookup[v] = (v.bit_length() - 1)
        except Exception:
            # Fallback: small table to at least cover common range
            self._novelty_lookup = np.zeros(1025, dtype=np.uint8)
            for v in range(1, 1025):
                self._novelty_lookup[v] = (v.bit_length() - 1)

        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.best_score = 0
        self.update_steps = 0
        # Intrinsic blending schedule (anneal to zero over updates)
        # Reduce intrinsic interference on smaller models, decay faster
        if profile == "lightning":
            self.intrinsic_weight_initial = 0.1
            self.intrinsic_weight_decay_updates = 600
        elif profile == "base":
            self.intrinsic_weight_initial = 0.2
            self.intrinsic_weight_decay_updates = 1200
        else:
            self.intrinsic_weight_initial = 0.25
            self.intrinsic_weight_decay_updates = 1600

        # Profile-aware discounting for extrinsic/intrinsic streams
        if profile == "lightning":
            self.gamma_ext, self.lambda_ext = 0.985, 0.92
            self.gamma_int, self.lambda_int = 0.97, 0.90
        elif profile == "expert":
            self.gamma_ext, self.lambda_ext = 0.995, 0.97
            self.gamma_int, self.lambda_int = 0.99, 0.95
        else:  # base
            self.gamma_ext, self.lambda_ext = 0.99, 0.95
            self.gamma_int, self.lambda_int = 0.985, 0.93

        # AMP convenience flag
        self.cuda_autocast_enabled = self.use_amp and self.device and str(self.device).startswith("cuda")
        
        # Store latest action probabilities for visualization
        self.latest_action_probs = [0.25, 0.25, 0.25, 0.25]  # Initialize with uniform
        
        # Metrics tracking (store as (episode, value) tuples)
        self.loss_history = deque(maxlen=1000)
        self.score_history = deque(maxlen=1000)
        self.policy_loss_history = deque(maxlen=1000)
        self.value_loss_history = deque(maxlen=1000)
        self.entropy_history = deque(maxlen=1000)
        
        # Experience buffer (separate extrinsic/intrinsic rewards and values) with profile-aware size
        if profile == "lightning":
            self.buffer_size = 1536
        elif profile == "expert":
            self.buffer_size = 4096
        else:
            self.buffer_size = 2048
        # Preallocated ring buffer on device to minimize H2D transfers
        self._buf_capacity = self.buffer_size
        device_for_buf = self.device
        self._buf_states_t = torch.zeros((self._buf_capacity, 4, 4), dtype=torch.float32, device=device_for_buf)
        self._buf_actions_t = torch.zeros((self._buf_capacity,), dtype=torch.long, device=device_for_buf)
        self._buf_rewards_ext_t = torch.zeros((self._buf_capacity,), dtype=torch.float32, device=device_for_buf)
        self._buf_rewards_int_t = torch.zeros((self._buf_capacity,), dtype=torch.float32, device=device_for_buf)
        self._buf_values_ext_t = torch.zeros((self._buf_capacity,), dtype=torch.float32, device=device_for_buf)
        self._buf_values_int_t = torch.zeros((self._buf_capacity,), dtype=torch.float32, device=device_for_buf)
        self._buf_log_probs_t = torch.zeros((self._buf_capacity,), dtype=torch.float32, device=device_for_buf)
        self._buf_dones_t = torch.zeros((self._buf_capacity,), dtype=torch.bool, device=device_for_buf)
        self._buf_count = 0  # number of valid entries (<= capacity)
        self._buf_head = 0   # next write index (ring buffer)
        # Legacy field retained for compatibility with cleanup routines
        self.buffer = {'states': []}
        
        # Optimization: Track if policy update is needed
        self._policy_update_counter = 0
        # With synchronous updates from manager, we can keep a fallback frequency
        self._policy_update_frequency = 999999  # Manager drives updates
        
        print(f"PPO Trainer initialized:")
        print(f"  Model parameters: {self.model.count_parameters():,}")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Buffer size: {self.buffer_size}")
        print(f"  Policy update frequency: every {self._policy_update_frequency} episodes")
        
        self.timing_logger.end_operation("trainer_init", "setup", 
                                       f"model_params={self.model.count_parameters():,}, device={self.device}, batch_size={self.batch_size}")
        # Ensure any external router bias is cleared; rely on on-GPU adaptive mitigation
        try:
            self.model.set_router_bias(None)
        except Exception:
            pass
    
    def calculate_load_balancing_reward(self) -> float:
        """Calculate intrinsic reward focused on anti-starvation and anti-dominance.

        Components:
        - Expert utilization balance (low variance, avoid starvation)
        - Router entropy/diversity (no sparsity pressure)
        - Dominance penalties for over-concentration
        - Recovery incentives to prevent collapse
        - Progressive scaling and early-training boosts
        """
        self.timing_logger.start_operation("calculate_lb_reward", "reward")
        
        # Get current expert usage from the model
        expert_usage = self.model.get_expert_usage()
        if expert_usage is None:
            self.timing_logger.end_operation("calculate_lb_reward", "reward", "no_expert_usage")
            return 0.0
        
        # Convert to numpy for calculations
        if hasattr(expert_usage, 'cpu'):
            usage_np = expert_usage.cpu().numpy()
        elif hasattr(expert_usage, 'numpy'):
            usage_np = expert_usage.numpy()
        else:
            usage_np = np.array(expert_usage)
        n_experts = len(usage_np)
        
        # Store usage in history for diversity tracking (thread-safe)
        with self._buffer_lock:  # Reuse existing lock for thread safety
            self.expert_usage_history.append(np.array(usage_np))
        
        # Calculate ideal uniform distribution
        ideal_usage = 1.0 / n_experts
        
        # ENHANCED: Multi-component load balancing reward
        
        # 1. Variance penalty (lower variance = better balance)
        variance = np.var(usage_np)
        max_variance = (1.0 - ideal_usage) ** 2
        normalized_variance = variance / max_variance if max_variance > 0 else 0.0
        variance_penalty = normalized_variance * 2.0  # Keep current penalty
        
        # 2. ENHANCED: Expert starvation penalty with progressive scaling
        starvation_penalty = 0.0
        starved_experts = 0
        starved_expert_indices = []
        
        for i, usage in enumerate(usage_np):
            if float(usage) < self.lb_critical_threshold:
                # ENHANCED: Progressive penalty based on how far below threshold
                penalty_factor = (self.lb_critical_threshold - float(usage)) / self.lb_critical_threshold
                # Additional penalty for experts that are severely starved
                if float(usage) < self.lb_critical_threshold * 0.5:
                    penalty_factor *= 2.0  # Double penalty for severely starved experts
                
                starvation_penalty += penalty_factor
                starved_experts += 1
                starved_expert_indices.append(i)
        
        # ENHANCED: Track starved experts for recovery analysis
        starved_expert_indices_list = list(starved_expert_indices)
        for idx in starved_expert_indices_list:
            if idx not in self.starved_experts_tracker:
                self.starved_experts_tracker[idx] = 0
            self.starved_experts_tracker[idx] += 1
        
        # Clean up tracker (remove experts that haven't been starved recently)
        for idx in list(self.starved_experts_tracker.keys()):
            if idx not in starved_expert_indices_list:
                current_count = int(self.starved_experts_tracker[idx])
                new_count = max(0, current_count - 1)
                self.starved_experts_tracker[idx] = new_count
                if new_count <= 0:
                    del self.starved_experts_tracker[idx]
        
        # Additional penalty for multiple starved experts
        if starved_experts > 1:
            starvation_penalty *= (1.0 + starved_experts * 0.5)
        
        # 3. Entropy bonus (higher entropy = better diversity) - no sparsity enforcement
        epsilon = 1e-8
        log_usage = np.log(usage_np + epsilon)
        entropy = float(-np.sum(usage_np.astype(np.float64) * log_usage))
        max_entropy = float(np.log(n_experts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        entropy_bonus = normalized_entropy * 1.5  # Keep current bonus
        
        # 4. ENHANCED: Diversity tracking penalty with recovery incentives
        diversity_penalty = 0.0
        if len(self.expert_usage_history) >= 10:
            # Calculate diversity over recent history
            recent_usage = np.array(list(self.expert_usage_history)[-10:])
            avg_usage = np.mean(recent_usage, axis=0)
            
            # ENHANCED: Penalty for experts that are consistently underused
            for i, avg_usage_val in enumerate(avg_usage):
                if avg_usage_val < ideal_usage * 0.5:  # Consistently underused
                    diversity_penalty += (ideal_usage * 0.5 - avg_usage_val) / ideal_usage
                    
                    # NEW: Recovery bonus for experts that were starved but are improving
                    if i in self.starved_experts_tracker and self.starved_experts_tracker[i] > 0:
                        # If this expert was starved but current usage is improving, reduce penalty
                        current_usage = usage_np[i]
                        if current_usage > avg_usage_val * 1.2:  # 20% improvement
                            diversity_penalty *= 0.8  # Reduce penalty for recovering experts
        
        # 5. Dominance penalties (anti-monopoly): penalize over-concentration
        hhi = float(np.sum(np.square(usage_np)))  # Herfindahl index
        min_hhi = 1.0 / n_experts
        norm_hhi = (hhi - min_hhi) / (1.0 - min_hhi) if n_experts > 1 else 0.0
        hhi_penalty = norm_hhi * 0.8

        top1 = float(np.max(usage_np))
        dom_threshold = max(0.5, 1.5 * ideal_usage)
        top1_overflow = max(0.0, (top1 - dom_threshold) / (1.0 - dom_threshold))
        dominance_penalty = top1_overflow ** 2 * 1.2
        
        # 6. Balance quality metric (light touch)
        balance_quality = 1.0 - np.mean(np.abs(usage_np - ideal_usage)) / ideal_usage
        balance_bonus = balance_quality * 0.1
        
        # ENHANCED: Progressive load balancing scaling
        if self.episode_count > self.lb_progressive_episodes:
            # Gradually increase load balancing strength for larger models
            progress_factor = min(2.0, 1.0 + (self.episode_count - self.lb_progressive_episodes) / 5000)
            self.lb_progressive_factor = progress_factor
        
        # Combine all components (no sparsity bonus)
        lb_reward = (entropy_bonus +
                     balance_bonus -
                     variance_penalty -
                     starvation_penalty -
                     diversity_penalty -
                     hhi_penalty -
                     dominance_penalty)
        
        # Apply adaptive factor based on model size
        lb_reward *= self.lb_adaptive_factor
        
        # Apply progressive scaling for larger models
        lb_reward *= self.lb_progressive_factor
        
        # Apply early training boost
        if self.episode_count < self.lb_episode_threshold:
            lb_reward *= (1.0 + self.lb_early_training_boost)
        
        # Scale by coefficient
        final_reward = float(lb_reward * self.lb_reward_coef)
        
        # Log detailed metrics for debugging
        self.timing_logger.end_operation("calculate_lb_reward", "reward", 
                                       f"reward={final_reward:.4f}, variance={normalized_variance:.4f}, "
                                       f"entropy={normalized_entropy:.4f}, starvation={starvation_penalty:.4f}, "
                                       f"diversity={diversity_penalty:.4f}, "
                                       f"balance={balance_bonus:.4f}, hhi_penalty={hhi_penalty:.4f}, "
                                       f"dominance={dominance_penalty:.4f}, progressive_factor={self.lb_progressive_factor:.2f}")
        
        return final_reward
    
    # Router biasing removed – rely on on-GPU adaptive mitigation in MoE

    def select_action(self, state: np.ndarray, legal_actions: List[int], env_game) -> Tuple[int, float, float, float]:
        """Select action using current policy with fallback for invalid moves"""
        
        self.timing_logger.start_operation("action_selection", "inference")
        
        # Single forward pass for logits and values (AMP-enabled)
        self.timing_logger.start_operation("model_forward", "inference")
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            # Enable AMP for inference on CUDA to accelerate forward pass
            if self.cuda_autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    policy_logits, value_ext, value_int = self.model(state_tensor)
            else:
                policy_logits, value_ext, value_int = self.model(state_tensor)

            # Mask illegal actions for visualization
            action_mask = torch.full((4,), -float('inf'), device=self.device)
            action_mask[legal_actions] = 0.0
            masked_logits = policy_logits[0] + action_mask
            if torch.isnan(masked_logits).any() or torch.isinf(masked_logits).any():
                # Fallback to uniform over legal actions if logits are invalid
                masked_logits = torch.full((4,), -float('inf'), device=self.device)
                masked_logits[legal_actions] = 0.0
            action_probs = F.softmax(masked_logits, dim=-1)

            # Store latest action probabilities for visualization
            self.latest_action_probs = action_probs.detach().cpu().numpy().tolist()
        self.timing_logger.end_operation("model_forward", "inference", f"legal_actions={len(legal_actions)}")

        # Use fallback mechanism to select action with adaptive exploration (no extra forward)
        self.timing_logger.start_operation("action_fallback", "inference")
        # Novelty- and health-aware exploration: scale with novelty, stagnation, and dominance
        profile = getattr(self.config, "model_size", None) or "base"
        state_novelty = self._get_state_novelty(state)
        dominance_factor = self._get_router_dominance_factor()
        stagnation_factor = min(1.0, self._stagnation_steps / max(1, self._stagnation_threshold_steps))

        # Profile-aware exploration schedules
        if profile == "lightning":
            base_temp = 1.0 - 0.00004 * self.total_steps
            base_eps = 0.08 - 0.00002 * self.total_steps
            dir_alpha_base = 0.02
            dir_weight_base = 0.04
        elif profile == "expert":
            base_temp = 1.2 - 0.00003 * self.total_steps
            base_eps = 0.10 - 0.00002 * self.total_steps
            dir_alpha_base = 0.05
            dir_weight_base = 0.08
        else:  # base
            base_temp = 1.1 - 0.00004 * self.total_steps
            base_eps = 0.09 - 0.00002 * self.total_steps
            dir_alpha_base = 0.03
            dir_weight_base = 0.06

        base_temp = float(np.clip(base_temp, 0.7, 1.6))
        temp = base_temp + 0.5 * (1.0 - state_novelty) + 0.3 * stagnation_factor + 0.2 * dominance_factor
        temp = float(np.clip(temp, 0.7, 2.2))

        base_eps = float(np.clip(base_eps, 0.02, 0.15))
        eps = base_eps + 0.2 * (1.0 - state_novelty) + 0.15 * stagnation_factor + 0.1 * dominance_factor
        eps = float(np.clip(eps, 0.03, 0.6))

        dir_alpha = dir_alpha_base + 0.35 * (1.0 - state_novelty) + 0.3 * dominance_factor
        dir_alpha = float(np.clip(dir_alpha, 0.02, 1.0))
        dir_weight = dir_weight_base + 0.25 * (1.0 - state_novelty) + 0.2 * stagnation_factor + 0.15 * dominance_factor
        dir_weight = float(np.clip(dir_weight, 0.02, 0.6))
        action, log_prob = select_action_from_logits_with_validation(
            policy_logits=policy_logits[0],
            legal_actions=legal_actions,
            env_game=env_game,
            device=self.device,
            sample_action=True,
            max_attempts=4,
            temperature=temp,
            epsilon=eps,
            min_explore_prob=0.01,
            dirichlet_alpha=dir_alpha,
            dirichlet_weight=dir_weight,
            validate_moves=self.validate_moves,
        )
        self.timing_logger.end_operation("action_fallback", "inference", f"selected_action={action}")

        self.timing_logger.end_operation("action_selection", "inference", f"action={action}")
        
        return action, log_prob, float(value_ext.item()), float(value_int.item())

    # ------------------------------ Evaluation ------------------------------
    def _select_action_greedy(self, state: np.ndarray, legal_actions: List[int]) -> int:
        """Deterministic greedy action selection (no exploration, no validation).
        Masks illegal actions and picks argmax.
        """
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            if self.use_amp and self.device and str(self.device).startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    policy_logits, _, _ = self.model(state_tensor)
            else:
                policy_logits, _, _ = self.model(state_tensor)
            action_mask = torch.full((4,), -float('inf'), device=self.device)
            action_mask[legal_actions] = 0.0
            masked_logits = policy_logits[0] + action_mask
            action = int(torch.argmax(masked_logits).item())
            if action not in legal_actions and len(legal_actions) > 0:
                action = legal_actions[0]
            return action

    def evaluate_policy(self, *, num_episodes: int = 100, seeds: Optional[List[int]] = None) -> Dict[str, Any]:
        """Run deterministic evaluation episodes and compute metrics.

        - No exploration, novelty, or intrinsic signals are used.
        - Uses greedy masked argmax action selection.
        - Returns aggregate metrics for checkpointing/monitoring.
        """
        scores: List[int] = []
        lengths: List[int] = []
        max_tiles: List[int] = []

        seeds = seeds or list(range(num_episodes))
        if len(seeds) < num_episodes:
            # Repeat seeds deterministically if fewer provided
            seeds = (seeds * ((num_episodes + len(seeds) - 1) // len(seeds)))[:num_episodes]

        for i in range(num_episodes):
            env = Gym2048Env(seed=seeds[i])
            state, _ = env.reset()
            done = False
            steps = 0
            while not done:
                legal_actions = env.game.legal_moves()
                if not legal_actions:
                    break
                action = self._select_action_greedy(state, legal_actions)
                state, reward, done, _, _ = env.step(action)
                steps += 1

            lengths.append(steps)
            scores.append(int(env.get_score()))
            max_tiles.append(int(np.max(state)))

        # Aggregate metrics
        scores_sorted = sorted(scores)
        p50 = statistics.median(scores_sorted) if scores_sorted else 0.0
        p75 = scores_sorted[int(0.75 * (len(scores_sorted) - 1))] if scores_sorted else 0.0
        p90 = scores_sorted[int(0.90 * (len(scores_sorted) - 1))] if scores_sorted else 0.0
        mean_score = float(sum(scores) / len(scores)) if scores else 0.0
        mean_len = float(sum(lengths) / len(lengths)) if lengths else 0.0
        # Solve rates by max tile achieved
        solve_1024 = float(sum(1 for t in max_tiles if t >= 1024)) / float(len(max_tiles)) if max_tiles else 0.0
        solve_2048 = float(sum(1 for t in max_tiles if t >= 2048)) / float(len(max_tiles)) if max_tiles else 0.0
        # Max tile frequency histogram
        tile_freq: Dict[int, int] = {}
        for t in max_tiles:
            tile_freq[t] = tile_freq.get(t, 0) + 1

        return {
            'episodes': num_episodes,
            'mean_score': mean_score,
            'median_score': float(p50),
            'p75_score': float(p75),
            'p90_score': float(p90),
            'mean_length': mean_len,
            'solve_rate_1024': solve_1024,
            'solve_rate_2048': solve_2048,
            'max_tile_frequency': tile_freq,
            'scores': scores,
            'lengths': lengths,
        }

    # ---- Novelty utilities ----
    def _encode_state(self, state: np.ndarray) -> tuple:
        # Vectorized lookup: map tile values to exponents
        arr = np.asarray(state, dtype=np.int64).ravel()
        if self._novelty_lookup.shape[0] > 1025:
            exps = self._novelty_lookup[arr]
        else:
            # Fallback for larger tiles not in small table
            exps = np.empty_like(arr, dtype=np.uint8)
            for i, v in enumerate(arr):
                if 0 <= v < self._novelty_lookup.shape[0]:
                    exps[i] = self._novelty_lookup[v]
                elif v > 0:
                    exps[i] = (int(v).bit_length() - 1)
                else:
                    exps[i] = 0
        return tuple(int(x) for x in exps)

    def _hash_state(self, state: np.ndarray) -> tuple:
        # Use encoded tuple directly as dictionary key (faster than cryptographic hash)
        return self._encode_state(state)

    def _get_state_novelty(self, state: np.ndarray) -> float:
        h = self._hash_state(state)
        c = self._novelty_counts.get(h, 0)
        novelty = 1.0 / float(np.sqrt(1.0 + c))
        return float(np.clip(novelty, self.novelty_min, self.novelty_max))

    def _update_state_novelty(self, state: np.ndarray) -> float:
        h = self._hash_state(state)
        novelty = self._get_state_novelty(state)
        if h not in self._novelty_counts:
            if len(self._novelty_counts) >= self._novelty_order.maxlen:
                try:
                    old_key = self._novelty_order.popleft()
                    if old_key in self._novelty_counts:
                        del self._novelty_counts[old_key]
                except Exception:
                    pass
            self._novelty_order.append(h)
            self._novelty_counts[h] = 1
        else:
            self._novelty_counts[h] += 1
        return novelty

    def _get_router_dominance_factor(self) -> float:
        expert_usage = self.model.get_expert_usage()
        if expert_usage is None:
            return 0.0
        usage_np = expert_usage.cpu().numpy() if hasattr(expert_usage, 'cpu') else np.array(expert_usage)
        n = len(usage_np) if len(usage_np) > 0 else 1
        hhi = float(np.sum(np.square(usage_np)))
        min_hhi = 1.0 / n
        norm_hhi = (hhi - min_hhi) / (1.0 - min_hhi) if n > 1 else 0.0
        return float(np.clip(norm_hhi, 0.0, 1.0))
    
    def get_latest_action_probs(self) -> List[float]:
        """Get the latest action probabilities for visualization"""
        return self.latest_action_probs
    
    def _current_clip(self) -> float:
        # Linear schedule from clip_epsilon to clip_epsilon_final over intrinsic_weight_decay_updates
        steps = max(1, self.intrinsic_weight_decay_updates)
        progress = min(1.0, self.update_steps / steps)
        base = float(self.clip_epsilon + (self.clip_epsilon_final - self.clip_epsilon) * progress)
        scaled = base * self._clip_scale
        return float(np.clip(scaled, 0.05, 0.3))

    def _current_entropy_coef(self) -> float:
        steps = max(1, self.intrinsic_weight_decay_updates)
        progress = min(1.0, self.update_steps / steps)
        base = float(self.entropy_coef + (self.entropy_coef_final - self.entropy_coef) * progress)
        scaled = base * self._entropy_scale
        return float(np.clip(scaled, 0.0005, 0.05))
    
    def store_transition(self, state: np.ndarray, action: int, reward_ext: float, reward_int: float,
                        value_ext: float, value_int: float, log_prob: float, done: bool):
        """Store transition in experience buffer (thread-safe)"""
        
        self.timing_logger.start_operation("store_transition", "buffer")
        lock_start = time.perf_counter()
        with self._buffer_lock:
            idx = self._buf_head
            self._buf_head = (self._buf_head + 1) % self._buf_capacity
            if self._buf_count < self._buf_capacity:
                self._buf_count += 1
            # Write directly to device tensors
            # Convert inputs minimally; expect state as numpy float32 already
            if not isinstance(state, np.ndarray):
                state = np.asarray(state, dtype=np.float32)
            self._buf_states_t[idx].copy_(torch.from_numpy(state).to(self.device))
            self._buf_actions_t[idx] = int(action)
            self._buf_rewards_ext_t[idx] = float(reward_ext)
            self._buf_rewards_int_t[idx] = float(reward_int)
            self._buf_values_ext_t[idx] = float(value_ext)
            self._buf_values_int_t[idx] = float(value_int)
            self._buf_log_probs_t[idx] = float(log_prob)
            self._buf_dones_t[idx] = bool(done)
        lock_duration = time.perf_counter() - lock_start
        self.timing_logger.end_operation("store_transition", "buffer", f"buffer_size={self._buf_count}, lock_duration={lock_duration:.4f}s")
        
        if lock_duration > 0.01:
            print(f"[yellow]store_transition lock duration {lock_duration:.4f}s")
    
    def compute_advantages(self, rewards: List[float], values: List[float], 
                          dones: List[bool], gamma: float = 0.99, 
                          lambda_: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advantages using GAE (Generalized Advantage Estimation) for a single reward/value stream."""
        
        self.timing_logger.start_operation("compute_advantages", "training")
        
        advantages = []
        returns = []
        
        # Add bootstrap value for last state (0 if terminal)
        next_value = 0.0
        
        # Compute advantages backwards
        advantage = 0.0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0.0
                advantage = 0.0
            
            delta = rewards[i] + gamma * next_value - values[i]
            advantage = delta + gamma * lambda_ * advantage
            
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[i])
            
            next_value = values[i]
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.timing_logger.end_operation("compute_advantages", "training", f"sequence_length={len(rewards)}")
        
        return advantages, returns

    def _intrinsic_weight(self) -> float:
        # Linearly anneal from initial to 0 over intrinsic_weight_decay_updates
        w0 = self.intrinsic_weight_initial
        steps = max(1, self.intrinsic_weight_decay_updates)
        w = max(0.0, w0 * (1.0 - self.update_steps / steps))
        return float(w)
    
    def update_policy(self) -> Dict[str, Optional[float]]:
        """Update policy using PPO (thread-safe)"""

        self.timing_logger.start_operation("update_policy", "training")
        
        with self._update_lock:
            lock_start = time.perf_counter()
            with self._buffer_lock:
                if self._buf_count < self.batch_size:
                    self.timing_logger.end_operation("update_policy", "training", "insufficient_buffer")
                    return {'policy_loss': None, 'value_loss': None, 'entropy': None}

                # Snapshot indices for contiguous view
                count = self._buf_count
                if count < self._buf_capacity:
                    idxs = torch.arange(0, count, device=self.device)
                else:
                    # Wrapped: roll index so that oldest is first
                    start = self._buf_head
                    idxs = torch.arange(0, self._buf_capacity, device=self.device)
                    idxs = (idxs + start) % self._buf_capacity
                # Reset pointers
                self._buf_count = 0
                self._buf_head = 0

            lock_duration = time.perf_counter() - lock_start
            self.timing_logger.log_event("buffer_snapshot", "training", lock_duration * 1000, f"states={int(count)}")
            
            if lock_duration > 0.2:
                print(f"[yellow]update_policy held buffer lock for {lock_duration:.3f}s with {int(count)} states")

            # Slice device tensors directly using idxs (no H2D transfers)
            self.timing_logger.start_operation("tensor_gather", "training")
            states = self._buf_states_t[idxs]
            actions = self._buf_actions_t[idxs]
            old_log_probs = self._buf_log_probs_t[idxs]
            rewards_ext_t = self._buf_rewards_ext_t[idxs]
            rewards_int_t = self._buf_rewards_int_t[idxs]
            values_ext_t = self._buf_values_ext_t[idxs]
            values_int_t = self._buf_values_int_t[idxs]
            dones_list = self._buf_dones_t[idxs].tolist()  # small boolean list used in GAE loops
            self.timing_logger.end_operation("tensor_gather", "training", f"tensors={len(states)}")

            # Compute separate advantages/returns for extrinsic and intrinsic with profile-aware gamma/lambda
            profile = getattr(self.config, "model_size", None) or "base"
            # Compute advantages on device (vectorized) for both streams
            def gae_compute(rewards_t: torch.Tensor, values_t: torch.Tensor, dones_py: list[bool], gamma: float, lam: float):
                T = rewards_t.shape[0]
                adv = torch.zeros(T, dtype=torch.float32, device=self.device)
                ret = torch.zeros(T, dtype=torch.float32, device=self.device)
                next_value = torch.tensor(0.0, device=self.device)
                running_adv = torch.tensor(0.0, device=self.device)
                for i in range(T - 1, -1, -1):
                    if dones_py[i]:
                        next_value = torch.tensor(0.0, device=self.device)
                        running_adv = torch.tensor(0.0, device=self.device)
                    delta = rewards_t[i] + gamma * next_value - values_t[i]
                    running_adv = delta + gamma * lam * running_adv
                    adv[i] = running_adv
                    ret[i] = running_adv + values_t[i]
                    next_value = values_t[i]
                # Normalize advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                return adv, ret

            advantages_ext, returns_ext = gae_compute(rewards_ext_t, values_ext_t, dones_list, self.gamma_ext, self.lambda_ext)
            advantages_int, returns_int = gae_compute(rewards_int_t, values_int_t, dones_list, self.gamma_int, self.lambda_int)

            # Blend advantages for policy with annealed intrinsic weight
            w_int = self._intrinsic_weight()
            advantages = advantages_ext + w_int * advantages_int
            # Keep value losses separate (extrinsic/intinsic) to avoid interference
            # returns_ext/returns_int are already tensors

            # PPO update
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            total_lb_loss = 0.0
            
            self.timing_logger.start_operation("ppo_epochs", "training", f"epochs={self.ppo_epochs}")
            for epoch in range(self.ppo_epochs):
                self.timing_logger.start_operation("ppo_epoch", "training", f"epoch={epoch}")
                
                # Shuffle data
                indices = torch.randperm(len(states), device=self.device)
                
                batch_count = 0
                early_stop = False
                for start in range(0, len(states), self.batch_size):
                    batch_count += 1
                    self.timing_logger.start_operation("batch_update", "training", f"epoch={epoch}, batch={batch_count}")
                    
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    
                    # Get batch
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    # For dual critics, we keep separate returns and use clipped losses; batch_returns not used
                    
                    # Forward pass (AMP-enabled)
                    self.timing_logger.start_operation("batch_forward", "training")
                    # Biasing handled internally by MoE layer on-GPU
                    if self.cuda_autocast_enabled:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            policy_logits, values_ext, values_int = self.model(batch_states)
                            action_dist = torch.distributions.Categorical(logits=policy_logits)
                            new_log_probs = action_dist.log_prob(batch_actions)
                            ratio = torch.exp(new_log_probs - batch_old_log_probs)
                            surr1 = ratio * batch_advantages
                            surr2 = torch.clamp(ratio, 1 - self._current_clip(), 1 + self._current_clip()) * batch_advantages
                            policy_loss = -torch.min(surr1, surr2).mean()
                            # Value losses with clipping
                            values_ext = values_ext.view(-1)
                            values_int = values_int.view(-1)
                            v_ext_pred_clipped = values_ext + (returns_ext[batch_indices] - values_ext).clamp(-self._current_clip(), self._current_clip())
                            v_int_pred_clipped = values_int + (returns_int[batch_indices] - values_int).clamp(-self._current_clip(), self._current_clip())
                            v_ext_loss_unclipped = F.huber_loss(values_ext, returns_ext[batch_indices], delta=1.0)
                            v_ext_loss_clipped = F.huber_loss(v_ext_pred_clipped, returns_ext[batch_indices], delta=1.0)
                            v_int_loss_unclipped = F.huber_loss(values_int, returns_int[batch_indices], delta=1.0)
                            v_int_loss_clipped = F.huber_loss(v_int_pred_clipped, returns_int[batch_indices], delta=1.0)
                            value_loss = 0.5 * (torch.max(v_ext_loss_unclipped, v_ext_loss_clipped) + torch.max(v_int_loss_unclipped, v_int_loss_clipped))
                            entropy = action_dist.entropy().mean()
                    else:
                        policy_logits, values_ext, values_int = self.model(batch_states)
                        action_dist = torch.distributions.Categorical(logits=policy_logits)
                        new_log_probs = action_dist.log_prob(batch_actions)
                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - self._current_clip(), 1 + self._current_clip()) * batch_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        values_ext = values_ext.view(-1)
                        values_int = values_int.view(-1)
                        v_ext_pred_clipped = values_ext + (returns_ext[batch_indices] - values_ext).clamp(-self._current_clip(), self._current_clip())
                        v_int_pred_clipped = values_int + (returns_int[batch_indices] - values_int).clamp(-self._current_clip(), self._current_clip())
                        v_ext_loss_unclipped = F.huber_loss(values_ext, returns_ext[batch_indices], delta=1.0)
                        v_ext_loss_clipped = F.huber_loss(v_ext_pred_clipped, returns_ext[batch_indices], delta=1.0)
                        v_int_loss_unclipped = F.huber_loss(values_int, returns_int[batch_indices], delta=1.0)
                        v_int_loss_clipped = F.huber_loss(v_int_pred_clipped, returns_int[batch_indices], delta=1.0)
                        value_loss = 0.5 * (torch.max(v_ext_loss_unclipped, v_ext_loss_clipped) + torch.max(v_int_loss_unclipped, v_int_loss_clipped))
                        entropy = action_dist.entropy().mean()

                    # Auxiliary loss and final loss (common)
                    lb_loss = self.model.latest_lb_loss if getattr(self.model, 'latest_lb_loss', None) is not None else 0.0
                    if isinstance(lb_loss, (int, float)):
                        lb_loss_t = torch.tensor(float(lb_loss), device=self.device)
                    else:
                        lb_loss_t = lb_loss.to(self.device)
                    loss = (policy_loss + self.value_loss_coef * value_loss - self._current_entropy_coef() * entropy + self.lb_loss_coef * lb_loss_t)
                    self.timing_logger.end_operation("batch_forward", "training", f"batch_size={len(batch_states)}")

                    # Backward pass (AMP-aware)
                    self.timing_logger.start_operation("backward_pass", "training")
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.use_amp and self.device and str(self.device).startswith("cuda"):
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                    # Step the LR scheduler once per optimisation step
                    self.scheduler.step()
                    self.timing_logger.end_operation("backward_pass", "training")
                    
                    total_policy_loss += float(policy_loss.item())
                    total_value_loss += float(value_loss.item())
                    total_entropy += float(entropy.item())
                    if isinstance(lb_loss, torch.Tensor):
                        total_lb_loss += float(lb_loss.item())
                    elif isinstance(lb_loss, (int, float)):
                        total_lb_loss += float(lb_loss)
                    
                    self.timing_logger.end_operation("batch_update", "training", f"batch_size={len(batch_states)}")

                    # Approximate KL on this mini-batch and early stop if too large
                    with torch.no_grad():
                        approx_kl_mb = torch.mean((batch_old_log_probs - new_log_probs)).abs().item()
                    if approx_kl_mb > 1.5 * self.target_kl:
                        early_stop = True
                        # Lightly cool exploration if overshooting
                        self._entropy_scale *= 0.98
                        continue
                
                self.timing_logger.end_operation("ppo_epoch", "training", f"batches={batch_count}, early_stop={early_stop}")
                if early_stop:
                    break
            
            self.timing_logger.end_operation("ppo_epochs", "training")
            
            # Early stop by target KL (approximate KL)
            with torch.no_grad():
                # Approximate KL over the last minibatch processed (cheap proxy)
                approx_kl = torch.mean((batch_old_log_probs - new_log_probs)).abs().item()

            # Average losses
            num_updates = self.ppo_epochs * max(1, len(states) // self.batch_size)
            avg_policy_loss = total_policy_loss / num_updates
            avg_value_loss = total_value_loss / num_updates
            avg_entropy = total_entropy / num_updates
            avg_lb_loss = total_lb_loss / num_updates
            
            self.timing_logger.end_operation("update_policy", "training", 
                                           f"updates={num_updates}, policy_loss={avg_policy_loss:.4f}")
            
            self.update_steps += 1

            # KL-adaptive schedules: keep updates stable without extra passes
            try:
                if approx_kl > 1.2 * self.target_kl:
                    self._clip_scale *= 0.95
                    self._entropy_scale *= 0.95
                elif approx_kl < 0.5 * self.target_kl:
                    self._clip_scale *= 1.03
                    self._entropy_scale *= 1.05
                # Clamp scales
                self._clip_scale = float(np.clip(self._clip_scale, 0.75, 1.25))
                self._entropy_scale = float(np.clip(self._entropy_scale, 0.5, 1.5))
            except Exception:
                pass

            # Periodic GPU cache trim to prevent slow memory creep on some drivers
            try:
                if torch.cuda.is_available() and (self.update_steps % 200 == 0):
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return {
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'entropy': avg_entropy,
                'lb_loss': avg_lb_loss,
                'kl': approx_kl
            }
    
    async def train_episode(self, env: Gym2048Env) -> Dict[str, Any]:
        """Train for one episode"""
        
        self.timing_logger.start_operation("train_episode", "episode", f"episode={self.episode_count + 1}")
        
        try:
            self.timing_logger.start_operation("env_reset", "episode")
            obs, _ = env.reset()
            self.timing_logger.end_operation("env_reset", "episode")
            
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            self.timing_logger.start_operation("episode_loop", "episode")
            # Hard safety cap to prevent pathological episodes from running forever
            max_episode_steps = 5000
            while not done and episode_length < max_episode_steps:
                try:
                    # Select action
                    legal_actions = env.game.legal_moves()
                    if not legal_actions:
                        break

                    try:
                        action, log_prob, value_ext, value_int = self.select_action(obs, legal_actions, env.game)
                    except Exception as sel_err:
                        print(f"Warning: select_action failed at step {episode_length}: {sel_err}")
                        action = legal_actions[0]
                        log_prob = 0.0
                        value_ext = 0.0
                        value_int = 0.0

                    # Take step
                    self.timing_logger.start_operation("env_step", "episode")
                    next_obs, reward, done, _, _ = env.step(action)
                    self.timing_logger.end_operation("env_step", "episode", f"action={action}, reward={reward}")
                    
                    # Calculate intrinsic reward only (do not mix into extrinsic returns)
                    lb_reward = self.calculate_load_balancing_reward()
                    novelty_next = self._get_state_novelty(next_obs)
                    intrinsic_reward = self.novelty_bonus_coef * novelty_next  # consider adding RND later
                    # Keep extrinsic reward as environment reward

                    # Update novelty table and stagnation tracker
                    self._update_state_novelty(next_obs)
                    if (abs(reward) < 1e-6) and (novelty_next < 0.2):
                        self._stagnation_steps += 1
                    else:
                        self._stagnation_steps = 0

                    # Store transition with separated extrinsic/intrinsic rewards and values
                    self.store_transition(obs, action, reward, intrinsic_reward + lb_reward, value_ext, value_int, log_prob, done)
                    
                    obs = next_obs
                    episode_reward += reward
                    episode_length += 1
                    self.total_steps += 1
                    
                    # CRITICAL FIX: Yield control to event loop every 10 steps to prevent blocking
                    if episode_length % 10 == 0:
                        await asyncio.sleep(0)  # Yield to event loop without delay
                except Exception as step_error:
                    print(f"Error in episode step {episode_length}: {step_error}")
                    import traceback
                    traceback.print_exc()
                    break
            # Note if we hit the step cap
            cap_note = " (capped)" if episode_length >= max_episode_steps else ""
            self.timing_logger.end_operation("episode_loop", "episode", f"length={episode_length}{cap_note}")
            
        except Exception as e:
            print(f"Error in train_episode: {e}")
            import traceback
            traceback.print_exc()
            self.timing_logger.end_operation("train_episode", "episode", f"error={str(e)}")
            # Return default values
            return {
                'episode': self.episode_count + 1,
                'score': 0,
                'reward': 0.0,
                'length': 0,
                'losses': {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0},
                'best_score': self.best_score
            }
        
        # Defer updates to manager's synchronous phase; keep placeholder losses
        losses = {'policy_loss': None, 'value_loss': None, 'entropy': None}
        
        # Update metrics
        self.episode_count += 1
        current_score = int(env.game.score)
        
        if current_score > self.best_score:
            self.best_score = current_score
        
        # Store metrics with episode numbers
        if losses['policy_loss'] is not None and losses['value_loss'] is not None:
            total_loss = losses['policy_loss'] + losses['value_loss']
            self.loss_history.append((self.episode_count, total_loss))
        self.score_history.append((self.episode_count, current_score))
        self.policy_loss_history.append((self.episode_count, losses['policy_loss']))
        self.value_loss_history.append((self.episode_count, losses['value_loss']))
        self.entropy_history.append((self.episode_count, losses['entropy']))
        
        self.timing_logger.end_operation("train_episode", "episode", 
                                       f"score={current_score}, reward={episode_reward:.2f}, length={episode_length}")
        
        return {
            'episode': self.episode_count,
            'score': current_score,
            'reward': episode_reward,
            'length': episode_length,
            'losses': losses,
            'best_score': self.best_score
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        
        self.timing_logger.start_operation("get_metrics", "monitoring")
        
        # Get model statistics
        self.timing_logger.start_operation("model_stats", "monitoring")
        attention_weights = self.model.get_attention_weights()
        expert_usage = self.model.get_expert_usage()
        self.timing_logger.end_operation("model_stats", "monitoring")
        
        # Convert attention weights to list for JSON serialization
        attention_matrix = [[0.0] * 4 for _ in range(4)]
        if attention_weights is not None:
            attention_np = attention_weights.cpu().numpy()
            for i in range(4):
                for j in range(4):
                    attention_matrix[i][j] = float(attention_np[i, j])
        
        # Convert expert usage to list
        expert_usage_list = [0.0] * self.config.n_experts
        if expert_usage is not None:
            for i, usage in enumerate(expert_usage):
                if i < len(expert_usage_list):
                    expert_usage_list[i] = float(usage)
        
        # Extract episode numbers and values from histories
        loss_episodes = [ep for ep, _ in self.loss_history][-100:]
        loss_values = [val for _, val in self.loss_history][-100:]
        
        score_episodes = [ep for ep, _ in self.score_history][-100:]
        score_values = [val for _, val in self.score_history][-100:]
        
        self.timing_logger.end_operation("get_metrics", "monitoring")
        
        return {
            'model_params': self.model.count_parameters() / 1e6,  # In millions
            'gpu_memory': self.model.get_memory_usage(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'attention_weights': attention_matrix,
            'expert_usage': expert_usage_list,
            'loss_history': {'episodes': loss_episodes, 'values': loss_values},
            'score_history': {'episodes': score_episodes, 'values': score_values},
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'best_score': self.best_score
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        self.timing_logger.start_operation("save_checkpoint", "io", f"filepath={filepath}")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'best_score': self.best_score,
            'loss_history': list(self.loss_history),
            'score_history': list(self.score_history)
        }, filepath)
        
        self.timing_logger.end_operation("save_checkpoint", "io", f"filepath={filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        self.timing_logger.start_operation("load_checkpoint", "io", f"filepath={filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint.get('episode_count', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        self.best_score = checkpoint.get('best_score', 0)
        
        # Restore history (convert old format to new tuple format if needed)
        old_loss_history = checkpoint.get('loss_history', [])
        old_score_history = checkpoint.get('score_history', [])
        
        # Check if histories are in old format (just values) or new format (tuples)
        if old_loss_history and not isinstance(old_loss_history[0], tuple):
            # Convert old format to new format, assuming sequential episodes
            self.loss_history = deque([(i+1, val) for i, val in enumerate(old_loss_history)], maxlen=1000)
        else:
            self.loss_history = deque(old_loss_history, maxlen=1000)
            
        if old_score_history and not isinstance(old_score_history[0], tuple):
            # Convert old format to new format, assuming sequential episodes
            self.score_history = deque([(i+1, val) for i, val in enumerate(old_score_history)], maxlen=1000)
        else:
            self.score_history = deque(old_score_history, maxlen=1000)
        
        self.timing_logger.end_operation("load_checkpoint", "io", f"episode_count={self.episode_count}")
    
    def write_timing_summary(self, filename: str = "logs/ppo_timing_summary.json"):
        """Write timing summary to JSON file"""
        self.timing_logger.write_summary(filename) 