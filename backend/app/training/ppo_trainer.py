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
        # TEMPORARILY DISABLED: Skip file I/O to isolate timing logger issues
        return
        
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # Cosine decay scheduler for smoother optimisation
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100_000, eta_min=1e-6
        )
        self.timing_logger.end_operation("optimizer_setup", "setup")
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        # Auxiliary loss coefficients - ENHANCED for tiny models
        self.lb_loss_coef = 0.05  # Increased from 0.01 to 0.05 for stronger load balancing
        self.max_grad_norm = 0.5
        
        # Load balancing reward parameters - ENHANCED for all model sizes
        self.lb_reward_coef = 0.3  # Keep current strength
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
        if self.config.n_experts <= 4:
            self.lb_early_training_boost = 1.0  # Current value for tiny/small
            self.lb_episode_threshold = 2000
        elif self.config.n_experts <= 6:
            self.lb_early_training_boost = 1.5  # Stronger boost for medium
            self.lb_episode_threshold = 3000
        else:
            self.lb_early_training_boost = 2.0  # Strongest boost for large
            self.lb_episode_threshold = 4000
        
        # ENHANCED: Improved adaptive factor that scales properly with model size
        # Instead of reducing strength for larger models, maintain or increase it
        if self.config.n_experts <= 4:
            self.lb_adaptive_factor = 1.5  # Strong for tiny/small
        elif self.config.n_experts <= 6:
            self.lb_adaptive_factor = 1.8  # Stronger for medium
        else:
            self.lb_adaptive_factor = 2.0  # Strongest for large
        
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
        self.ppo_epochs = 2  # Reduced from 4 to 2 for faster updates
        # Use more aggressive batch sizing when ample VRAM is available
        self.batch_size = DynamicModelConfig.get_batch_size(config)

        # ---- NEW: lock for thread-safe experience buffer operations ----
        self._buffer_lock = threading.Lock()
        # NEW: serialize policy updates to avoid concurrent optimizer steps on shared trainer
        self._update_lock = threading.Lock()

        # Mixed precision configuration
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

        # Action validation toggle (avoid expensive deep copies by default)
        self.validate_moves = validate_moves

        # Novelty-driven exploration settings
        self._novelty_counts = {}
        self._novelty_order = deque(maxlen=200_000)
        self.novelty_bonus_coef = 0.12
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
        
        # Store latest action probabilities for visualization
        self.latest_action_probs = [0.25, 0.25, 0.25, 0.25]  # Initialize with uniform
        
        # Metrics tracking (store as (episode, value) tuples)
        self.loss_history = deque(maxlen=1000)
        self.score_history = deque(maxlen=1000)
        self.policy_loss_history = deque(maxlen=1000)
        self.value_loss_history = deque(maxlen=1000)
        self.entropy_history = deque(maxlen=1000)
        
        # Experience buffer
        self.buffer_size = 512  # Reduced from 1024 to 512 for faster policy updates
        # Preallocated ring buffer for faster writes and snapshotting
        self._buf_capacity = self.buffer_size
        self._buf_states = np.zeros((self._buf_capacity, 4, 4), dtype=np.float32)
        self._buf_actions = np.zeros((self._buf_capacity,), dtype=np.int64)
        self._buf_rewards = np.zeros((self._buf_capacity,), dtype=np.float32)
        self._buf_values = np.zeros((self._buf_capacity,), dtype=np.float32)
        self._buf_log_probs = np.zeros((self._buf_capacity,), dtype=np.float32)
        self._buf_dones = np.zeros((self._buf_capacity,), dtype=np.bool_)
        self._buf_count = 0  # number of valid entries (<= capacity)
        self._buf_head = 0   # next write index (ring buffer)
        # Legacy field retained for compatibility with cleanup routines
        self.buffer = {'states': []}
        
        # Optimization: Track if policy update is needed
        self._policy_update_counter = 0
        self._policy_update_frequency = 2  # Update policy every 2 episodes instead of every buffer fill
        
        print(f"PPO Trainer initialized:")
        print(f"  Model parameters: {self.model.count_parameters():,}")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Buffer size: {self.buffer_size}")
        print(f"  Policy update frequency: every {self._policy_update_frequency} episodes")
        
        self.timing_logger.end_operation("trainer_init", "setup", 
                                       f"model_params={self.model.count_parameters():,}, device={self.device}, batch_size={self.batch_size}")
    
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
    
    def select_action(self, state: np.ndarray, legal_actions: List[int], env_game) -> Tuple[int, float, float]:
        """Select action using current policy with fallback for invalid moves"""
        
        self.timing_logger.start_operation("action_selection", "inference")
        
        # Single forward pass for logits and value (AMP-enabled)
        self.timing_logger.start_operation("model_forward", "inference")
        with torch.inference_mode():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            # Enable AMP for inference on CUDA to accelerate forward pass
            if self.use_amp and self.device and str(self.device).startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    policy_logits, value = self.model(state_tensor)
            else:
                policy_logits, value = self.model(state_tensor)

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
        state_novelty = self._get_state_novelty(state)
        dominance_factor = self._get_router_dominance_factor()
        stagnation_factor = min(1.0, self._stagnation_steps / max(1, self._stagnation_threshold_steps))

        base_temp = 1.2 - 0.00005 * self.total_steps
        base_temp = float(np.clip(base_temp, 0.7, 1.6))
        temp = base_temp + 0.6 * (1.0 - state_novelty) + 0.4 * stagnation_factor + 0.3 * dominance_factor
        temp = float(np.clip(temp, 0.7, 2.2))

        base_eps = 0.12 - 0.00003 * self.total_steps
        base_eps = float(np.clip(base_eps, 0.02, 0.15))
        eps = base_eps + 0.25 * (1.0 - state_novelty) + 0.2 * stagnation_factor + 0.15 * dominance_factor
        eps = float(np.clip(eps, 0.03, 0.6))

        dir_alpha = 0.05 + 0.45 * (1.0 - state_novelty) + 0.4 * dominance_factor
        dir_alpha = float(np.clip(dir_alpha, 0.05, 1.2))
        dir_weight = 0.08 + 0.35 * (1.0 - state_novelty) + 0.25 * stagnation_factor + 0.2 * dominance_factor
        dir_weight = float(np.clip(dir_weight, 0.05, 0.8))
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

        self.timing_logger.end_operation("action_selection", "inference", f"action={action}, value={value.item():.3f}")
        
        return action, log_prob, value.item()

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
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        value: float, log_prob: float, done: bool):
        """Store transition in experience buffer (thread-safe)"""
        
        self.timing_logger.start_operation("store_transition", "buffer")
        lock_start = time.perf_counter()
        with self._buffer_lock:
            # Write at head, then advance (ring buffer)
            idx = self._buf_head
            self._buf_head = (self._buf_head + 1) % self._buf_capacity
            if self._buf_count < self._buf_capacity:
                self._buf_count += 1
            self._buf_states[idx, :, :] = state
            self._buf_actions[idx] = int(action)
            self._buf_rewards[idx] = float(reward)
            self._buf_values[idx] = float(value)
            self._buf_log_probs[idx] = float(log_prob)
            self._buf_dones[idx] = bool(done)
        lock_duration = time.perf_counter() - lock_start
        self.timing_logger.end_operation("store_transition", "buffer", f"buffer_size={len(self.buffer['states'])}, lock_duration={lock_duration:.4f}s")
        
        if lock_duration > 0.01:
            print(f"[yellow]store_transition lock duration {lock_duration:.4f}s")
    
    def compute_advantages(self, rewards: List[float], values: List[float], 
                          dones: List[bool], gamma: float = 0.99, 
                          lambda_: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advantages using GAE (Generalized Advantage Estimation)"""
        
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
    
    def update_policy(self) -> Dict[str, Optional[float]]:
        """Update policy using PPO (thread-safe)"""

        self.timing_logger.start_operation("update_policy", "training")
        
        with self._update_lock:
            lock_start = time.perf_counter()
            with self._buffer_lock:
                if self._buf_count < self.batch_size:
                    self.timing_logger.end_operation("update_policy", "training", "insufficient_buffer")
                    return {'policy_loss': None, 'value_loss': None, 'entropy': None}

                # Snapshot buffer and clear for concurrent writes while updating
                count = self._buf_count
                if count < self._buf_capacity:
                    # Simple slice when not wrapped
                    states_np = self._buf_states[:count].copy()
                    actions_list = self._buf_actions[:count].copy()
                    rewards_list = self._buf_rewards[:count].copy().tolist()
                    values_list = self._buf_values[:count].copy().tolist()
                    log_probs_list = self._buf_log_probs[:count].copy()
                    dones_list = self._buf_dones[:count].copy().tolist()
                else:
                    # When full and wrapped, roll so that oldest is first
                    roll = -self._buf_head
                    states_np = np.roll(self._buf_states, roll, axis=0).copy()
                    actions_list = np.roll(self._buf_actions, roll, axis=0).copy()
                    rewards_list = np.roll(self._buf_rewards, roll, axis=0).copy().tolist()
                    values_list = np.roll(self._buf_values, roll, axis=0).copy().tolist()
                    log_probs_list = np.roll(self._buf_log_probs, roll, axis=0).copy()
                    dones_list = np.roll(self._buf_dones, roll, axis=0).copy().tolist()
                # Reset pointers (arrays are reused)
                self._buf_count = 0
                self._buf_head = 0

            lock_duration = time.perf_counter() - lock_start
            self.timing_logger.log_event("buffer_snapshot", "training", lock_duration * 1000, f"states={len(states_np)}")
            
            if lock_duration > 0.2:
                print(f"[yellow]update_policy held buffer lock for {lock_duration:.3f}s with {len(states_np)} states")

            # Convert buffer to tensors _outside_ the lock so other threads can continue
            self.timing_logger.start_operation("tensor_conversion", "training")
            # Use pinned memory + non_blocking H2D for better transfer throughput
            cpu_states = torch.from_numpy(states_np).float()
            cpu_actions = torch.from_numpy(actions_list).long()
            cpu_old_log_probs = torch.from_numpy(log_probs_list).float()
            if self.use_amp and self.device and str(self.device).startswith("cuda"):
                try:
                    cpu_states = cpu_states.pin_memory()
                    cpu_actions = cpu_actions.pin_memory()
                    cpu_old_log_probs = cpu_old_log_probs.pin_memory()
                except Exception:
                    pass
                states = cpu_states.to(self.device, non_blocking=True)
                actions = cpu_actions.to(self.device, non_blocking=True)
                old_log_probs = cpu_old_log_probs.to(self.device, non_blocking=True)
            else:
                states = cpu_states.to(self.device)
                actions = cpu_actions.to(self.device)
                old_log_probs = cpu_old_log_probs.to(self.device)
            self.timing_logger.end_operation("tensor_conversion", "training", f"tensors={len(states)}")

            # Compute advantages and returns
            advantages, returns = self.compute_advantages(
                rewards_list,
                values_list,
                dones_list,
            )

            advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
            returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

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
                    batch_returns = returns[batch_indices]
                    
                    # Forward pass (AMP-enabled)
                    self.timing_logger.start_operation("batch_forward", "training")
                    if self.use_amp and self.device and str(self.device).startswith("cuda"):
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            policy_logits, values = self.model(batch_states)
                            action_dist = torch.distributions.Categorical(logits=policy_logits)
                            new_log_probs = action_dist.log_prob(batch_actions)
                            ratio = torch.exp(new_log_probs - batch_old_log_probs)
                            surr1 = ratio * batch_advantages
                            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                            policy_loss = -torch.min(surr1, surr2).mean()
                            value_loss = F.mse_loss(values.view(-1), batch_returns)
                            entropy = action_dist.entropy().mean()
                            lb_loss = self.model.latest_lb_loss if getattr(self.model, 'latest_lb_loss', None) is not None else 0.0
                            loss = (policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy + self.lb_loss_coef * (lb_loss if isinstance(lb_loss, (int, float)) else 0.0))
                    else:
                        policy_logits, values = self.model(batch_states)
                        action_dist = torch.distributions.Categorical(logits=policy_logits)
                        new_log_probs = action_dist.log_prob(batch_actions)
                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_loss = F.mse_loss(values.view(-1), batch_returns)
                        entropy = action_dist.entropy().mean()
                        lb_loss = self.model.latest_lb_loss if getattr(self.model, 'latest_lb_loss', None) is not None else 0.0
                        loss = (policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy + self.lb_loss_coef * (lb_loss if isinstance(lb_loss, (int, float)) else 0.0))
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
                
                self.timing_logger.end_operation("ppo_epoch", "training", f"batches={batch_count}")
            
            self.timing_logger.end_operation("ppo_epochs", "training")
            
            # Average losses
            num_updates = self.ppo_epochs * max(1, len(states) // self.batch_size)
            avg_policy_loss = total_policy_loss / num_updates
            avg_value_loss = total_value_loss / num_updates
            avg_entropy = total_entropy / num_updates
            avg_lb_loss = total_lb_loss / num_updates
            
            self.timing_logger.end_operation("update_policy", "training", 
                                           f"updates={num_updates}, policy_loss={avg_policy_loss:.4f}")
            
            return {
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'entropy': avg_entropy,
                'lb_loss': avg_lb_loss
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
            while not done:
                try:
                    # Select action
                    legal_actions = env.game.legal_moves()
                    if not legal_actions:
                        break

                    try:
                        action, log_prob, value = self.select_action(obs, legal_actions, env.game)
                    except Exception as sel_err:
                        print(f"Warning: select_action failed at step {episode_length}: {sel_err}")
                        action = legal_actions[0]
                        log_prob = 0.0
                        value = 0.0

                    # Take step
                    self.timing_logger.start_operation("env_step", "episode")
                    next_obs, reward, done, _, _ = env.step(action)
                    self.timing_logger.end_operation("env_step", "episode", f"action={action}, reward={reward}")
                    
                    # Calculate load balancing reward (anti-starvation, anti-dominance)
                    lb_reward = self.calculate_load_balancing_reward()

                    # Intrinsic novelty bonus on next state
                    novelty_next = self._get_state_novelty(next_obs)
                    novelty_bonus = self.novelty_bonus_coef * novelty_next

                    # Combine extrinsic and intrinsic rewards
                    total_reward = reward + lb_reward + novelty_bonus

                    # Update novelty table and stagnation tracker
                    self._update_state_novelty(next_obs)
                    if (abs(reward) < 1e-6) and (novelty_next < 0.2):
                        self._stagnation_steps += 1
                    else:
                        self._stagnation_steps = 0

                    # Store transition with combined reward
                    self.store_transition(obs, action, total_reward, value, log_prob, done)
                    
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
            self.timing_logger.end_operation("episode_loop", "episode", f"length={episode_length}")
            
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
        
        # Update policy if buffer is full or based on frequency
        losses = {'policy_loss': None, 'value_loss': None, 'entropy': None}
        self._policy_update_counter += 1
        
        # Update policy based on frequency rather than buffer size for more consistent updates
        if self._policy_update_counter >= self._policy_update_frequency:
            self.timing_logger.start_operation("policy_update", "episode")
            losses = self.update_policy()
            self.timing_logger.end_operation("policy_update", "episode")
            self._policy_update_counter = 0  # Reset counter
        elif len(self.buffer['states']) >= self.buffer_size:
            # Fallback: still update if buffer is full
            self.timing_logger.start_operation("policy_update", "episode")
            losses = self.update_policy()
            self.timing_logger.end_operation("policy_update", "episode")
            self._policy_update_counter = 0  # Reset counter
        
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