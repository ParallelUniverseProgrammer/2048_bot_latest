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

from app.models import GameTransformer, DynamicModelConfig
from app.environment.gym_2048_env import Gym2048Env
from app.utils.action_selection import select_action_with_fallback

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
    
    def __init__(self, config=None, learning_rate: float = 3e-5, device=None):
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
        self.model = GameTransformer(config).to(self.device)
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
        
        # Load balancing reward parameters - ENHANCED for tiny models
        self.lb_reward_coef = 0.3  # Increased from 0.1 to 0.3 for stronger rewards
        # Adaptive critical threshold based on number of experts
        self.lb_critical_threshold = max(0.15, 1.0 / (self.config.n_experts * 2))  # Adaptive threshold
        self.lb_early_training_boost = 1.0  # Increased from 0.5 to 1.0 for stronger early training
        self.lb_episode_threshold = 2000  # Increased from 1000 to 2000 for longer early training
        # New: Adaptive load balancing based on model size
        self.lb_adaptive_factor = max(1.0, 8.0 / self.config.n_experts)  # Stronger for smaller models
        # New: Expert diversity tracking
        self.expert_usage_history = deque(maxlen=100)  # Track recent usage patterns
        self.lb_diversity_penalty = 0.2  # Penalty for low diversity
        self.ppo_epochs = 2  # Reduced from 4 to 2 for faster updates
        self.batch_size = DynamicModelConfig.get_batch_size(config)

        # ---- NEW: lock for thread-safe experience buffer operations ----
        self._buffer_lock = threading.Lock()

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
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
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
        """Calculate load balancing reward based on expert usage distribution with enhanced anti-starvation mechanisms"""
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
        variance_penalty = normalized_variance * 2.0  # Increased penalty
        
        # 2. Expert starvation penalty (enhanced)
        starvation_penalty = 0.0
        starved_experts = 0
        for i, usage in enumerate(usage_np):
            if float(usage) < self.lb_critical_threshold:
                starvation_penalty += (self.lb_critical_threshold - float(usage)) / self.lb_critical_threshold
                starved_experts += 1
        
        # Additional penalty for multiple starved experts
        if starved_experts > 1:
            starvation_penalty *= (1.0 + starved_experts * 0.5)
        
        # 3. Entropy bonus (higher entropy = better diversity)
        epsilon = 1e-8
        log_usage = np.log(usage_np + epsilon)
        entropy = float(-np.sum(usage_np * log_usage))
        max_entropy = float(np.log(n_experts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        entropy_bonus = normalized_entropy * 1.5  # Increased bonus
        
        # 4. NEW: Diversity tracking penalty
        diversity_penalty = 0.0
        if len(self.expert_usage_history) >= 10:
            # Calculate diversity over recent history
            recent_usage = np.array(list(self.expert_usage_history)[-10:])
            avg_usage = np.mean(recent_usage, axis=0)
            
            # Penalty for experts that are consistently underused
            for i, avg_usage_val in enumerate(avg_usage):
                if avg_usage_val < ideal_usage * 0.5:  # Consistently underused
                    diversity_penalty += (ideal_usage * 0.5 - avg_usage_val) / ideal_usage
        
        # 5. NEW: Sparsity promotion (encourage using more experts)
        active_experts = float(np.sum((usage_np > ideal_usage * 0.1).astype(np.float64)))  # Experts with >10% of ideal usage
        sparsity_bonus = (active_experts / n_experts) * 0.5  # Bonus for using more experts
        
        # 6. NEW: Balance quality metric
        # Reward for having usage close to uniform distribution
        balance_quality = 1.0 - np.mean(np.abs(usage_np - ideal_usage)) / ideal_usage
        balance_bonus = balance_quality * 0.3
        
        # Combine all components
        lb_reward = (entropy_bonus + 
                    sparsity_bonus + 
                    balance_bonus - 
                    variance_penalty - 
                    starvation_penalty - 
                    diversity_penalty)
        
        # Apply adaptive factor based on model size
        lb_reward *= self.lb_adaptive_factor
        
        # Apply early training boost
        if self.episode_count < self.lb_episode_threshold:
            lb_reward *= (1.0 + self.lb_early_training_boost)
        
        # Scale by coefficient
        final_reward = float(lb_reward * self.lb_reward_coef)
        
        # Log detailed metrics for debugging
        self.timing_logger.end_operation("calculate_lb_reward", "reward", 
                                       f"reward={final_reward:.4f}, variance={normalized_variance:.4f}, "
                                       f"entropy={normalized_entropy:.4f}, starvation={starvation_penalty:.4f}, "
                                       f"diversity={diversity_penalty:.4f}, sparsity={sparsity_bonus:.4f}, "
                                       f"balance={balance_bonus:.4f}, active_experts={active_experts}/{n_experts}")
        
        return final_reward
    
    def select_action(self, state: np.ndarray, legal_actions: List[int], env_game) -> Tuple[int, float, float]:
        """Select action using current policy with fallback for invalid moves"""
        
        self.timing_logger.start_operation("action_selection", "inference")
        
        # Get policy and value for visualization
        self.timing_logger.start_operation("model_forward", "inference")
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy_logits, value = self.model(state_tensor)
            # Ensure logits are on trainer device to prevent CUDA/CPU mismatch
            policy_logits = policy_logits.to(self.device)
            
            # Mask illegal actions for visualization
            action_mask = torch.full((4,), -float('inf'), device=self.device)
            action_mask[legal_actions] = 0.0
            masked_logits = policy_logits[0] + action_mask
            action_probs = F.softmax(masked_logits, dim=-1)
            
            # Store latest action probabilities for visualization
            self.latest_action_probs = action_probs.cpu().numpy().tolist()
        self.timing_logger.end_operation("model_forward", "inference", f"legal_actions={len(legal_actions)}")
        
        # Use fallback mechanism to select action
        self.timing_logger.start_operation("action_fallback", "inference")
        action, log_prob, attention_weights = select_action_with_fallback(
            model=self.model,
            state=state,
            legal_actions=legal_actions,
            env_game=env_game,
            device=self.device,
            sample_action=True,
            max_attempts=4
        )
        self.timing_logger.end_operation("action_fallback", "inference", f"selected_action={action}")
        
        # Get value prediction for the selected action
        self.timing_logger.start_operation("value_prediction", "inference")
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, value = self.model(state_tensor)
        self.timing_logger.end_operation("value_prediction", "inference")
        
        self.timing_logger.end_operation("action_selection", "inference", f"action={action}, value={value.item():.3f}")
        
        return action, log_prob, value.item()
    
    def get_latest_action_probs(self) -> List[float]:
        """Get the latest action probabilities for visualization"""
        return self.latest_action_probs
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        value: float, log_prob: float, done: bool):
        """Store transition in experience buffer (thread-safe)"""
        
        self.timing_logger.start_operation("store_transition", "buffer")
        lock_start = time.perf_counter()
        with self._buffer_lock:
            self.buffer['states'].append(state.copy())
            self.buffer['actions'].append(action)
            self.buffer['rewards'].append(reward)
            self.buffer['values'].append(value)
            self.buffer['log_probs'].append(log_prob)
            self.buffer['dones'].append(done)
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
        
        lock_start = time.perf_counter()
        with self._buffer_lock:
            if len(self.buffer['states']) < self.batch_size:
                self.timing_logger.end_operation("update_policy", "training", "insufficient_buffer")
                return {'policy_loss': None, 'value_loss': None, 'entropy': None}

            # Snapshot buffer and clear for concurrent writes while updating
            states_np = np.array(self.buffer['states'])
            actions_list = self.buffer['actions']
            rewards_list = self.buffer['rewards']
            values_list = self.buffer['values']
            log_probs_list = self.buffer['log_probs']
            dones_list = self.buffer['dones']
            # Reset buffer
            self.buffer = {key: [] for key in self.buffer}

        lock_duration = time.perf_counter() - lock_start
        self.timing_logger.log_event("buffer_snapshot", "training", lock_duration * 1000, f"states={len(states_np)}")
        
        if lock_duration > 0.2:
            print(f"[yellow]update_policy held buffer lock for {lock_duration:.3f}s with {len(states_np)} states")

        # Convert buffer to tensors _outside_ the lock so other threads can continue
        self.timing_logger.start_operation("tensor_conversion", "training")
        states = torch.FloatTensor(states_np).to(self.device)
        actions = torch.LongTensor(actions_list).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs_list).to(self.device)
        self.timing_logger.end_operation("tensor_conversion", "training", f"tensors={len(states)}")

        # Compute advantages and returns
        advantages, returns = self.compute_advantages(
            rewards_list,
            values_list,
            dones_list,
        )

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # PPO update
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_lb_loss = 0.0
        
        self.timing_logger.start_operation("ppo_epochs", "training", f"epochs={self.ppo_epochs}")
        for epoch in range(self.ppo_epochs):
            self.timing_logger.start_operation("ppo_epoch", "training", f"epoch={epoch}")
            
            # Shuffle data
            indices = torch.randperm(len(states))
            
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
                
                # Forward pass
                self.timing_logger.start_operation("batch_forward", "training")
                policy_logits, values = self.model(batch_states)
                self.timing_logger.end_operation("batch_forward", "training", f"batch_size={len(batch_states)}")
                
                # Compute policy loss
                self.timing_logger.start_operation("loss_computation", "training")
                action_dist = torch.distributions.Categorical(logits=policy_logits)
                new_log_probs = action_dist.log_prob(batch_actions)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(values.view(-1), batch_returns)
                
                # Compute entropy
                entropy = action_dist.entropy().mean()
                
                # Auxiliary load-balancing loss from MoE layer
                lb_loss = self.model.latest_lb_loss if getattr(self.model, 'latest_lb_loss', None) is not None else 0.0
                
                # Total loss (PPO + regularisers)
                loss = (policy_loss +
                        self.value_loss_coef * value_loss -
                        self.entropy_coef * entropy +
                        self.lb_loss_coef * (lb_loss if isinstance(lb_loss, (int, float)) else 0.0))
                self.timing_logger.end_operation("loss_computation", "training")
                
                # Backward pass
                self.timing_logger.start_operation("backward_pass", "training")
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                # Step the LR scheduler once per optimisation step
                self.scheduler.step()
                self.timing_logger.end_operation("backward_pass", "training")
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                if isinstance(lb_loss, torch.Tensor):
                    total_lb_loss += lb_loss.item()
                elif isinstance(lb_loss, (int, float)):
                    total_lb_loss += lb_loss
                
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
                    
                    action, log_prob, value = self.select_action(obs, legal_actions, env.game)
                    
                    # Take step
                    self.timing_logger.start_operation("env_step", "episode")
                    next_obs, reward, done, _, _ = env.step(action)
                    self.timing_logger.end_operation("env_step", "episode", f"action={action}, reward={reward}")
                    
                    # Calculate load balancing reward
                    lb_reward = self.calculate_load_balancing_reward()
                    
                    # Combine game reward with load balancing reward
                    total_reward = reward + lb_reward
                    
                    # Store transition with combined reward
                    self.store_transition(obs, action, total_reward, value, log_prob, done)
                    
                    obs = next_obs
                    episode_reward += reward
                    episode_length += 1
                    self.total_steps += 1
                    
                    # CRITICAL FIX: Yield control to event loop every 10 steps to prevent blocking
                    if episode_length % 10 == 0:
                        await asyncio.sleep(0.001)  # Minimal yield to allow WebSocket processing
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