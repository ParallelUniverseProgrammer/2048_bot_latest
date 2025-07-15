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

from app.models import GameTransformer, DynamicModelConfig
from app.environment.gym_2048_env import Gym2048Env
from app.utils.action_selection import select_action_with_fallback

class PPOTrainer:
    """PPO trainer for the GameTransformer model"""
    
    def __init__(self, config=None, learning_rate: float = 3e-5, device=None):
        """Initialize PPO trainer"""
        
        # Model configuration
        if config is None:
            config = DynamicModelConfig.select_config()  # Use automatic VRAM detection
        
        self.config = config
        self.device = device or DynamicModelConfig.get_device()
        
        # Initialize model
        self.model = GameTransformer(config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # Cosine decay scheduler for smoother optimisation
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100_000, eta_min=1e-6
        )
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        # Auxiliary loss coefficients
        self.lb_loss_coef = 0.01  # weight for MoE load-balancing loss
        self.max_grad_norm = 0.5
        self.ppo_epochs = 4  # Increased epochs for more stable updates
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
        self.buffer_size = 2048  # Larger buffer for more stable gradient estimates
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        print(f"PPO Trainer initialized:")
        print(f"  Model parameters: {self.model.count_parameters():,}")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Buffer size: {self.buffer_size}")
    
    def select_action(self, state: np.ndarray, legal_actions: List[int], env_game) -> Tuple[int, float, float]:
        """Select action using current policy with fallback for invalid moves"""
        
        # Get policy and value for visualization
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy_logits, value = self.model(state_tensor)
            
            # Mask illegal actions for visualization
            action_mask = torch.full((4,), -float('inf'), device=self.device)
            action_mask[legal_actions] = 0.0
            masked_logits = policy_logits[0] + action_mask
            action_probs = F.softmax(masked_logits, dim=-1)
            
            # Store latest action probabilities for visualization
            self.latest_action_probs = action_probs.cpu().numpy().tolist()
        
        # Use fallback mechanism to select action
        action, log_prob, attention_weights = select_action_with_fallback(
            model=self.model,
            state=state,
            legal_actions=legal_actions,
            env_game=env_game,
            device=self.device,
            sample_action=True,
            max_attempts=4
        )
        
        # Get value prediction for the selected action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, value = self.model(state_tensor)
        
        return action, log_prob, value.item()
    
    def get_latest_action_probs(self) -> List[float]:
        """Get the latest action probabilities for visualization"""
        return self.latest_action_probs
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        value: float, log_prob: float, done: bool):
        """Store transition in experience buffer (thread-safe)"""
        
        with self._buffer_lock:
            self.buffer['states'].append(state.copy())
            self.buffer['actions'].append(action)
            self.buffer['rewards'].append(reward)
            self.buffer['values'].append(value)
            self.buffer['log_probs'].append(log_prob)
            self.buffer['dones'].append(done)
    
    def compute_advantages(self, rewards: List[float], values: List[float], 
                          dones: List[bool], gamma: float = 0.99, 
                          lambda_: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advantages using GAE (Generalized Advantage Estimation)"""
        
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
        
        return advantages, returns
    
    def update_policy(self) -> Dict[str, float]:
        """Update policy using PPO (thread-safe)"""

        with self._buffer_lock:
            if len(self.buffer['states']) < self.batch_size:
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

        # Convert buffer to tensors _outside_ the lock so other threads can continue
        states = torch.FloatTensor(states_np).to(self.device)
        actions = torch.LongTensor(actions_list).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs_list).to(self.device)

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
        
        for _ in range(self.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                policy_logits, values = self.model(batch_states)
                
                # Compute policy loss
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
                        self.lb_loss_coef * lb_loss)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                # Step the LR scheduler once per optimisation step
                self.scheduler.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                if isinstance(lb_loss, torch.Tensor):
                    total_lb_loss += lb_loss.item()
                else:
                    total_lb_loss += lb_loss
        
        # Clear buffer
        # self.buffer = {
        #     'states': [],
        #     'actions': [],
        #     'rewards': [],
        #     'values': [],
        #     'log_probs': [],
        #     'dones': []
        # }
        
        # Average losses
        num_updates = self.ppo_epochs * max(1, len(states) // self.batch_size)
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        avg_lb_loss = total_lb_loss / num_updates
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'lb_loss': avg_lb_loss
        }
    
    def train_episode(self, env: Gym2048Env) -> Dict[str, Any]:
        """Train for one episode"""
        
        try:
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                # Select action
                legal_actions = env.game.legal_moves()
                if not legal_actions:
                    break
                
                action, log_prob, value = self.select_action(obs, legal_actions, env.game)
                
                # Take step
                next_obs, reward, done, _, _ = env.step(action)
                
                # Store transition
                self.store_transition(obs, action, reward, value, log_prob, done)
                
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                self.total_steps += 1
        except Exception as e:
            print(f"Error in train_episode: {e}")
            import traceback
            traceback.print_exc()
            # Return default values
            return {
                'episode': self.episode_count + 1,
                'score': 0,
                'reward': 0.0,
                'length': 0,
                'losses': {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0},
                'best_score': self.best_score
            }
        
        # Update policy if buffer is full
        losses = {'policy_loss': None, 'value_loss': None, 'entropy': None}
        if len(self.buffer['states']) >= self.buffer_size:
            losses = self.update_policy()
        
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
        
        # Get model statistics
        attention_weights = self.model.get_attention_weights()
        expert_usage = self.model.get_expert_usage()
        
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
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
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