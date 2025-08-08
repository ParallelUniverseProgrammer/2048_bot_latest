"""training_manager.py
======================
Single–responsibility *manager* running the training loop **asynchronously** so
we can control it from HTTP endpoints and stream metrics to the frontend via
WebSockets.

Now integrated with PPO trainer and real neural network model.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional, Dict, Any, List
import os
import json
from datetime import datetime
import threading
from pathlib import Path

import numpy as np
import torch

from app.api.websocket_manager import WebSocketManager
from app.environment.gym_2048_env import Gym2048Env
from app.training.ppo_trainer import PPOTrainer
from app.models.checkpoint_metadata import CheckpointManager


class TrainingManager:
    """Manage training lifecycle and push streaming metrics."""

    def __init__(self, websocket_manager: WebSocketManager, n_envs: int = 2):  # Default; will auto-scale
        self.ws_manager = websocket_manager
        
        # Initialize timing logger
        self.timing_logger = self._create_timing_logger()
        self.timing_logger.start_operation("manager_init", "setup", f"n_envs={n_envs}")
        
        # Auto-scale number of environments based on VRAM + CPU cores
        self.timing_logger.start_operation("env_creation", "setup")
        try:
            from app.models.model_config import DynamicModelConfig
            available_vram = DynamicModelConfig.get_available_vram()
        except Exception:
            available_vram = 0.0
        try:
            import multiprocessing
            cpu_cores = multiprocessing.cpu_count()
        except Exception:
            cpu_cores = 2
        # Base on VRAM tiers; envs are light but training shares a single model
        if available_vram >= 24.0:
            auto_envs = min(8, max(4, cpu_cores // 2))
        elif available_vram >= 12.0:
            auto_envs = min(6, max(3, (cpu_cores + 1) // 2))
        elif available_vram >= 8.0:
            auto_envs = min(4, max(2, (cpu_cores + 1) // 3))
        else:
            auto_envs = max(2, min(3, cpu_cores // 2))
        self._auto_envs = max(n_envs, auto_envs)
        self.envs: List[Gym2048Env] = [Gym2048Env() for _ in range(self._auto_envs)]
        self.timing_logger.end_operation("env_creation", "setup", f"created={len(self.envs)}_environments (auto)")
        
        self._task: Optional[asyncio.Task] = None
        
        # Initialize PPO trainer with default config (will be updated when training starts)
        self.timing_logger.start_operation("trainer_init", "setup")
        # Validation clones are expensive; rely on env.legal_moves + fast can-move
        self.trainer = PPOTrainer(validate_moves=False, compile_model=False)
        self.timing_logger.end_operation("trainer_init", "setup")
        
        # MEMORY OPTIMIZATION: Use single shared trainer instead of multiple trainers
        # This significantly reduces GPU memory usage
        self.timing_logger.start_operation("shared_trainer_setup", "setup")
        self.env_trainers = [self.trainer] * len(self.envs)  # All environments share the same trainer
        self.timing_logger.end_operation("shared_trainer_setup", "setup", f"shared_trainer_for={len(self.envs)}_environments")
        
        self.current_config = None
        
        # run-time state
        self.is_training: bool = False
        self.is_paused: bool = False
        self.current_episode: int = 0
        self.total_episodes: int = 10_000
        self.is_initializing: bool = False  # NEW FIELD
        
        # Checkpoint management
        self.checkpoint_dir = os.getenv("CHECKPOINTS_DIR", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        
        # NEW: Checkpoint configuration
        self._checkpoint_interval = 50  # Default: save every 50 episodes
        self._long_run_mode = False  # Default: keep all checkpoints
        self._current_run_id = None  # Track current training run
        self._last_checkpoint_path = None  # Track last checkpoint for long run mode
        
        # Metrics tracking
        self._start_time: Optional[float] = None
        self._game_lengths: List[int] = []
        self._episode_start_times: List[float] = []
        
        # Enhanced metrics tracking
        self._recent_scores: List[int] = []  # Last 100 scores for trend analysis
        self._recent_losses: List[float] = []  # Last 100 losses for trend analysis
        self._max_tiles_achieved: List[int] = []  # Track max tiles achieved
        self._efficiency_metrics: Dict[str, float] = {}  # Training efficiency metrics
        self._load_balancing_metrics: List[float] = []  # Track load balancing rewards
        # NEW: Enhanced load balancing tracking
        self._expert_starvation_count: int = 0  # Count of episodes with expert starvation
        self._expert_usage_history: List[List[float]] = []  # Track expert usage over time
        self._sparsity_scores: List[float] = []  # Track sparsity promotion scores
        self._load_balance_quality: List[float] = []  # Track load balance quality
        
        # ENHANCED: Expert starvation tracking with model-size awareness
        self._expert_starvation_by_model_size: Dict[str, int] = {
            'tiny': 0,
            'small': 0, 
            'medium': 0,
            'large': 0
        }
        self._expert_recovery_tracking: Dict[int, List[float]] = {}  # Track recovery of specific experts
        self._starvation_severity_tracking: List[float] = []  # Track how severe starvation is
        
        # Optimization: Rate limit broadcasts adaptively based on episode time
        self._last_broadcast_time = 0.0
        self._broadcast_interval = 0.05  # Faster baseline cadence for snappier UI
        
        # Thread safety for load balancing calculations
        self._lb_lock = threading.Lock()
        
        self.timing_logger.end_operation("manager_init", "setup")

    def __del__(self):
        """Cleanup when training manager is destroyed"""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup
    
    def cleanup(self):
        """Clean up resources and free GPU memory"""
        print("🧹 Cleaning up training manager resources...")
        
        # Stop training if running
        if self.is_training:
            self.stop_sync()
        
        # Clean up trainers
        if hasattr(self, 'trainer') and self.trainer:
            self._cleanup_trainer(self.trainer)
            self.trainer = None
        
        # Clean up environment trainers
        if hasattr(self, 'env_trainers'):
            for trainer in self.env_trainers:
                if trainer:
                    self._cleanup_trainer(trainer)
            self.env_trainers.clear()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    
    def _cleanup_trainer(self, trainer):
        """Clean up a single trainer instance"""
        try:
            if hasattr(trainer, 'model') and trainer.model:
                # Move model to CPU and delete
                trainer.model.cpu()
                del trainer.model
                trainer.model = None
            
            if hasattr(trainer, 'optimizer') and trainer.optimizer:
                del trainer.optimizer
                trainer.optimizer = None
            
            # Clear buffers
            if hasattr(trainer, 'buffer'):
                trainer.buffer.clear()
            
            # Clear histories
            if hasattr(trainer, 'loss_history'):
                trainer.loss_history.clear()
            if hasattr(trainer, 'score_history'):
                trainer.score_history.clear()
            
        except Exception as e:
            print(f"Warning: Error cleaning up trainer: {e}")

    def _create_timing_logger(self):
        """Create timing logger for training manager"""
        from app.training.ppo_trainer import TimingLogger
        return TimingLogger("logs/training_manager_timing.log")

    # ------------------------------------------------------------------ Checkpoint Configuration
    def set_checkpoint_interval(self, interval: int):
        """Set the interval between automatic checkpoints (in episodes)"""
        if interval < 1:
            raise ValueError("Checkpoint interval must be at least 1 episode")
        self._checkpoint_interval = interval
        print(f"Checkpoint interval set to {interval} episodes")

    def set_long_run_mode(self, enabled: bool):
        """Enable/disable long run mode (only keep latest checkpoint from current run)"""
        self._long_run_mode = enabled
        if enabled:
            self._current_run_id = f"run_{int(time.time())}"
            # Reset last checkpoint tracking to avoid deleting checkpoints from a previous run
            self._last_checkpoint_path = None
            print(f"Long run mode enabled with run ID: {self._current_run_id}")
            
            # REMOVED: Don't clean up existing checkpoints when enabling long run mode
            # This allows users to keep their existing checkpoints and only affect future ones
            print("Long run mode will only affect future checkpoints from this run")
        else:
            self._current_run_id = None
            # Also clear last checkpoint pointer when disabling
            self._last_checkpoint_path = None
            print("Long run mode disabled")

    def _cleanup_previous_run_checkpoints(self):
        """Clean up checkpoints from previous runs when long run mode is enabled"""
        # REMOVED: This method is no longer used since we don't delete existing checkpoints
        # when long run mode is first enabled
        pass

    def get_checkpoint_config(self) -> Dict[str, Any]:
        """Get current checkpoint configuration"""
        return {
            'interval': self._checkpoint_interval,
            'long_run_mode': self._long_run_mode,
            'current_run_id': self._current_run_id,
            'next_checkpoint_episode': self._get_next_checkpoint_episode()
        }

    def _get_next_checkpoint_episode(self) -> int:
        """Calculate the next episode when a checkpoint will be saved"""
        if self.current_episode == 0:
            return self._checkpoint_interval
        return ((self.current_episode // self._checkpoint_interval) + 1) * self._checkpoint_interval

    # ------------------------------------------------------------------ Control
    def update_config(self, config_dict: Dict[str, Any]):
        """Update training configuration"""
        self.timing_logger.start_operation("update_config", "config", f"config={config_dict}")
        
        from app.models.model_config import DynamicModelConfig
        
        # Get the model profile from config
        model_size = config_dict.get('model_size', 'base')  # lightning | base | expert

        # Optionally allow forcing by VRAM, else select by profile
        target_vram = config_dict.get('target_vram', None)
        self.current_config = DynamicModelConfig.select_config(target_vram=target_vram, target_profile=model_size)
        
        # Reinitialize trainer with new config
        self.timing_logger.start_operation("trainer_reinit", "config")
        self.trainer = PPOTrainer(
            config=self.current_config,
            learning_rate=config_dict.get('learning_rate', 0.0003),
            validate_moves=False,
            compile_model=False,
        )
        
        # MEMORY OPTIMIZATION: Update shared trainer references
        self.timing_logger.start_operation("shared_trainer_update", "config")
        self.env_trainers = [self.trainer] * len(self.envs)  # Update all references to use new trainer
        self.timing_logger.end_operation("shared_trainer_update", "config", f"updated={len(self.env_trainers)}_references")
        
        self.timing_logger.end_operation("trainer_reinit", "config")
        
        self.timing_logger.end_operation("update_config", "config", f"model_profile={model_size}, params={self.current_config.estimated_params:.1f}M (active ~{self.current_config.estimated_active_params:.1f}M)")

        print(f"Updated training config: {model_size} profile with {self.current_config.estimated_params:.1f}M params, ~{self.current_config.estimated_active_params:.1f}M active")

    def load_checkpoint_trainer(self, config, checkpoint_path: str):
        """Load a checkpoint and create trainer with the correct configuration"""
        self.timing_logger.start_operation("load_checkpoint_trainer", "checkpoint", f"checkpoint_path={checkpoint_path}")
        
        # Set the config from checkpoint
        self.current_config = config
        
        # Create new trainer with checkpoint config
        self.timing_logger.start_operation("trainer_checkpoint_init", "checkpoint")
        self.trainer = PPOTrainer(
            config=config,
            learning_rate=0.0003,  # Default learning rate, can be adjusted via UI later
            validate_moves=False,
            compile_model=False,
        )
        
        # CRITICAL FIX: Update shared trainer references to use the new trainer
        self.timing_logger.start_operation("shared_trainer_checkpoint_update", "checkpoint")
        self.env_trainers = [self.trainer] * len(self.envs)
        self.timing_logger.end_operation("shared_trainer_checkpoint_update", "checkpoint", f"updated={len(self.env_trainers)}_references")
        
        # Load checkpoint weights
        self.timing_logger.start_operation("load_checkpoint_weights", "checkpoint")
        self.trainer.load_checkpoint(checkpoint_path)
        self.timing_logger.end_operation("load_checkpoint_weights", "checkpoint", f"episode_count={self.trainer.episode_count}")
        
        # Sync manager state with trainer state
        self.current_episode = self.trainer.episode_count
        self._game_lengths = []
        self._episode_start_times = []
        self._start_time = time.time()
        self.is_paused = False
        
        self.timing_logger.end_operation("trainer_checkpoint_init", "checkpoint")
        self.timing_logger.end_operation("load_checkpoint_trainer", "checkpoint", f"config={config}, episode={self.current_episode}")
        
        print(f"Loaded checkpoint trainer with config: {config}")
        print(f"Resumed from episode: {self.current_episode}")
        print(f"Updated {len(self.env_trainers)} environment trainers to use new trainer")
    
    def start(self):
        self.timing_logger.start_operation("training_start", "control")
        
        if self._task and not self._task.done():
            # already running
            self.timing_logger.end_operation("training_start", "control", "already_running")
            return
            
        self.is_training = True
        self.is_paused = False
        self._start_time = time.time()  # Track start time for speed calculation
        
        # NEW: Initialize run tracking for long run mode
        if self._long_run_mode and not self._current_run_id:
            self._current_run_id = f"run_{int(time.time())}"
            print(f"Training run started with ID: {self._current_run_id}")
        
        # Send training start message (handle both sync and async contexts)
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # We're in an async context, create the task
            asyncio.create_task(self._send_training_start_message())
        except RuntimeError:
            # No running event loop, we're in a sync context
            # Just log the message instead of sending via WebSocket
            print("Training session started (no WebSocket context)")
        
        # schedule on current event loop
        try:
            loop = asyncio.get_event_loop()
            self._task = loop.create_task(self._training_loop())
        except RuntimeError:
            # No event loop available, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._task = loop.create_task(self._training_loop())
        
        self.timing_logger.end_operation("training_start", "control")
    
    def reset_to_fresh_model(self):
        """Reset training state and create a fresh model with current config"""
        self.timing_logger.start_operation("reset_model", "control")
        
        # Clean up existing trainers first
        if hasattr(self, 'trainer') and self.trainer:
            self._cleanup_trainer(self.trainer)
        
        if hasattr(self, 'env_trainers'):
            for trainer in self.env_trainers:
                if trainer:
                    self._cleanup_trainer(trainer)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reset training state
        self.current_episode = 0
        self.is_paused = False
        self._start_time = None
        self._game_lengths = []
        self._episode_start_times = []
        
        # Clear all history lists to prevent data inconsistency between sessions
        self._recent_scores = []
        self._recent_losses = []
        self._max_tiles_achieved = []
        self._load_balancing_metrics = []
        self._expert_usage_history = []
        
        # Create a fresh trainer with the current config (or default if none set)
        if self.current_config is None:
            # Use default config if none is set
            from app.models.model_config import DynamicModelConfig
            self.current_config = DynamicModelConfig.select_config()
        
        # Reinitialize trainer with fresh state
        self.timing_logger.start_operation("trainer_fresh_init", "control")
        self.trainer = PPOTrainer(
            config=self.current_config,
            learning_rate=0.0003  # Default learning rate
        )
        
        # MEMORY OPTIMIZATION: Update shared trainer references
        self.timing_logger.start_operation("shared_trainer_reset", "control")
        self.env_trainers = [self.trainer] * len(self.envs)  # Update all references to use new trainer
        self.timing_logger.end_operation("shared_trainer_reset", "control", f"updated={len(self.env_trainers)}_references")
        
        self.timing_logger.end_operation("trainer_fresh_init", "control")
        
        self.timing_logger.end_operation("reset_model", "control", f"config={self.current_config}")
        
        print(f"Reset to fresh model with config: {self.current_config}")
        if torch.cuda.is_available():
            print(f"GPU memory after reset: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    async def _send_training_start_message(self):
        """Send training start message via WebSocket"""
        self.timing_logger.start_operation("send_start_message", "websocket")
        
        try:
            await self.ws_manager.broadcast_high_priority({
                'type': 'training_start',
                'message': 'Training session started',
                'model_config': self.current_config.__dict__ if self.current_config else {},
                'status': 'starting'
            })
            self.timing_logger.end_operation("send_start_message", "websocket", "success")
        except Exception as e:
            self.timing_logger.end_operation("send_start_message", "websocket", f"error={str(e)}")
            print(f"Error sending training start message: {e}")

    def pause(self):
        self.timing_logger.log_event("pause", "control")
        self.is_paused = True

    def resume(self):
        self.timing_logger.log_event("resume", "control")
        if self.is_training:
            self.is_paused = False

    async def stop(self):
        self.timing_logger.start_operation("training_stop", "control")
        
        self.is_training = False
        self.is_paused = False
        if self._task:
            if not self._task.done():
                self._task.cancel()
                print("Training task cancellation requested.")
            else:
                print("Training task already completed.")
        
        self.timing_logger.end_operation("training_stop", "control")
        # Do not await self._task; return immediately
        # Optionally, schedule a background cleanup if needed

    def stop_sync(self):
        """Synchronous version of stop for non-async contexts"""
        self.timing_logger.start_operation("training_stop_sync", "control")
        
        self.is_training = False
        self.is_paused = False
        if self._task:
            if not self._task.done():
                self._task.cancel()
                print("Training task cancellation requested.")
            else:
                print("Training task already completed.")
        
        # Clean up memory after stopping
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory after stop: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        
        self.timing_logger.end_operation("training_stop_sync", "control")

    # ------------------------------------------------------------------ Loop
    async def _training_loop(self):
        self.timing_logger.start_operation("training_loop", "main")
        
        try:
            while self.is_training and self.current_episode < self.total_episodes:
                if self.is_paused:
                    await asyncio.sleep(0.5)
                    continue

                try:
                    # Send new episode start message for first few episodes (reduced frequency)
                    if self.current_episode % 50 == 0:  # Reduced from 25 to 50 for less overhead
                        self.timing_logger.start_operation("send_episode_message", "websocket")
                        try:
                            await self.ws_manager.broadcast_high_priority({
                                'type': 'new_episode_started',
                                'episode': self.current_episode,
                                'message': f'Starting training episode {self.current_episode}'
                            })
                            self.timing_logger.end_operation("send_episode_message", "websocket", "success")
                        except Exception as e:
                            self.timing_logger.end_operation("send_episode_message", "websocket", f"error={str(e)}")
                            print(f"Error sending new episode message: {e}")
                    
                    # Record episode start time
                    episode_start_time = time.time()
                    
                    # Track episode start times for progress estimation
                    if not hasattr(self, '_episode_start_times'):
                        self._episode_start_times = []
                    self._episode_start_times.append(episode_start_time)
                    
                    # Keep only last 20 episode times to avoid memory bloat
                    if len(self._episode_start_times) > 20:
                        self._episode_start_times = self._episode_start_times[-20:]
                    
                    # Train one episode per environment concurrently with robust safety
                    if self.current_episode < 5:  # Debug logging for first few iterations
                        print(f"Training {len(self.envs)} environments concurrently...")

                    self.timing_logger.start_operation("concurrent_training", "training", f"n_envs={len(self.envs)}")

                    # Create tasks for each environment
                    tasks = []
                    for i, env in enumerate(self.envs):
                        async def run_env(idx: int, environment: Gym2048Env):
                            try:
                                return await self.env_trainers[idx].train_episode(environment)
                            except Exception as err:
                                print(f"Environment {idx} training failed: {err}")
                                import traceback
                                traceback.print_exc()
                                return {
                                    'episode': self.current_episode + 1,
                                    'score': 0,
                                    'reward': 0.0,
                                    'length': 0,
                                    'losses': {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0},
                                    'best_score': self.env_trainers[0].best_score if self.env_trainers else 0
                                }
                        tasks.append(asyncio.create_task(run_env(i, env)))

                    # Stream the first finished result immediately to reduce perceived latency
                    episode_results = []
                    first_broadcast_done = False
                    for future in asyncio.as_completed(tasks):
                        res = await future
                        if isinstance(res, Exception):
                            print(f"Environment task raised exception: {res}")
                            res = {
                                'episode': self.current_episode + 1,
                                'score': 0,
                                'reward': 0.0,
                                'length': 0,
                                'losses': {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0},
                                'best_score': self.env_trainers[0].best_score if self.env_trainers else 0
                            }
                        episode_results.append(res)

                        # Early, lightweight broadcast from the first finished environment
                        if not first_broadcast_done:
                            try:
                                metrics_preview = self._build_metrics(res, self.envs[-1])
                                metrics_preview['is_training_active'] = True
                                metrics_preview['next_episode_estimate'] = self._estimate_next_episode_time()
                                now = time.time()
                                if now - self._last_broadcast_time >= self._broadcast_interval:
                                    await asyncio.wait_for(
                                        self.ws_manager.broadcast(metrics_preview, priority="normal"),
                                        timeout=0.5
                                    )
                                    self._last_broadcast_time = now
                                first_broadcast_done = True
                            except Exception as e:
                                print(f"Warning: early metrics preview broadcast failed: {e}")

                    self.timing_logger.end_operation("concurrent_training", "training", f"results={len(episode_results)}")
                    if self.current_episode < 5:  # Debug logging
                        print(f"Concurrent training completed, got {len(episode_results)} results")

                    # Convenience: last result for logging / checkpoint metrics
                    last_res = episode_results[-1] if episode_results else {}
                    
                    # Record episode metrics for each environment
                    self.timing_logger.start_operation("metrics_processing", "training")
                    episode_end_time = time.time()
                    for result in episode_results:
                        game_length = result.get('length', 0)
                        score = result.get('score', 0)
                        losses = result.get('losses', {})
                        policy_loss = losses.get('policy_loss', 0) if losses else 0
                        value_loss = losses.get('value_loss', 0) if losses else 0
                        loss = (policy_loss or 0) + (value_loss or 0)
                        
                        self._game_lengths.append(game_length)
                        self._recent_scores.append(score)
                        if loss > 0:
                            self._recent_losses.append(loss)
                        
                        # Track max tile achieved (estimate from score)
                        max_tile = self._estimate_max_tile_from_score(score)
                        self._max_tiles_achieved.append(max_tile)
                        
                        # Track load balancing metrics from trainer (thread-safe)
                        # Use the first trainer as representative for metrics
                        with self._lb_lock:
                            lb_reward = self.env_trainers[0].calculate_load_balancing_reward()
                            self._load_balancing_metrics.append(lb_reward)
                        
                        # NEW: Enhanced load balancing tracking (thread-safe)
                        # Get expert usage from model (use first trainer as representative)
                        with self._lb_lock:
                            expert_usage = self.env_trainers[0].model.get_expert_usage()
                            if expert_usage is not None:
                                usage_list = expert_usage.tolist() if hasattr(expert_usage, 'tolist') else list(expert_usage)
                                self._expert_usage_history.append([float(u) for u in usage_list])
                                
                                # ENHANCED: Check for expert starvation with model-size awareness
                                n_experts = len(usage_list)
                                ideal_usage = 1.0 / n_experts
                                
                                # Determine model size for tracking
                                model_size = 'medium'  # default
                                if n_experts <= 4:
                                    model_size = 'small' if (self.current_config and self.current_config.d_model > 60) else 'tiny'
                                elif n_experts <= 6:
                                    model_size = 'medium'
                                else:
                                    model_size = 'large'
                                
                                # ENHANCED: Track starvation with severity
                                starved_experts = 0
                                total_starvation_severity = 0.0
                                
                                for i, usage in enumerate(usage_list):
                                    if usage < ideal_usage * 0.25:  # Starved expert
                                        starved_experts += 1
                                        # Calculate severity (how far below threshold)
                                        severity = (ideal_usage * 0.25 - usage) / (ideal_usage * 0.25)
                                        total_starvation_severity += severity
                                        
                                        # Track recovery for this expert
                                        if i not in self._expert_recovery_tracking:
                                            self._expert_recovery_tracking[i] = []
                                        self._expert_recovery_tracking[i].append(float(usage))
                                        
                                        # Keep only recent history
                                        if len(self._expert_recovery_tracking[i]) > 20:
                                            self._expert_recovery_tracking[i] = self._expert_recovery_tracking[i][-20:]
                                
                                if starved_experts > 0:
                                    self._expert_starvation_count += 1
                                    self._expert_starvation_by_model_size[model_size] += 1
                                    avg_severity = total_starvation_severity / starved_experts
                                    self._starvation_severity_tracking.append(float(avg_severity))
                                
                                # Calculate sparsity score (how many experts are actively used)
                                active_experts = sum(1 for usage in usage_list if usage > ideal_usage * 0.1)
                                sparsity_score = active_experts / n_experts
                                self._sparsity_scores.append(sparsity_score)
                                
                                # Calculate load balance quality
                                variance = np.var(usage_list)
                                max_variance = (1.0 - ideal_usage) ** 2
                                normalized_variance = variance / max_variance if max_variance > 0 else 0.0
                                balance_quality = 1.0 - normalized_variance
                                self._load_balance_quality.append(float(balance_quality))
                    
                    self._episode_start_times.append(episode_start_time)
                    
                    # Keep only recent data to avoid memory issues
                    if len(self._game_lengths) > 1000:
                        self._game_lengths = self._game_lengths[-1000:]
                    if len(self._episode_start_times) > 1000:
                        self._episode_start_times = self._episode_start_times[-1000:]
                    if len(self._recent_scores) > 100:
                        self._recent_scores = self._recent_scores[-100:]
                    if len(self._recent_losses) > 100:
                        self._recent_losses = self._recent_losses[-100:]
                    if len(self._max_tiles_achieved) > 100:
                        self._max_tiles_achieved = self._max_tiles_achieved[-100:]
                    if len(self._load_balancing_metrics) > 100:
                        self._load_balancing_metrics = self._load_balancing_metrics[-100:]
                    
                    self.timing_logger.end_operation("metrics_processing", "training", f"processed={len(episode_results)}_results")
                    
                    # FIX: Broadcast metrics after every episode (reduced from 2 episodes)
                    # This provides immediate feedback to users
                    if episode_results:
                        self.timing_logger.start_operation("metrics_broadcast", "websocket")
                        
                        # Build metrics for the last environment only (representative sample)
                        metrics = self._build_metrics(episode_results[-1], self.envs[-1])
                        
                        # Add animated progress indicator
                        metrics['is_training_active'] = True
                        metrics['next_episode_estimate'] = self._estimate_next_episode_time()
                        
                        # CRITICAL FIX: Add timeout and error handling for WebSocket broadcast
                        try:
                            await asyncio.wait_for(
                                self.ws_manager.broadcast(metrics, priority="normal"),
                                timeout=1.0  # 1 second timeout to prevent blocking
                            )
                        except asyncio.TimeoutError:
                            print("Warning: WebSocket broadcast timed out, skipping metrics update")
                        except Exception as e:
                            print(f"Warning: WebSocket broadcast failed: {e}")
                        
                        self.timing_logger.end_operation("metrics_broadcast", "websocket", "immediate_feedback")

                    # Extract metrics we need later for checkpoint bookkeeping
                    training_speed = self._calculate_training_speed_with_checkpoint_offset()
                    
                    avg_game_length = sum(self._game_lengths) / len(self._game_lengths) if self._game_lengths else 0
                    
                    # Debug logging – show stats from the last environment in the batch
                    if self.current_episode % 20 == 0 and episode_results:  # Reduced frequency
                        print(
                            f"Episode {self.current_episode}: Score={last_res['score']}, "
                            f"Params={self.env_trainers[0].model.count_parameters() / 1e6:.1f}M, "
                            f"LR={self.env_trainers[0].optimizer.param_groups[0]['lr']:.6f}"
                        )

                    # Update current episode - increment by number of environments trained
                    if episode_results:
                        # Each iteration trains one episode per environment, so increment by number of environments
                        self.current_episode += len(self.envs)
                        print(f"Training progress: {self.current_episode}/{self.total_episodes} episodes completed")
                        
                        # Send training status update to frontend (reduced frequency)
                        if self.current_episode % 20 == 0:  # Reduced from every iteration to every 20 episodes
                            self.timing_logger.start_operation("status_update", "websocket")
                            try:
                                # CRITICAL FIX: Add timeout for status updates to prevent blocking
                                await asyncio.wait_for(
                                    self.ws_manager.broadcast({
                                        'type': 'training_status_update',
                                        'is_training': True,
                                        'is_paused': False,
                                        'current_episode': self.current_episode,
                                        'total_episodes': self.total_episodes
                                    }),
                                    timeout=0.5  # 500ms timeout for status updates
                                )
                                self.timing_logger.end_operation("status_update", "websocket", "success")
                            except asyncio.TimeoutError:
                                self.timing_logger.end_operation("status_update", "websocket", "timeout")
                                print("Warning: Training status update timed out")
                            except Exception as e:
                                self.timing_logger.end_operation("status_update", "websocket", f"error={str(e)}")
                                print(f"Error broadcasting training status: {e}")
                    else:
                        # If no results, increment by 1 to prevent infinite loop
                        self.current_episode += 1
                        print(f"Warning: No episode results, incrementing episode count to {self.current_episode}")
                    
                    # Save checkpoint periodically
                    if self.current_episode % self._checkpoint_interval == 0:
                        await self._save_checkpoint(training_speed, avg_game_length, metrics if 'metrics' in locals() else {})
                    
                    await asyncio.sleep(0.001)  # Reduced from 0.005 to 0.001 for faster feedback
                    
                except Exception as e:
                    self.timing_logger.log_event("episode_error", "training", 0, f"episode={self.current_episode}, error={str(e)}")
                    print(f"Error in training episode {self.current_episode}: {e}")
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(1)  # Wait before retrying
                    
        except Exception as e:
            self.timing_logger.log_event("fatal_error", "training", 0, f"error={str(e)}")
            print(f"Fatal error in training loop: {e}")
            import traceback
            traceback.print_exc()
            self.is_training = False
            
        # Training complete
        self.timing_logger.start_operation("training_complete", "websocket")
        await self.ws_manager.broadcast_high_priority({
            "type": "training_complete",
            "message": "Training finished",
            "final_episode": self.current_episode,
        })
        self.timing_logger.end_operation("training_complete", "websocket")
        
        self.timing_logger.end_operation("training_loop", "main", f"final_episode={self.current_episode}")
        
        # Write timing summaries
        self.timing_logger.start_operation("write_timing_summaries", "cleanup")
        self.write_timing_summary()
        if self.trainer is not None:
            self.trainer.write_timing_summary()
        self.timing_logger.end_operation("write_timing_summaries", "cleanup")
        
        # Generate performance analysis report
        try:
            from app.utils.timing_analyzer import TimingAnalyzer
            analyzer = TimingAnalyzer()
            report_file = analyzer.generate_report()
            print(f"Performance analysis report generated: {report_file}")
        except Exception as e:
            print(f"Warning: Could not generate performance analysis report: {e}")
        
        self.is_training = False
        self.current_episode = 0

    async def _save_checkpoint(self, training_speed: float, avg_game_length: float, metrics: Dict[str, Any]):
        """Save checkpoint with timing information"""
        self.timing_logger.start_operation("checkpoint_save", "io", f"episode={self.current_episode}")
        
        checkpoint_id = f"checkpoint_episode_{self.current_episode}"
        checkpoint_path = Path(self.checkpoint_dir) / f"{checkpoint_id}.pt"
        checkpoint_path = checkpoint_path.resolve()  # Ensure absolute path
        
        # NEW: Handle long run mode - delete previous checkpoint from this run
        if self._long_run_mode and self._current_run_id and self._last_checkpoint_path:
            previous_id = self._last_checkpoint_path.stem
            try:
                # Prefer deleting via CheckpointManager so metadata is also removed
                deleted = False
                try:
                    deleted = self.checkpoint_manager.delete_checkpoint(previous_id)
                except Exception as meta_e:
                    print(f"Warning: Could not delete metadata for previous checkpoint {previous_id}: {meta_e}")
                # Fallback: ensure file is gone even if metadata deletion path failed
                if self._last_checkpoint_path.exists():
                    try:
                        self._last_checkpoint_path.unlink()
                        deleted = True or deleted
                    except Exception as file_e:
                        print(f"Warning: Could not delete previous checkpoint file {self._last_checkpoint_path}: {file_e}")
                if deleted:
                    print(f"Deleted previous checkpoint from long run: {previous_id}")
            except Exception as e:
                print(f"Warning: Could not clean up previous checkpoint {previous_id}: {e}")
        
        # Save model checkpoint
        self.timing_logger.start_operation("model_checkpoint_save", "io")
        if self.trainer is not None:
            self.trainer.save_checkpoint(str(checkpoint_path))
        self.timing_logger.end_operation("model_checkpoint_save", "io", f"filepath={checkpoint_path}")
        
        # NEW: Track this checkpoint for long run mode
        if self._long_run_mode:
            self._last_checkpoint_path = checkpoint_path
        
        # Create metadata for the checkpoint
        training_duration = time.time() - self._start_time if self._start_time else 0
        n_experts = getattr(self.current_config, 'n_experts', 6)
        d_model = getattr(self.current_config, 'd_model', 384)
        inferred_size = self.checkpoint_manager._infer_model_size_from_experts(n_experts, d_model)
        model_config = {
            'model_size': getattr(self.current_config, 'model_size', inferred_size),
            'learning_rate': 0.0003,  # Current learning rate
            'n_experts': n_experts,
            'n_layers': getattr(self.current_config, 'n_layers', 6),
            'd_model': d_model,
            'n_heads': getattr(self.current_config, 'n_heads', 8),
        }
        
        performance_metrics = {
            'best_score': self.trainer.best_score if self.trainer is not None else 0,
            'avg_score': avg_game_length * 10,  # Rough estimate
            # Use latest computed loss from broadcast metrics; fallback to 0.0
            'final_loss': metrics.get('loss', 0.0) if metrics.get('loss') is not None else 0.0,
            'training_speed': training_speed,
            # Enhanced metrics
            'score_trend': metrics.get('score_trend', 0.0),
            'loss_trend': metrics.get('loss_trend', 0.0),
            'max_tile_frequency': metrics.get('max_tile_frequency', {}),
            'training_efficiency': metrics.get('training_efficiency', {
                'score_consistency': 0.0,
                'loss_stability': 0.0,
                'improvement_rate': 0.0,
                'plateau_detection': 0.0
            }),
        }
        
        # NEW: Add run information to metadata
        tags = []
        if self._long_run_mode and self._current_run_id:
            tags.append(f"run_{self._current_run_id}")
            tags.append("long_run")
        
        self.timing_logger.start_operation("metadata_creation", "io")
        self.checkpoint_manager.create_checkpoint_metadata(
            checkpoint_id=checkpoint_id,
            episode=self.current_episode,
            training_duration=training_duration,
            model_config=model_config,
            performance_metrics=performance_metrics,
            tags=tags
        )
        self.timing_logger.end_operation("metadata_creation", "io")

        # Notify all connected clients that a new checkpoint is available
        self.timing_logger.start_operation("checkpoint_notification", "websocket")
        await self.ws_manager.broadcast({
            'type': 'checkpoint_created',
            'checkpoint_id': checkpoint_id,
            'episode': self.current_episode,
            'absolute_path': str(checkpoint_path),
            'created_at': time.time(),
            'long_run_mode': self._long_run_mode,
            'run_id': self._current_run_id
        })
        self.timing_logger.end_operation("checkpoint_notification", "websocket")
        
        self.timing_logger.end_operation("checkpoint_save", "io", f"checkpoint_id={checkpoint_id}")

    async def create_manual_checkpoint(self) -> str:
        """Manually create a checkpoint from current training state"""
        if not self.is_training or not self.trainer:
            raise ValueError("Cannot create checkpoint: training not active")
        
        # Calculate current metrics
        training_speed = self._calculate_training_speed_with_checkpoint_offset()
        avg_game_length = sum(self._game_lengths) / len(self._game_lengths) if self._game_lengths else 0
        
        # In long run mode, remove previous checkpoint from this run before creating a new manual one
        if self._long_run_mode and self._current_run_id and self._last_checkpoint_path:
            previous_id = self._last_checkpoint_path.stem
            try:
                deleted = False
                try:
                    deleted = self.checkpoint_manager.delete_checkpoint(previous_id)
                except Exception as meta_e:
                    print(f"Warning: Could not delete metadata for previous checkpoint {previous_id}: {meta_e}")
                if self._last_checkpoint_path.exists():
                    try:
                        self._last_checkpoint_path.unlink()
                        deleted = True or deleted
                    except Exception as file_e:
                        print(f"Warning: Could not delete previous checkpoint file {self._last_checkpoint_path}: {file_e}")
                if deleted:
                    print(f"Deleted previous manual checkpoint from long run: {previous_id}")
            except Exception as e:
                print(f"Warning: Could not clean up previous manual checkpoint {previous_id}: {e}")
        
        # Create a manual checkpoint with timestamp
        checkpoint_id = f"checkpoint_manual_{int(time.time())}"
        checkpoint_path = Path(self.checkpoint_dir) / f"{checkpoint_id}.pt"
        checkpoint_path = checkpoint_path.resolve()
        
        # Save model checkpoint
        self.trainer.save_checkpoint(str(checkpoint_path))
        
        # Create metadata
        training_duration = time.time() - self._start_time if self._start_time else 0
        n_experts = getattr(self.current_config, 'n_experts', 6)
        d_model = getattr(self.current_config, 'd_model', 384)
        inferred_size = self.checkpoint_manager._infer_model_size_from_experts(n_experts, d_model)
        model_config = {
            'model_size': getattr(self.current_config, 'model_size', inferred_size),
            'learning_rate': 0.0003,
            'n_experts': n_experts,
            'n_layers': getattr(self.current_config, 'n_layers', 6),
            'd_model': d_model,
            'n_heads': getattr(self.current_config, 'n_heads', 8),
        }
        
        performance_metrics = {
            'best_score': self.trainer.best_score if self.trainer is not None else 0,
            'avg_score': avg_game_length * 10,
            'final_loss': 0.0,  # Would need current loss calculation
            'training_speed': training_speed,
        }
        
        # Add manual checkpoint tags
        tags = ['manual']
        if self._long_run_mode and self._current_run_id:
            tags.append(f"run_{self._current_run_id}")
            tags.append("long_run")
        
        self.checkpoint_manager.create_checkpoint_metadata(
            checkpoint_id=checkpoint_id,
            episode=self.current_episode,
            training_duration=training_duration,
            model_config=model_config,
            performance_metrics=performance_metrics,
            nickname=f"Manual Checkpoint (Episode {self.current_episode})",
            tags=tags
        )
        
        # Track this checkpoint for long run mode so subsequent saves can clean up properly
        if self._long_run_mode:
            self._last_checkpoint_path = checkpoint_path
        
        # Notify clients
        await self.ws_manager.broadcast({
            'type': 'checkpoint_created',
            'checkpoint_id': checkpoint_id,
            'episode': self.current_episode,
            'absolute_path': str(checkpoint_path),
            'created_at': time.time(),
            'manual': True,
            'long_run_mode': self._long_run_mode,
            'run_id': self._current_run_id
        })
        
        print(f"Manual checkpoint created: {checkpoint_id}")
        return checkpoint_id

    # ------------------------------------------------------------------ Helpers
    def _build_metrics(self, episode_result: Dict[str, Any], env: Gym2048Env) -> Dict[str, Any]:
        """Map raw data → message consumed by frontend."""
        self.timing_logger.start_operation("build_metrics", "processing")
        
        board_state = [list(row) for row in env.game.board]
        
        # Get trainer metrics (use first trainer as representative)
        self.timing_logger.start_operation("get_trainer_metrics", "processing")
        trainer_metrics = self.env_trainers[0].get_metrics()
        self.timing_logger.end_operation("get_trainer_metrics", "processing")
        
        # Get action probabilities from last model forward pass
        actions_probs = self.env_trainers[0].get_latest_action_probs()
        
        # Calculate training speed (episodes per minute)
        training_speed = self._calculate_training_speed_with_checkpoint_offset()
        
        # Calculate game length metrics
        avg_game_length = 0
        min_game_length = 0
        max_game_length = 0
        if self._game_lengths:
            avg_game_length = sum(self._game_lengths) / len(self._game_lengths)
            min_game_length = min(self._game_lengths)
            max_game_length = max(self._game_lengths)
        
        # Calculate wall clock time elapsed
        wall_clock_elapsed = 0
        if self._start_time:
            wall_clock_elapsed = time.time() - self._start_time
        
        # Estimate time to next checkpoint
        episodes_to_checkpoint = self._checkpoint_interval - (self.current_episode % self._checkpoint_interval)
        estimated_time_to_checkpoint = 0
        if training_speed > 0:
            estimated_time_to_checkpoint = (episodes_to_checkpoint / training_speed) * 60  # seconds
        
        # Calculate enhanced metrics
        score_trend = self._calculate_score_trend()
        loss_trend = self._calculate_loss_trend()
        max_tile_frequency = self._calculate_max_tile_frequency()
        training_efficiency = self._calculate_training_efficiency()
        
        # Calculate load balancing metrics
        avg_lb_reward = 0.0
        if self._load_balancing_metrics:
            avg_lb_reward = sum(self._load_balancing_metrics) / len(self._load_balancing_metrics)
        
        # NEW: Enhanced load balancing metrics
        expert_starvation_rate = 0.0
        if self._expert_starvation_count > 0 and self.current_episode > 0:
            expert_starvation_rate = self._expert_starvation_count / self.current_episode
        
        avg_sparsity_score = 0.0
        if self._sparsity_scores:
            avg_sparsity_score = sum(self._sparsity_scores) / len(self._sparsity_scores)
        
        avg_balance_quality = 0.0
        if self._load_balance_quality:
            avg_balance_quality = sum(self._load_balance_quality) / len(self._load_balance_quality)
        
        # ENHANCED: Expert starvation metrics by model size
        starvation_by_model_size = {}
        for model_size, count in self._expert_starvation_by_model_size.items():
            if self.current_episode > 0:
                starvation_by_model_size[model_size] = count / self.current_episode
            else:
                starvation_by_model_size[model_size] = 0.0
        
        # NEW: Average starvation severity
        avg_starvation_severity = 0.0
        if self._starvation_severity_tracking:
            avg_starvation_severity = sum(self._starvation_severity_tracking) / len(self._starvation_severity_tracking)
        
        # NEW: Expert recovery tracking
        expert_recovery_rates = {}
        for expert_idx, usage_history in self._expert_recovery_tracking.items():
            if len(usage_history) >= 5:
                # Calculate if usage is trending upward
                recent_avg = sum(usage_history[-5:]) / 5
                older_avg = sum(usage_history[:-5]) / max(1, len(usage_history) - 5)
                recovery_rate = (recent_avg - older_avg) / max(older_avg, 1e-8)
                expert_recovery_rates[expert_idx] = recovery_rate
        
        # Expert usage trend analysis
        expert_usage_trend = 0.0
        if len(self._expert_usage_history) >= 10:
            recent_usage = self._expert_usage_history[-10:]
            # Safety check: ensure all usage entries have the same length
            if recent_usage and all(len(usage) == len(recent_usage[0]) for usage in recent_usage):
                try:
                    avg_recent = [sum(usage[i] for usage in recent_usage) / len(recent_usage) 
                                 for i in range(len(recent_usage[0]))]
                    ideal_usage = 1.0 / len(avg_recent)
                    expert_usage_trend = 1.0 - np.mean(np.abs(np.array(avg_recent) - ideal_usage)) / ideal_usage
                except (IndexError, ZeroDivisionError) as e:
                    print(f"Warning: Error calculating expert usage trend: {e}")
                    expert_usage_trend = 0.0
            else:
                print(f"Warning: Inconsistent expert usage data lengths, skipping trend calculation")
                expert_usage_trend = 0.0
        
        metrics = {
            "type": "training_update",
            "timestamp": time.time(),
            "episode": episode_result['episode'],
            "score": episode_result['score'],
            "reward": float(episode_result['reward']),
            "loss": (episode_result['losses']['policy_loss'] + episode_result['losses']['value_loss']) if (episode_result['losses']['policy_loss'] is not None and episode_result['losses']['value_loss'] is not None) else None,
            "policy_loss": episode_result['losses']['policy_loss'],
            "value_loss": episode_result['losses']['value_loss'],
            "entropy": episode_result['losses']['entropy'],
            "learning_rate": trainer_metrics['learning_rate'],
            "actions": actions_probs,
            "board_state": board_state,
            "attention_weights": trainer_metrics['attention_weights'],
            "expert_usage": trainer_metrics['expert_usage'],
            "gpu_memory": trainer_metrics['gpu_memory'],
            "model_params": trainer_metrics['model_params'],
            "loss_history": trainer_metrics['loss_history'],
            "score_history": trainer_metrics['score_history'],
            "training_speed": training_speed,
            "avg_game_length": avg_game_length,
            "min_game_length": min_game_length,
            "max_game_length": max_game_length,
            "wall_clock_elapsed": wall_clock_elapsed,
            "estimated_time_to_checkpoint": estimated_time_to_checkpoint,
            # Enhanced metrics
            "score_trend": score_trend,
            "loss_trend": loss_trend,
            "max_tile_frequency": max_tile_frequency,
            "training_efficiency": training_efficiency,
            "load_balancing_reward": avg_lb_reward,
            "expert_starvation_rate": expert_starvation_rate,
            "avg_sparsity_score": avg_sparsity_score,
            "avg_balance_quality": avg_balance_quality,
            "expert_usage_trend": expert_usage_trend,
            "starvation_by_model_size": starvation_by_model_size,
            "avg_starvation_severity": avg_starvation_severity,
            "expert_recovery_rates": expert_recovery_rates,
        }
        
        self.timing_logger.end_operation("build_metrics", "processing")
        return metrics
    
    def _estimate_max_tile_from_score(self, score: int) -> int:
        """Estimate the maximum tile achieved based on score"""
        # Rough estimation: higher scores typically mean higher max tiles
        if score >= 100000:
            return 8192
        elif score >= 50000:
            return 4096
        elif score >= 20000:
            return 2048
        elif score >= 10000:
            return 1024
        elif score >= 5000:
            return 512
        elif score >= 2000:
            return 256
        elif score >= 1000:
            return 128
        else:
            return 64
    
    def _calculate_score_trend(self) -> float:
        """Calculate score trend over recent episodes (positive = improving)"""
        if len(self._recent_scores) < 10:
            return 0.0
        
        # Calculate trend using linear regression
        n = len(self._recent_scores)
        x = list(range(n))
        y = self._recent_scores
        
        # Simple linear regression
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    
    def _calculate_loss_trend(self) -> float:
        """Calculate loss trend over recent episodes (negative = improving)"""
        if len(self._recent_losses) < 10:
            return 0.0
        
        # Calculate trend using linear regression
        n = len(self._recent_losses)
        x = list(range(n))
        y = self._recent_losses
        
        # Simple linear regression
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    
    def _calculate_max_tile_frequency(self) -> Dict[int, float]:
        """Calculate frequency of different max tiles achieved"""
        if not self._max_tiles_achieved:
            return {}
        
        tile_counts = {}
        total = len(self._max_tiles_achieved)
        
        for tile in self._max_tiles_achieved:
            tile_counts[tile] = tile_counts.get(tile, 0) + 1
        
        return {tile: count / total for tile, count in tile_counts.items()}
    
    def _calculate_training_efficiency(self) -> Dict[str, float]:
        """Calculate various training efficiency metrics"""
        if not self._recent_scores or not self._recent_losses:
            return {
                'score_consistency': 0.0,
                'loss_stability': 0.0,
                'improvement_rate': 0.0,
                'plateau_detection': 0.0,
                'load_balancing_efficiency': 0.0
            }
        
        # Score consistency (lower variance = more consistent)
        score_mean = sum(self._recent_scores) / len(self._recent_scores)
        score_variance = sum((s - score_mean) ** 2 for s in self._recent_scores) / len(self._recent_scores)
        score_consistency = max(0, 1 - (score_variance / (score_mean ** 2 + 1)))
        
        # Loss stability (lower variance = more stable)
        loss_mean = sum(self._recent_losses) / len(self._recent_losses)
        loss_variance = sum((l - loss_mean) ** 2 for l in self._recent_losses) / len(self._recent_losses)
        loss_stability = max(0, 1 - (loss_variance / (loss_mean ** 2 + 1)))
        
        # Improvement rate (positive trend = improving)
        score_trend = self._calculate_score_trend()
        improvement_rate = max(0, min(1, (score_trend + 1000) / 2000))  # Normalize to 0-1
        
        # Plateau detection (low variance + low trend = plateau)
        plateau_score = 1 - (abs(score_trend) / 1000)  # Lower trend = higher plateau score
        plateau_detection = (plateau_score + score_consistency) / 2
        
        # Load balancing efficiency
        load_balancing_efficiency = 0.0
        if self._load_balancing_metrics:
            lb_mean = sum(self._load_balancing_metrics) / len(self._load_balancing_metrics)
            load_balancing_efficiency = max(0, min(1, lb_mean * 10))  # Normalize to 0-1
        
        return {
            'score_consistency': score_consistency,
            'loss_stability': loss_stability,
            'improvement_rate': improvement_rate,
            'plateau_detection': plateau_detection,
            'load_balancing_efficiency': load_balancing_efficiency
        }
    
    def _estimate_next_episode_time(self) -> float:
        """Estimate time until next episode completion based on recent performance"""
        if not self._episode_start_times or len(self._episode_start_times) < 2:
            return 0.0
        
        # Calculate average time per episode from recent episodes
        recent_times = self._episode_start_times[-10:]  # Last 10 episodes
        if len(recent_times) < 2:
            return 0.0
        
        # Calculate average time between episodes
        total_time = recent_times[-1] - recent_times[0]
        avg_time_per_episode = total_time / (len(recent_times) - 1)
        
        return avg_time_per_episode

    def _calculate_training_speed_with_checkpoint_offset(self) -> float:
        """Calculate training speed accounting for loaded checkpoint episodes"""
        if not self._start_time or self.current_episode <= 0:
            return 0.0
        
        # Get the episode count from the trainer (includes loaded checkpoint episodes)
        trainer_episode_count = self.env_trainers[0].episode_count if self.env_trainers else 0
        
        # Calculate elapsed time since training started
        elapsed_time = time.time() - self._start_time
        
        # Use trainer episode count for accurate speed calculation
        # This includes episodes from loaded checkpoints
        total_episodes_completed = trainer_episode_count
        
        if elapsed_time > 0:
            training_speed = (total_episodes_completed / elapsed_time) * 60  # episodes per minute
            return training_speed
        
        return 0.0

    def write_timing_summary(self, filename: str = "logs/training_manager_timing_summary.json"):
        """Write timing summary to JSON file"""
        self.timing_logger.write_summary(filename) 