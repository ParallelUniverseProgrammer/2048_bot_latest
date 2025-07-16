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

import numpy as np

from app.api.websocket_manager import WebSocketManager
from app.environment.gym_2048_env import Gym2048Env
from app.training.ppo_trainer import PPOTrainer
from app.models.checkpoint_metadata import CheckpointManager


class TrainingManager:
    """Manage training lifecycle and push streaming metrics."""

    def __init__(self, websocket_manager: WebSocketManager, n_envs: int = 4):
        self.ws_manager = websocket_manager
        # Create multiple parallel environments
        self.envs: List[Gym2048Env] = [Gym2048Env() for _ in range(n_envs)]
        self._task: Optional[asyncio.Task] = None
        
        # Initialize PPO trainer with default config (will be updated when training starts)
        self.trainer = PPOTrainer()
        self.current_config = None
        
        # run-time state
        self.is_training: bool = False
        self.is_paused: bool = False
        self.current_episode: int = 0
        self.total_episodes: int = 10_000
        
        # Checkpoint management
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        
        # Metrics tracking
        self._start_time: Optional[float] = None
        self._game_lengths: List[int] = []
        self._episode_start_times: List[float] = []
        self._checkpoint_interval = 100  # Save checkpoint every 100 episodes
        
        # Enhanced metrics tracking
        self._recent_scores: List[int] = []  # Last 100 scores for trend analysis
        self._recent_losses: List[float] = []  # Last 100 losses for trend analysis
        self._max_tiles_achieved: List[int] = []  # Track max tiles achieved
        self._efficiency_metrics: Dict[str, float] = {}  # Training efficiency metrics

    # ------------------------------------------------------------------ Control
    def update_config(self, config_dict: Dict[str, Any]):
        """Update training configuration"""
        from app.models.model_config import DynamicModelConfig
        
        # Get the model size from config
        model_size = config_dict.get('model_size', 'medium')
        
        # Map model size to VRAM target for configuration selection
        vram_targets = {
            'small': 2.0,
            'medium': 4.0,
            'large': 6.0
        }
        
        target_vram = vram_targets.get(model_size, 4.0)
        self.current_config = DynamicModelConfig.select_config(target_vram=target_vram)
        
        # Reinitialize trainer with new config
        self.trainer = PPOTrainer(
            config=self.current_config,
            learning_rate=config_dict.get('learning_rate', 0.0003)
        )
        
        print(f"Updated training config: {model_size} model with {self.current_config.estimated_params:.1f}M parameters")
    
    def start(self):
        if self._task and not self._task.done():
            # already running
            return
        self.is_training = True
        self.is_paused = False
        self._start_time = time.time()  # Track start time for speed calculation
        
        # Send training start message
        asyncio.create_task(self._send_training_start_message())
        
        # schedule on current event loop
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._training_loop())
    
    def reset_to_fresh_model(self):
        """Reset training state and create a fresh model with current config"""
        # Reset training state
        self.current_episode = 0
        self.is_paused = False
        self._start_time = None
        self._game_lengths = []
        self._episode_start_times = []
        
        # Create a fresh trainer with the current config (or default if none set)
        if self.current_config is None:
            # Use default config if none is set
            from app.models.model_config import DynamicModelConfig
            self.current_config = DynamicModelConfig.select_config()
        
        # Reinitialize trainer with fresh state
        self.trainer = PPOTrainer(
            config=self.current_config,
            learning_rate=0.0003  # Default learning rate
        )
        
        print(f"Reset to fresh model with config: {self.current_config}")

    async def _send_training_start_message(self):
        """Send training start message via WebSocket"""
        try:
            await self.ws_manager.broadcast_high_priority({
                'type': 'training_start',
                'message': 'Training session started',
                'model_config': self.current_config.__dict__ if self.current_config else {},
                'status': 'starting'
            })
        except Exception as e:
            print(f"Error sending training start message: {e}")

    def pause(self):
        self.is_paused = True

    def resume(self):
        if self.is_training:
            self.is_paused = False

    async def stop(self):
        self.is_training = False
        self.is_paused = False
        if self._task:
            if not self._task.done():
                self._task.cancel()
                print("Training task cancellation requested.")
            else:
                print("Training task already completed.")
        # Do not await self._task; return immediately
        # Optionally, schedule a background cleanup if needed

    # ------------------------------------------------------------------ Loop
    async def _training_loop(self):
        try:
            while self.is_training and self.current_episode < self.total_episodes:
                if self.is_paused:
                    await asyncio.sleep(0.5)
                    continue

                try:
                    # Send new episode start message for first few episodes
                    if self.current_episode % 50 == 0:  # Every 50th episode
                        try:
                            await self.ws_manager.broadcast_high_priority({
                                'type': 'new_episode_started',
                                'episode': self.current_episode,
                                'message': f'Starting training episode {self.current_episode}'
                            })
                        except Exception as e:
                            print(f"Error sending new episode message: {e}")
                    
                    # Record episode start time
                    episode_start_time = time.time()
                    
                    # Train one episode **per environment** using PPO in parallel threads.
                    if self.current_episode < 5:  # Debug logging for first few iterations
                        print(f"🔄 Training {len(self.envs)} environments in parallel...")
                    
                    gather_coro = asyncio.gather(*[
                        asyncio.to_thread(self.trainer.train_episode, env) for env in self.envs
                    ])
                    try:
                        episode_results: List[Dict[str, Any]] = await asyncio.wait_for(gather_coro, timeout=60.0)
                        if self.current_episode < 5:  # Debug logging
                            print(f"✅ Training completed, got {len(episode_results)} results")
                    except asyncio.TimeoutError:
                        print(f"[red]Training episode timeout after 60s (episode {self.current_episode}) – cancelling tasks")
                        gather_coro.cancel()
                        # Attempt graceful cancellation
                        try:
                            await gather_coro
                        except Exception:
                            pass
                        # Re-raise to trigger outer retry logic
                        raise

                    # Convenience: last result for logging / checkpoint metrics
                    last_res = episode_results[-1] if episode_results else {}
                    
                    # Record episode metrics for each environment
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
                    
                    # After each environment episode generate metrics & broadcast (normal priority for training updates)
                    for env_instance, result in zip(self.envs, episode_results):
                        metrics = self._build_metrics(result, env_instance)
                        await self.ws_manager.broadcast(metrics, priority="normal")

                    # Extract metrics we need later for checkpoint bookkeeping
                    training_speed = metrics['training_speed']
                    avg_game_length = metrics['avg_game_length']
                    
                    # Debug logging – show stats from the last environment in the batch
                    if self.current_episode % 10 == 0 and episode_results:
                        print(
                            f"Episode {self.current_episode}: Score={last_res['score']}, "
                            f"Params={metrics['model_params']:.1f}M, "
                            f"LR={metrics['learning_rate']:.6f}"
                        )

                    # Update current episode - increment by number of environments trained
                    if episode_results:
                        # Each iteration trains one episode per environment, so increment by number of environments
                        self.current_episode += len(self.envs)
                        print(f"📈 Training progress: {self.current_episode}/{self.total_episodes} episodes completed")
                        
                        # Send training status update to frontend
                        try:
                            await self.ws_manager.broadcast({
                                'type': 'training_status_update',
                                'is_training': True,
                                'is_paused': False,
                                'current_episode': self.current_episode,
                                'total_episodes': self.total_episodes
                            })
                        except Exception as e:
                            print(f"Error broadcasting training status: {e}")
                    else:
                        # If no results, increment by 1 to prevent infinite loop
                        self.current_episode += 1
                        print(f"⚠️ No episode results, incrementing episode count to {self.current_episode}")
                    
                    # Save checkpoint periodically
                    if self.current_episode % self._checkpoint_interval == 0:
                        checkpoint_id = f"checkpoint_episode_{self.current_episode}"
                        checkpoint_path = os.path.join(
                            self.checkpoint_dir, 
                            f"{checkpoint_id}.pt"
                        )
                        self.trainer.save_checkpoint(checkpoint_path)
                        
                        # Create metadata for the checkpoint
                        training_duration = time.time() - self._start_time if self._start_time else 0
                        n_experts = getattr(self.current_config, 'n_experts', 6)
                        inferred_size = self.checkpoint_manager._infer_model_size_from_experts(n_experts)
                        model_config = {
                            'model_size': getattr(self.current_config, 'model_size', inferred_size),
                            'learning_rate': 0.0003,  # Current learning rate
                            'n_experts': n_experts,
                            'n_layers': getattr(self.current_config, 'n_layers', 6),
                            'd_model': getattr(self.current_config, 'd_model', 384),
                            'n_heads': getattr(self.current_config, 'n_heads', 8),
                        }
                        
                        performance_metrics = {
                            'best_score': self.trainer.best_score,
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
                        
                        self.checkpoint_manager.create_checkpoint_metadata(
                            checkpoint_id=checkpoint_id,
                            episode=self.current_episode,
                            training_duration=training_duration,
                            model_config=model_config,
                            performance_metrics=performance_metrics
                        )

                        # Notify all connected clients that a new checkpoint is available
                        await self.ws_manager.broadcast({
                            'type': 'checkpoint_created',
                            'checkpoint_id': checkpoint_id,
                            'episode': self.current_episode,
                            'created_at': time.time()
                        })
                    
                    await asyncio.sleep(0.01)  # Reduced delay for quicker feedback during training
                    
                except Exception as e:
                    print(f"Error in training episode {self.current_episode}: {e}")
                    await asyncio.sleep(1)  # Wait before retrying
                    
        except Exception as e:
            print(f"Fatal error in training loop: {e}")
            import traceback
            traceback.print_exc()
            self.is_training = False
            
        # Training complete
        await self.ws_manager.broadcast_high_priority({
            "type": "training_complete",
            "message": "Training finished",
            "final_episode": self.current_episode,
        })
        self.is_training = False
        self.current_episode = 0

    # ------------------------------------------------------------------ Helpers
    def _build_metrics(self, episode_result: Dict[str, Any], env: Gym2048Env) -> Dict[str, Any]:
        """Map raw data → message consumed by frontend."""
        board_state = [list(row) for row in env.game.board]
        
        # Get trainer metrics
        trainer_metrics = self.trainer.get_metrics()
        
        # Get action probabilities from last model forward pass
        actions_probs = self.trainer.get_latest_action_probs()
        
        # Calculate training speed (episodes per minute)
        training_speed = 0.0
        if self._start_time and self.current_episode > 0:
            elapsed_time = time.time() - self._start_time
            training_speed = (self.current_episode / elapsed_time) * 60  # episodes per minute
        
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
        }
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
                'plateau_detection': 0.0
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
        
        return {
            'score_consistency': score_consistency,
            'loss_stability': loss_stability,
            'improvement_rate': improvement_rate,
            'plateau_detection': plateau_detection
        } 