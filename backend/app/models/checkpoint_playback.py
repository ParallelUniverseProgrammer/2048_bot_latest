"""
Checkpoint playback system for watching saved models play 2048
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import asyncio
import time
import traceback
from datetime import datetime

from app.models.game_transformer import GameTransformer
from app.environment.gym_2048_env import Gym2048Env
from app.models.checkpoint_metadata import CheckpointManager
from app.utils.action_selection import select_action_with_fallback_for_playback

class CheckpointPlayback:
    """System for playing back saved checkpoints with performance optimizations"""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self.current_model = None
        self.current_config = None
        self.current_checkpoint_id = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.env = Gym2048Env()
        
        # Playback state
        self.is_playing = False
        self.is_paused = False
        self.game_history = []
        self.current_step = 0
        
        # Pause/resume state
        self.current_game_result = None
        self.current_step_index = 0
        self.current_game_count = 0
        
        # Performance optimizations
        self.adaptive_broadcast_interval = 0.5  # Start with 500ms between broadcasts
        self.min_broadcast_interval = 0.1  # Minimum 100ms
        self.max_broadcast_interval = 2.0  # Maximum 2s
        self.last_broadcast_time = 0
        self.broadcast_performance_history = []
        self.lightweight_mode = False  # Enable for high-speed playback
        
        # Message throttling
        self.message_throttle = {
            'last_step_broadcast': 0.0,
            'step_skip_count': 0,
            'target_fps': 10,  # Target 10 updates per second max
            'adaptive_skip': 1  # Skip every N steps
        }
        
        # Health monitoring
        self.last_heartbeat = time.time()
        self.last_broadcast_success = time.time()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        self.heartbeat_interval = 10.0  # seconds
        self.broadcast_timeout = 3.0  # Reduced from 5.0 for faster recovery
        
        # Error tracking
        self.error_history = []
        self.max_error_history = 50
        
        # Performance monitoring
        self.performance_metrics = {
            'total_broadcasts': 0,
            'successful_broadcasts': 0,
            'failed_broadcasts': 0,
            'avg_broadcast_time': 0.0,
            'slow_broadcasts': 0,
            'steps_skipped': 0,
            'games_completed': 0
        }
    
    def _log_error(self, error: str, context: str = ""):
        """Log error with timestamp and context"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_entry = {
            'timestamp': timestamp,
            'error': error,
            'context': context,
            'checkpoint_id': self.current_checkpoint_id,
            'game_count': self.current_game_count,
            'step_index': self.current_step_index
        }
        
        self.error_history.append(error_entry)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
        
        print(f"[{timestamp}] PLAYBACK ERROR in {context}: {error}")
        if self.current_checkpoint_id:
            print(f"  Checkpoint: {self.current_checkpoint_id}, Game: {self.current_game_count}, Step: {self.current_step_index}")
    
    def _update_heartbeat(self):
        """Update heartbeat timestamp"""
        self.last_heartbeat = time.time()
    

    
    def _is_healthy(self) -> bool:
        """Check if playback system is healthy"""
        current_time = time.time()
        
        # Check if heartbeat is recent
        if current_time - self.last_heartbeat > self.heartbeat_interval * 2:
            return False
        
        # Check if broadcasts are succeeding
        if current_time - self.last_broadcast_success > self.broadcast_timeout * 3:
            return False
        
        # Check consecutive failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            return False
        
        return True
    
    def _should_broadcast_step(self, step_idx: int) -> bool:
        """Determine if we should broadcast this step based on throttling rules"""
        current_time = time.time()
        
        # Check time-based throttling
        time_since_last = current_time - self.message_throttle['last_step_broadcast']
        min_interval = 1.0 / self.message_throttle['target_fps']
        
        if time_since_last < min_interval:
            return False
        
        # Check step-based throttling (adaptive skipping)
        if self.message_throttle['step_skip_count'] < self.message_throttle['adaptive_skip']:
            self.message_throttle['step_skip_count'] += 1
            self.performance_metrics['steps_skipped'] += 1
            return False
        
        # Reset skip count and allow broadcast
        self.message_throttle['step_skip_count'] = 0
        return True
    
    def _adjust_adaptive_settings(self, broadcast_time: float, connection_count: int):
        """Adjust adaptive settings based on performance"""
        # Track broadcast performance
        self.broadcast_performance_history.append(broadcast_time)
        if len(self.broadcast_performance_history) > 20:
            self.broadcast_performance_history.pop(0)
        
        avg_broadcast_time = sum(self.broadcast_performance_history) / len(self.broadcast_performance_history)
        
        # Adjust broadcast interval based on performance
        if avg_broadcast_time > 0.5:  # Slow broadcasts
            self.adaptive_broadcast_interval = min(self.max_broadcast_interval, 
                                                 self.adaptive_broadcast_interval * 1.2)
        elif avg_broadcast_time < 0.1:  # Fast broadcasts
            self.adaptive_broadcast_interval = max(self.min_broadcast_interval,
                                                 self.adaptive_broadcast_interval * 0.9)
        
        # Adjust skip rate based on connection count and performance
        if connection_count > 3 and avg_broadcast_time > 0.3:
            self.message_throttle['adaptive_skip'] = min(5, self.message_throttle['adaptive_skip'] + 1)
        elif connection_count <= 2 and avg_broadcast_time < 0.2:
            self.message_throttle['adaptive_skip'] = max(1, self.message_throttle['adaptive_skip'] - 1)
        
        # Enable lightweight mode for many connections or slow performance
        self.lightweight_mode = (connection_count > 4 or 
                               avg_broadcast_time > 0.4)
    
    def _create_lightweight_message(self, step_data: Dict[str, Any], game_result: Dict[str, Any], 
                                  checkpoint_id: str, game_count: int) -> Dict[str, Any]:
        """Create lightweight message for high-performance scenarios"""
        return {
            'type': 'checkpoint_playback_light',
            'checkpoint_id': checkpoint_id,
            'game_number': game_count,
            'step': step_data['step'],
            'board_state': step_data['board_state'],
            'score': step_data['score'],
            'action': step_data['action'],
            'timestamp': step_data['timestamp'],
            'game_progress': {
                'current_step': step_data['step'],
                'total_steps': game_result['steps'],
                'final_score': game_result['final_score']
            }
        }
    
    def _create_full_message(self, step_data: Dict[str, Any], game_result: Dict[str, Any], 
                           checkpoint_id: str, game_count: int) -> Dict[str, Any]:
        """Create full message with all data"""
        return {
            'type': 'checkpoint_playback',
            'checkpoint_id': checkpoint_id,
            'game_number': game_count,
            'step_data': step_data,
            'game_summary': {
                'final_score': game_result['final_score'],
                'total_steps': game_result['steps'],
                'max_tile': game_result['max_tile']
            },
            'performance_info': {
                'adaptive_skip': self.message_throttle['adaptive_skip'],
                'lightweight_mode': self.lightweight_mode,
                'broadcast_interval': self.adaptive_broadcast_interval
            }
        }

    async def _safe_broadcast(self, websocket_manager, message: Dict[str, Any], context: str = "") -> bool:
        """Safely broadcast message with error handling, timeout, and performance tracking"""
        try:
            # Add health check data to message
            message['playback_health'] = {
                'last_heartbeat': self.last_heartbeat,
                'consecutive_failures': self.consecutive_failures,
                'is_healthy': self._is_healthy(),
                'performance_metrics': self.performance_metrics
            }
            
            broadcast_start = time.perf_counter()
            
            # Use shorter timeout for better responsiveness
            await asyncio.wait_for(
                websocket_manager.broadcast(message),
                timeout=self.broadcast_timeout
            )
            
            broadcast_time = time.perf_counter() - broadcast_start
            
            # Update performance metrics
            self.performance_metrics['total_broadcasts'] += 1
            self.performance_metrics['successful_broadcasts'] += 1
            
            if self.performance_metrics['avg_broadcast_time'] == 0:
                self.performance_metrics['avg_broadcast_time'] = broadcast_time
            else:
                self.performance_metrics['avg_broadcast_time'] = (
                    0.9 * self.performance_metrics['avg_broadcast_time'] + 
                    0.1 * broadcast_time
                )
            
            if broadcast_time > 0.5:
                self.performance_metrics['slow_broadcasts'] += 1
            
            # Adjust adaptive settings
            connection_count = websocket_manager.get_connection_count()
            self._adjust_adaptive_settings(broadcast_time, connection_count)
            
            self.last_broadcast_success = time.time()
            self.consecutive_failures = 0
            return True
            
        except asyncio.TimeoutError:
            self.consecutive_failures += 1
            self.performance_metrics['failed_broadcasts'] += 1
            self._log_error(f"Broadcast timeout after {self.broadcast_timeout}s", context)
            
            # If we have too many consecutive failures, try to recover
            if self.consecutive_failures >= self.max_consecutive_failures:
                self._log_error("Too many consecutive broadcast failures, attempting recovery", context)
                await self._attempt_broadcast_recovery(websocket_manager)
            
            return False
        except Exception as e:
            self.consecutive_failures += 1
            self.performance_metrics['failed_broadcasts'] += 1
            self._log_error(f"Broadcast failed: {str(e)}", context)
            
            # If we have too many consecutive failures, try to recover
            if self.consecutive_failures >= self.max_consecutive_failures:
                self._log_error("Too many consecutive broadcast failures, attempting recovery", context)
                await self._attempt_broadcast_recovery(websocket_manager)
            
            return False
    
    async def _attempt_broadcast_recovery(self, websocket_manager):
        """Attempt to recover from broadcast failures"""
        try:
            # Wait a bit to allow connections to recover
            await asyncio.sleep(1.0)  # Reduced from 2.0
            
            # Send a simple heartbeat to test connection
            test_message = {
                'type': 'playback_recovery',
                'checkpoint_id': self.current_checkpoint_id,
                'timestamp': time.time(),
                'message': 'Attempting to recover from broadcast failures'
            }
            
            # Try to broadcast with extended timeout
            await asyncio.wait_for(
                websocket_manager.broadcast(test_message),
                timeout=self.broadcast_timeout * 2
            )
            
            # If successful, reset failure count
            self.consecutive_failures = 0
            self.last_broadcast_success = time.time()
            self._log_error("Broadcast recovery successful", "_attempt_broadcast_recovery")
            
        except Exception as e:
            self._log_error(f"Broadcast recovery failed: {str(e)}", "_attempt_broadcast_recovery")
            # Don't reset consecutive failures - let the caller handle it

    def load_checkpoint(self, checkpoint_id: str) -> bool:
        """Load a checkpoint for playback"""
        try:
            print(f"Loading checkpoint {checkpoint_id} for playback...")
            
            metadata = self.checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
            if not metadata:
                self._log_error(f"Checkpoint metadata not found", f"load_checkpoint({checkpoint_id})")
                return False
            
            checkpoint_path = self.checkpoint_manager._get_checkpoint_path(checkpoint_id)
            if not checkpoint_path.exists():
                self._log_error(f"Checkpoint file not found: {checkpoint_path}", f"load_checkpoint({checkpoint_id})")
                return False
            
            # Load checkpoint data
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Get model config
            config = checkpoint_data.get('config')
            if not config:
                self._log_error("No config found in checkpoint", f"load_checkpoint({checkpoint_id})")
                return False
            
            # Create model
            self.current_model = GameTransformer(config).to(self.device)
            self.current_model.load_state_dict(checkpoint_data['model_state_dict'])
            self.current_model.eval()
            
            self.current_config = config
            self.current_checkpoint_id = checkpoint_id
            
            # Reset health monitoring and performance metrics
            self._update_heartbeat()
            self.last_broadcast_success = time.time()
            self.consecutive_failures = 0
            self.performance_metrics = {
                'total_broadcasts': 0,
                'successful_broadcasts': 0,
                'failed_broadcasts': 0,
                'avg_broadcast_time': 0.0,
                'slow_broadcasts': 0,
                'steps_skipped': 0,
                'games_completed': 0
            }
            
            print(f"Successfully loaded checkpoint {checkpoint_id}")
            return True
            
        except Exception as e:
            self._log_error(f"Exception loading checkpoint: {str(e)}", f"load_checkpoint({checkpoint_id})")
            print(f"Full traceback: {traceback.format_exc()}")
            return False

    def play_single_game(self) -> Dict[str, Any]:
        """Play a single game and return the full game history"""
        if self.current_model is None:
            return {"error": "No model loaded"}
        
        try:
            game_history = []
            self.env.reset()
            
            total_reward = 0
            step_count = 0
            max_steps = 10000
            
            print(f"Starting game with max_steps={max_steps}")
            
            while not self.env.is_done() and step_count < max_steps:
                try:
                    # Get current state
                    state = self.env.get_state()
                    legal_actions = self.env.get_legal_actions()
                    
                    if not legal_actions:
                        print(f"No legal actions available at step {step_count}")
                        break
                    
                    # Select action
                    action, action_probs, attention_weights = self.select_action(state, legal_actions, self.env.game)
                    
                    # Store step data (optimized for memory)
                    step_data = {
                        'step': step_count,
                        'board_state': [list(row) for row in self.env.game.board],
                        'score': self.env.game.score,
                        'action': action,
                        'action_probs': action_probs.tolist() if action_probs is not None else None,
                        'legal_actions': legal_actions,
                        'attention_weights': attention_weights.tolist() if attention_weights is not None else None,
                        'timestamp': time.time()
                    }
                    
                    # Take action
                    obs, reward, done, truncated, info = self.env.step(action)
                    # For 2048, treat truncated as done
                    done = done or truncated
                    
                    total_reward += reward
                    step_data['reward'] = reward
                    step_data['done'] = done
                    
                    game_history.append(step_data)
                    step_count += 1
                    
                    if done:
                        print(f"Game finished at step {step_count} with score {self.env.game.score}")
                        break
                        
                except Exception as e:
                    self._log_error(f"Error in game step {step_count}: {str(e)}", "play_single_game")
                    print(f"Step error traceback: {traceback.format_exc()}")
                    break
            
            # Calculate final metrics
            final_score = self.env.game.score
            max_tile = np.max(self.env.game.board)
            
            result = {
                'game_history': game_history,
                'final_score': final_score,
                'max_tile': int(max_tile),
                'steps': step_count,
                'total_reward': total_reward,
                'completed': step_count < max_steps and self.env.is_done()
            }
            
            print(f"Game completed: score={final_score}, max_tile={max_tile}, steps={step_count}")
            return result
            
        except Exception as e:
            self._log_error(f"Fatal error in play_single_game: {str(e)}", "play_single_game")
            print(f"Full traceback: {traceback.format_exc()}")
            return {"error": f"Game failed: {str(e)}"}
    
    async def start_live_playback(self, websocket_manager):
        """Start live playback streaming to websocket with performance optimizations"""
        if self.current_model is None:
            self._log_error("No model loaded for playback", "start_live_playback")
            return
        
        self.is_playing = True
        self.is_paused = False
        
        # Reset message throttling settings
        self.message_throttle['last_step_broadcast'] = 0
        self.message_throttle['step_skip_count'] = 0
        self.message_throttle['adaptive_skip'] = 1
        
        # Set default performance settings
        self.message_throttle['target_fps'] = 10
        self.lightweight_mode = False
        
        print(f"Starting live playback for checkpoint {self.current_checkpoint_id}")
        
        # Send initial status message
        success = await self._safe_broadcast(websocket_manager, {
            'type': 'playback_status',
            'message': f'Starting playback for checkpoint {self.current_checkpoint_id}',
            'checkpoint_id': self.current_checkpoint_id,
            'status': 'starting',
            'performance_mode': {
                'lightweight_mode': self.lightweight_mode,
                'target_fps': self.message_throttle['target_fps'],
                'adaptive_skip': self.message_throttle['adaptive_skip']
            }
        }, "initial_status")
        
        if not success:
            self._log_error("Failed to send initial status", "start_live_playback")
            self.is_playing = False
            return
        
        # Initialize game count if not resuming
        if self.current_game_result is None:
            self.current_game_count = 0
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(websocket_manager))
        
        # Track step timing to detect stuck states
        step_start_time = time.time()
        last_step_time = time.time()
        
        try:
            while self.is_playing:
                # Update heartbeat
                self._update_heartbeat()
                
                # Check health
                if not self._is_healthy():
                    self._log_error("Playback system unhealthy, stopping", "start_live_playback")
                    break
                
                if self.is_paused:
                    await asyncio.sleep(0.1)
                    continue
                
                # Check if we have any connected clients (but don't pause if using HTTP polling)
                connection_count = websocket_manager.get_connection_count()
                if connection_count == 0:
                    print(f"No WebSocket clients connected ({connection_count}), but continuing for HTTP polling...")
                    # Don't pause - allow HTTP polling to work
                    await asyncio.sleep(0.1)
                    continue
                
                # If we have a current game in progress, resume it
                if self.current_game_result is not None:
                    print(f"Resuming game #{self.current_game_count} from step {self.current_step_index}")
                    game_result = self.current_game_result
                    game_count = self.current_game_count
                    start_step = self.current_step_index
                    
                    # Reset timing for resumed game
                    step_start_time = time.time()
                    last_step_time = time.time()
                else:
                    # Start new game
                    self.current_game_count += 1
                    game_count = self.current_game_count
                    print(f"Playing game #{game_count} for checkpoint {self.current_checkpoint_id}")
                    
                    # Send new game started message
                    await self._safe_broadcast(websocket_manager, {
                        'type': 'new_game_started',
                        'checkpoint_id': self.current_checkpoint_id,
                        'game_number': game_count,
                        'message': f'Starting game #{game_count}'
                    }, "new_game_started")
                    
                    # Play one game
                    game_result = self.play_single_game()
                    
                    if 'error' in game_result:
                        self._log_error(f"Game error: {game_result['error']}", "start_live_playback")
                        await self._safe_broadcast(websocket_manager, {
                            'type': 'playback_error',
                            'error': game_result['error'],
                            'checkpoint_id': self.current_checkpoint_id
                        }, "game_error")
                        break
                    
                    print(f"Game #{game_count} completed: score={game_result.get('final_score', 0)}, steps={game_result.get('steps', 0)}")
                    
                    # Store current game state and start from beginning
                    self.current_game_result = game_result
                    start_step = 0
                    
                    # Reset timing for new game
                    step_start_time = time.time()
                    last_step_time = time.time()
                
                # Stream game history step by step with optimizations
                game_finished = False
                for step_idx in range(start_step, len(game_result['game_history'])):
                    if not self.is_playing or self.is_paused:
                        break
                    
                    step_data = game_result['game_history'][step_idx]
                    current_time = time.time()
                    
                    # Check if we should broadcast this step
                    if not self._should_broadcast_step(step_idx):
                        # Still need to wait for consistent timing
                        wait_time = 0.5  # Fixed 500ms wait between steps
                        await asyncio.sleep(wait_time)
                        continue
                    
                    # Update timing checks
                    step_duration = current_time - step_start_time
                    total_step_time = current_time - last_step_time
                    
                    # If a single step takes too long (> 30 seconds) or we've been on the same step for too long
                    if step_duration > 30.0 or (step_idx == self.current_step_index and total_step_time > 60.0):
                        self._log_error(f"Step {step_idx} taking too long ({step_duration:.1f}s), attempting recovery", "start_live_playback")
                        
                        # Send recovery status
                        await self._safe_broadcast(websocket_manager, {
                            'type': 'playback_status',
                            'message': f'Step {step_idx} taking too long, attempting recovery...',
                            'checkpoint_id': self.current_checkpoint_id,
                            'status': 'recovering'
                        }, "step_timeout_recovery")
                        
                        # Try to recover by resetting failure count and continuing
                        self.consecutive_failures = 0
                        last_step_time = current_time
                        step_start_time = current_time
                        
                        # Small delay to prevent rapid retries
                        await asyncio.sleep(0.5)  # Reduced from 1.0
                    
                    # Update current step index
                    self.current_step_index = step_idx
                    last_step_time = current_time
                    
                    # Create appropriate message based on mode
                    checkpoint_id = self.current_checkpoint_id or "unknown"
                    if self.lightweight_mode:
                        message = self._create_lightweight_message(step_data, game_result, 
                                                                 checkpoint_id, game_count)
                    else:
                        message = self._create_full_message(step_data, game_result, 
                                                          checkpoint_id, game_count)
                    
                    # Broadcast step data
                    success = await self._safe_broadcast(websocket_manager, message, f"step_{step_idx}")
                    
                    if success:
                        self.message_throttle['last_step_broadcast'] = current_time
                        if step_idx % 10 == 0:  # Log every 10th step
                            print(f"Broadcasted step {step_idx} of game {game_count}")
                    else:
                        self._log_error(f"Failed to broadcast step {step_idx}", "start_live_playback")
                        # Don't break immediately, the recovery mechanism will handle it
                        if self.consecutive_failures >= self.max_consecutive_failures:
                            print(f"Too many consecutive failures, pausing for recovery...")
                            await asyncio.sleep(2.0)
                            self.consecutive_failures = 0
                    
                    # Wait based on fixed timing
                    wait_time = 0.5  # Fixed 500ms between steps
                    await asyncio.sleep(wait_time)
                    
                    # Check if this was the last step
                    if step_idx == len(game_result['game_history']) - 1:
                        game_finished = True
                
                # If we finished the game completely, send completion message and reset
                if game_finished:
                    self.performance_metrics['games_completed'] += 1
                    
                    await self._safe_broadcast(websocket_manager, {
                        'type': 'game_completed',
                        'checkpoint_id': self.current_checkpoint_id,
                        'game_number': game_count,
                        'final_score': game_result['final_score'],
                        'total_steps': game_result['steps'],
                        'max_tile': game_result['max_tile'],
                        'performance_summary': self.performance_metrics
                    }, "game_completed")
                    
                    # Reset game state for next game
                    self.current_game_result = None
                    self.current_step_index = 0
                    
                    # Brief pause between games
                    if self.is_playing:
                        pause_time = 2.0  # Fixed 2 second pause between games
                        print(f"Waiting {pause_time:.1f}s before next game...")
                        await asyncio.sleep(pause_time)
                
        except asyncio.CancelledError:
            print("Playback cancelled")
        except Exception as e:
            self._log_error(f"Fatal error in live playback: {str(e)}", "start_live_playback")
            print(f"Full traceback: {traceback.format_exc()}")
            await self._safe_broadcast(websocket_manager, {
                'type': 'playback_error',
                'error': str(e),
                'checkpoint_id': self.current_checkpoint_id
            }, "fatal_error")
        finally:
            # Cancel heartbeat task
            if not heartbeat_task.done():
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            self.is_playing = False
            self.is_paused = False
            self.current_game_result = None
            self.current_step_index = 0
            print(f"Playback stopped for checkpoint {self.current_checkpoint_id} after {self.current_game_count} games")
            
            # Send final status message
            await self._safe_broadcast(websocket_manager, {
                'type': 'playback_status',
                'message': f'Playback stopped for checkpoint {self.current_checkpoint_id}',
                'checkpoint_id': self.current_checkpoint_id,
                'status': 'stopped',
                'games_played': self.current_game_count,
                'performance_summary': self.performance_metrics,
                'error_history': self.error_history[-3:]  # Send last 3 errors
            }, "final_status")
    
    async def _heartbeat_loop(self, websocket_manager):
        """Send periodic heartbeat messages to detect connection issues"""
        try:
            while self.is_playing:
                await asyncio.sleep(self.heartbeat_interval)
                
                if not self.is_playing:
                    break
                
                # Send heartbeat with performance info
                await self._safe_broadcast(websocket_manager, {
                    'type': 'playback_heartbeat',
                    'checkpoint_id': self.current_checkpoint_id,
                    'timestamp': time.time(),
                    'game_count': self.current_game_count,
                    'step_index': self.current_step_index,
                    'is_healthy': self._is_healthy(),
                    'consecutive_failures': self.consecutive_failures,
                    'performance_metrics': self.performance_metrics,
                    'adaptive_settings': {
                        'broadcast_interval': self.adaptive_broadcast_interval,
                        'lightweight_mode': self.lightweight_mode,
                        'adaptive_skip': self.message_throttle['adaptive_skip']
                    }
                }, "heartbeat")
                
        except asyncio.CancelledError:
            pass
    
    def pause_playback(self):
        """Pause live playback"""
        self.is_paused = True
        print(f"Playback paused at game {self.current_game_count}, step {self.current_step_index}")
    
    def resume_playback(self):
        """Resume live playback"""
        self.is_paused = False
        self._update_heartbeat()
        print(f"Playback resumed at game {self.current_game_count}, step {self.current_step_index}")
    
    def stop_playback(self):
        """Stop live playback"""
        self.is_playing = False
        self.is_paused = False
        # Reset game state when stopping
        self.current_game_result = None
        self.current_step_index = 0
        self.current_game_count = 0
        print("Playback stopped by user")
    
    def get_playback_status(self) -> Dict[str, Any]:
        """Get current playback status with performance metrics"""
        return {
            'is_playing': self.is_playing,
            'is_paused': self.is_paused,
            'current_checkpoint': self.current_checkpoint_id,
            'model_loaded': self.current_model is not None,
            'current_game': self.current_game_count,
            'current_step': self.current_step_index,
            'is_healthy': self._is_healthy(),
            'performance_metrics': self.performance_metrics,
            'adaptive_settings': {
                'broadcast_interval': self.adaptive_broadcast_interval,
                'lightweight_mode': self.lightweight_mode,
                'adaptive_skip': self.message_throttle['adaptive_skip'],
                'target_fps': self.message_throttle['target_fps']
            }
        }
    
    def get_current_step_data(self) -> Optional[Dict[str, Any]]:
        """Get current step data for polling fallback"""
        if (not self.is_playing or 
            self.current_game_result is None or 
            self.current_step_index >= len(self.current_game_result['game_history'])):
            return None
        
        step_data = self.current_game_result['game_history'][self.current_step_index]
        
        # Return lightweight data for polling
        return {
            'step_data': {
                'step': step_data['step'],
                'board_state': step_data['board_state'],
                'score': step_data['score'],
                'action': step_data['action'],
                'timestamp': step_data['timestamp']
            },
            'game_summary': {
                'final_score': self.current_game_result['final_score'],
                'total_steps': self.current_game_result['steps'],
                'max_tile': self.current_game_result['max_tile']
            }
        }

    def select_action(self, state: np.ndarray, legal_actions: List[int], game) -> Tuple[int, np.ndarray, Optional[np.ndarray]]:
        """Select action using the loaded model with comprehensive error handling"""
        if self.current_model is None:
            self._log_error("No model loaded for action selection", "select_action")
            # Return random action as fallback
            import random
            if not legal_actions:
                return 0, np.array([0.25, 0.25, 0.25, 0.25]), None
            
            action = random.choice(legal_actions)
            probs = np.array([0.0, 0.0, 0.0, 0.0])
            uniform_prob = 1.0 / len(legal_actions)
            for legal_action in legal_actions:
                probs[legal_action] = uniform_prob
            return action, probs, None
        
        try:
            # Validate inputs
            if state is None:
                self._log_error("State is None in action selection", "select_action")
                return self._fallback_action_selection(legal_actions)
            
            if not legal_actions:
                self._log_error("No legal actions available", "select_action")
                return 0, np.array([0.25, 0.25, 0.25, 0.25]), None
            
            # Call the enhanced action selection function
            action, action_probs, attention_weights = select_action_with_fallback_for_playback(
                self.current_model, state, legal_actions, game, self.device
            )
            
            # Validate outputs
            if action is None or action not in legal_actions:
                self._log_error(f"Invalid action returned: {action}, legal actions: {legal_actions}", "select_action")
                return self._fallback_action_selection(legal_actions)
            
            if action_probs is None or len(action_probs) != 4:
                self._log_error(f"Invalid action probabilities returned: {action_probs}", "select_action")
                action_probs = [0.25, 0.25, 0.25, 0.25]
            
            return action, np.array(action_probs), attention_weights
            
        except Exception as e:
            self._log_error(f"Exception in action selection: {str(e)}", "select_action")
            self._log_error(f"Action selection traceback: {traceback.format_exc()}", "select_action")
            return self._fallback_action_selection(legal_actions)
    
    def _fallback_action_selection(self, legal_actions: List[int]) -> Tuple[int, np.ndarray, Optional[np.ndarray]]:
        """Fallback action selection when model fails"""
        import random
        
        if not legal_actions:
            self._log_error("No legal actions in fallback", "_fallback_action_selection")
            return 0, np.array([0.25, 0.25, 0.25, 0.25]), None
        
        # Select random action from legal actions
        action = random.choice(legal_actions)
        
        # Create uniform probability distribution for legal actions
        probs = np.array([0.0, 0.0, 0.0, 0.0])
        uniform_prob = 1.0 / len(legal_actions)
        for legal_action in legal_actions:
            probs[legal_action] = uniform_prob
        
        self._log_error(f"Using fallback action selection: {action}", "_fallback_action_selection")
        return action, probs, None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        if self.current_model is None:
            return {
                "model_loaded": False,
                "error": "No model currently loaded"
            }
        
        try:
            return {
                "model_loaded": True,
                "checkpoint_id": self.current_checkpoint_id,
                "config": {
                    "model_size": getattr(self.current_config, 'model_size', 'unknown'),
                    "n_layers": getattr(self.current_config, 'n_layers', 'unknown'),
                    "n_experts": getattr(self.current_config, 'n_experts', 'unknown'),
                    "d_model": getattr(self.current_config, 'd_model', 'unknown'),
                    "n_heads": getattr(self.current_config, 'n_heads', 'unknown'),
                    "estimated_params": getattr(self.current_config, 'estimated_params', 'unknown')
                } if self.current_config else None,
                "device": str(self.device),
                "model_type": type(self.current_model).__name__
            }
        except Exception as e:
            return {
                "model_loaded": True,
                "error": f"Error getting model info: {str(e)}",
                "checkpoint_id": self.current_checkpoint_id
            }
    
    def get_error_history(self) -> List[Dict[str, Any]]:
        """Get recent error history"""
        return self.error_history.copy()
    
    def clear_error_history(self):
        """Clear error history"""
        self.error_history.clear() 