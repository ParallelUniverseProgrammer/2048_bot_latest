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
    """System for playing back saved checkpoints"""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self.current_model = None
        self.current_config = None
        self.current_checkpoint_id = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = Gym2048Env()
        
        # Playback state
        self.is_playing = False
        self.is_paused = False
        self.game_history = []
        self.current_step = 0
        self.playback_speed = 1.0  # seconds between moves
        
        # Pause/resume state
        self.current_game_result = None
        self.current_step_index = 0
        self.current_game_count = 0
        
        # Health monitoring
        self.last_heartbeat = time.time()
        self.last_broadcast_success = time.time()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        self.heartbeat_interval = 10.0  # seconds
        self.broadcast_timeout = 5.0  # seconds
        
        # Error tracking
        self.error_history = []
        self.max_error_history = 50
    
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
    
    async def _safe_broadcast(self, websocket_manager, message: Dict[str, Any], context: str = "") -> bool:
        """Safely broadcast message with error handling and timeout"""
        try:
            # Add health check data to message
            message['playback_health'] = {
                'last_heartbeat': self.last_heartbeat,
                'consecutive_failures': self.consecutive_failures,
                'is_healthy': self._is_healthy()
            }
            
            # Use asyncio.wait_for to add timeout
            await asyncio.wait_for(
                websocket_manager.broadcast(message),
                timeout=self.broadcast_timeout
            )
            
            self.last_broadcast_success = time.time()
            self.consecutive_failures = 0
            return True
            
        except asyncio.TimeoutError:
            self.consecutive_failures += 1
            self._log_error(f"Broadcast timeout after {self.broadcast_timeout}s", context)
            
            # If we have too many consecutive failures, try to recover
            if self.consecutive_failures >= self.max_consecutive_failures:
                self._log_error("Too many consecutive broadcast failures, attempting recovery", context)
                await self._attempt_broadcast_recovery(websocket_manager)
            
            return False
        except Exception as e:
            self.consecutive_failures += 1
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
            await asyncio.sleep(2.0)
            
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
            
            # Reset health monitoring
            self._update_heartbeat()
            self.last_broadcast_success = time.time()
            self.consecutive_failures = 0
            
            print(f"Successfully loaded checkpoint {checkpoint_id}")
            return True
            
        except Exception as e:
            self._log_error(f"Exception loading checkpoint: {str(e)}", f"load_checkpoint({checkpoint_id})")
            print(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def select_action(self, state: np.ndarray, legal_actions: List[int], env_game) -> Tuple[int, List[float], Optional[np.ndarray]]:
        """Select action using loaded model with fallback for invalid moves"""
        if self.current_model is None:
            return 0, [0.25, 0.25, 0.25, 0.25], None
        
        try:
            # Use fallback mechanism to select action
            action, action_probs, attention_weights = select_action_with_fallback_for_playback(
                model=self.current_model,
                state=state,
                legal_actions=legal_actions,
                env_game=env_game,
                device=self.device,
                deterministic=True  # Use deterministic selection for playback
            )
            
            return action, action_probs, attention_weights
            
        except Exception as e:
            self._log_error(f"Error in action selection: {str(e)}", "select_action")
            # Return fallback action
            return legal_actions[0] if legal_actions else 0, [0.25, 0.25, 0.25, 0.25], None
    
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
                    
                    # Store step data
                    step_data = {
                        'step': step_count,
                        'board_state': [list(row) for row in self.env.game.board],
                        'score': self.env.game.score,
                        'action': action,
                        'action_probs': action_probs,
                        'legal_actions': legal_actions,
                        'attention_weights': attention_weights.tolist() if attention_weights is not None else None,
                        'timestamp': time.time()
                    }
                    
                    # Take action
                    obs, reward, done, truncated, info = self.env.step(action)
                    # For 2048, treat truncated as done
                    done = done or truncated
                    total_reward += reward
                    step_count += 1
                    
                    # Add reward to step data
                    step_data['reward'] = reward
                    step_data['done'] = done
                    
                    game_history.append(step_data)
                    
                    if done:
                        # Add final step showing the board state after the last move
                        final_step_data = {
                            'step': step_count,
                            'board_state': [list(row) for row in self.env.game.board],
                            'score': self.env.game.score,
                            'action': None,
                            'action_probs': [0.0, 0.0, 0.0, 0.0],
                            'legal_actions': [],
                            'attention_weights': None,
                            'timestamp': time.time(),
                            'reward': 0,
                            'done': True
                        }
                        game_history.append(final_step_data)
                        break
                        
                except Exception as e:
                    self._log_error(f"Error in game step {step_count}: {str(e)}", "play_single_game")
                    # Try to continue the game
                    continue
            
            result = {
                'checkpoint_id': self.current_checkpoint_id,
                'game_history': game_history,
                'final_score': self.env.game.score,
                'total_reward': total_reward,
                'steps': step_count,
                'max_tile': self.env.game.get_max_tile()
            }
            
            print(f"Game completed: {step_count} steps, score={self.env.game.score}, max_tile={self.env.game.get_max_tile()}")
            return result
            
        except Exception as e:
            self._log_error(f"Fatal error in play_single_game: {str(e)}", "play_single_game")
            print(f"Full traceback: {traceback.format_exc()}")
            return {"error": f"Game failed: {str(e)}"}
    
    async def start_live_playback(self, websocket_manager, speed: float = 1.0):
        """Start live playback streaming to websocket"""
        if self.current_model is None:
            self._log_error("No model loaded for playback", "start_live_playback")
            return
        
        self.is_playing = True
        self.is_paused = False
        self.playback_speed = speed
        
        print(f"Starting live playback for checkpoint {self.current_checkpoint_id} at speed {speed}")
        
        # Send initial status message
        success = await self._safe_broadcast(websocket_manager, {
            'type': 'playback_status',
            'message': f'Starting playback for checkpoint {self.current_checkpoint_id}',
            'checkpoint_id': self.current_checkpoint_id,
            'status': 'starting'
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
                
                # Check if we have any connected clients
                if websocket_manager.get_connection_count() == 0:
                    print("No connected clients, pausing playback...")
                    await asyncio.sleep(1.0)
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
                
                # Stream game history step by step
                game_finished = False
                for step_idx, step_data in enumerate(game_result['game_history'][start_step:], start_step):
                    if not self.is_playing:
                        print("Playback stopped during game streaming")
                        self.current_game_result = None
                        self.current_step_index = 0
                        return
                    
                    if self.is_paused:
                        print(f"Playback paused at step {step_idx} of game {game_count}")
                        self.current_step_index = step_idx
                        break
                    
                    # Check for step timeout (detect stuck states like "stuck at move 17")
                    current_time = time.time()
                    step_duration = current_time - last_step_time
                    total_step_time = current_time - step_start_time
                    
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
                        await asyncio.sleep(1.0)
                    
                    # Update current step index
                    self.current_step_index = step_idx
                    last_step_time = current_time
                    
                    # Send step data to frontend
                    message = {
                        'type': 'checkpoint_playback',
                        'checkpoint_id': self.current_checkpoint_id,
                        'game_number': game_count,
                        'step_data': step_data if self.playback_speed <= 2.0 else {
                            # Lightweight payload for high-speed playback (>2x)
                            'step': step_data['step'],
                            'board_state': step_data['board_state'],
                            'score': step_data['score'],
                            'action': step_data['action'],
                            'timestamp': step_data['timestamp']
                        },
                        'game_summary': {
                            'final_score': game_result['final_score'],
                            'total_steps': game_result['steps'],
                            'max_tile': game_result['max_tile']
                        }
                    }
                    
                    success = await self._safe_broadcast(websocket_manager, message, f"step_{step_idx}")
                    if not success:
                        self._log_error(f"Failed to broadcast step {step_idx}", "start_live_playback")
                        # Don't break immediately, try a few more times
                        if self.consecutive_failures >= 3:
                            print(f"Too many consecutive broadcast failures, trying recovery...")
                            
                            # Attempt recovery instead of breaking
                            await self._attempt_broadcast_recovery(websocket_manager)
                            
                            # If still failing after recovery, pause briefly and continue
                            if self.consecutive_failures >= self.max_consecutive_failures:
                                print(f"Recovery failed, pausing playback for 5 seconds...")
                                await asyncio.sleep(5.0)
                                
                                # Reset failure count to give it another chance
                                self.consecutive_failures = 0
                                
                                # Send a status update about the issue
                                await self._safe_broadcast(websocket_manager, {
                                    'type': 'playback_status',
                                    'message': f'Playback experiencing connection issues, continuing...',
                                    'checkpoint_id': self.current_checkpoint_id,
                                    'status': 'recovering'
                                }, "recovery_status")
                    else:
                        print(f"Broadcasted step {step_idx} of game {game_count}")
                    
                    # Wait based on playback speed
                    wait_time = max(0.1, 1.0 / self.playback_speed)
                    await asyncio.sleep(wait_time)
                    
                    # Check if this was the last step
                    if step_idx == len(game_result['game_history']) - 1:
                        game_finished = True
                
                # If we finished the game completely, send completion message and reset
                if game_finished:
                    await self._safe_broadcast(websocket_manager, {
                        'type': 'game_completed',
                        'checkpoint_id': self.current_checkpoint_id,
                        'game_number': game_count,
                        'final_score': game_result['final_score'],
                        'total_steps': game_result['steps'],
                        'max_tile': game_result['max_tile']
                    }, "game_completed")
                    
                    # Reset game state for next game
                    self.current_game_result = None
                    self.current_step_index = 0
                    
                    # Brief pause between games
                    if self.is_playing:
                        print(f"Waiting 2.0s before next game...")
                        await asyncio.sleep(2.0)
                
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
                'error_history': self.error_history[-5:]  # Send last 5 errors
            }, "final_status")
    
    async def _heartbeat_loop(self, websocket_manager):
        """Send periodic heartbeat messages to detect connection issues"""
        try:
            while self.is_playing:
                await asyncio.sleep(self.heartbeat_interval)
                
                if not self.is_playing:
                    break
                
                # Send heartbeat
                await self._safe_broadcast(websocket_manager, {
                    'type': 'playback_heartbeat',
                    'checkpoint_id': self.current_checkpoint_id,
                    'timestamp': time.time(),
                    'game_count': self.current_game_count,
                    'step_index': self.current_step_index,
                    'is_healthy': self._is_healthy(),
                    'consecutive_failures': self.consecutive_failures
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
    
    def set_playback_speed(self, speed: float):
        """Set playback speed (seconds between moves)"""
        self.playback_speed = max(0.1, min(5.0, speed))
        print(f"Playback speed set to {self.playback_speed}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        if self.current_model is None:
            return {"error": "No model loaded"}
        
        metadata = self.checkpoint_manager.get_checkpoint_metadata(self.current_checkpoint_id)
        
        return {
            'checkpoint_id': self.current_checkpoint_id,
            'nickname': metadata.nickname if metadata else 'Unknown',
            'episode': metadata.episode if metadata else 0,
            'model_config': metadata.model_config if metadata else {},
            'performance_metrics': metadata.performance_metrics if metadata else {},
            'parameter_count': self.current_model.count_parameters() if hasattr(self.current_model, 'count_parameters') else 0,
            'device': str(self.device)
        }
    
    def get_playback_status(self) -> Dict[str, Any]:
        """Get current playback status"""
        return {
            'is_playing': self.is_playing,
            'is_paused': self.is_paused,
            'playback_speed': self.playback_speed,
            'current_checkpoint': self.current_checkpoint_id,
            'model_loaded': self.current_model is not None,
            'is_healthy': self._is_healthy(),
            'last_heartbeat': self.last_heartbeat,
            'consecutive_failures': self.consecutive_failures,
            'error_count': len(self.error_history)
        }
    
    def get_current_step_data(self) -> Optional[Dict[str, Any]]:
        """Get current playback step data for polling fallback"""
        if not self.is_playing or self.current_model is None:
            return None
        
        if self.current_game_result is None:
            return None
        
        # Get the current step from the game history
        game_history = self.current_game_result.get('game_history', [])
        if not game_history or self.current_step_index >= len(game_history):
            return None
        
        current_step = game_history[self.current_step_index]
        
        return {
            'step_data': current_step,
            'game_summary': {
                'final_score': self.current_game_result.get('final_score', 0),
                'total_steps': self.current_game_result.get('steps', 0),
                'max_tile': self.current_game_result.get('max_tile', 0)
            },
            'game_number': self.current_game_count,
            'step_index': self.current_step_index,
            'total_steps': len(game_history),
            'health_status': {
                'is_healthy': self._is_healthy(),
                'last_heartbeat': self.last_heartbeat,
                'consecutive_failures': self.consecutive_failures
            }
        }
    
    def get_error_history(self) -> List[Dict[str, Any]]:
        """Get recent error history"""
        return self.error_history.copy()
    
    def clear_error_history(self):
        """Clear error history"""
        self.error_history.clear()
        print("Error history cleared") 