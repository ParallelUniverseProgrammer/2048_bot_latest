"""
Checkpoint playback system for watching saved models play 2048
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import asyncio
import time

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
    
    def load_checkpoint(self, checkpoint_id: str) -> bool:
        """Load a checkpoint for playback"""
        try:
            metadata = self.checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
            if not metadata:
                return False
            
            checkpoint_path = self.checkpoint_manager._get_checkpoint_path(checkpoint_id)
            if not checkpoint_path.exists():
                return False
            
            # Load checkpoint data
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Get model config
            config = checkpoint_data.get('config')
            if not config:
                return False
            
            # Create model
            self.current_model = GameTransformer(config).to(self.device)
            self.current_model.load_state_dict(checkpoint_data['model_state_dict'])
            self.current_model.eval()
            
            self.current_config = config
            self.current_checkpoint_id = checkpoint_id
            
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_id} for playback: {e}")
            return False
    
    def select_action(self, state: np.ndarray, legal_actions: List[int], env_game) -> Tuple[int, List[float], Optional[np.ndarray]]:
        """Select action using loaded model with fallback for invalid moves"""
        if self.current_model is None:
            return 0, [0.25, 0.25, 0.25, 0.25], None
        
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
    
    def play_single_game(self) -> Dict[str, Any]:
        """Play a single game and return the full game history"""
        if self.current_model is None:
            return {"error": "No model loaded"}
        
        game_history = []
        self.env.reset()
        
        total_reward = 0
        step_count = 0
        max_steps = 10000  # Prevent infinite games
        
        while not self.env.is_done() and step_count < max_steps:
            # Get current state
            state = self.env.get_state()
            legal_actions = self.env.get_legal_actions()
            
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
                    'board_state': [list(row) for row in self.env.game.board],  # Final state after last move
                    'score': self.env.game.score,
                    'action': None,  # No action to take - game is over
                    'action_probs': [0.0, 0.0, 0.0, 0.0],  # No valid actions
                    'legal_actions': [],  # No legal actions
                    'attention_weights': None,
                    'timestamp': time.time(),
                    'reward': 0,
                    'done': True
                }
                game_history.append(final_step_data)
                break
        
        return {
            'checkpoint_id': self.current_checkpoint_id,
            'game_history': game_history,
            'final_score': self.env.game.score,
            'total_reward': total_reward,
            'steps': step_count,
            'max_tile': self.env.game.get_max_tile()
        }
    
    async def start_live_playback(self, websocket_manager, speed: float = 1.0):
        """Start live playback streaming to websocket"""
        if self.current_model is None:
            print("Error: No model loaded for playback")
            return
        
        self.is_playing = True
        self.is_paused = False
        self.playback_speed = speed
        
        print(f"Starting live playback for checkpoint {self.current_checkpoint_id} at speed {speed}")
        
        # Send initial status message
        try:
            await websocket_manager.broadcast({
                'type': 'playback_status',
                'message': f'Starting playback for checkpoint {self.current_checkpoint_id}',
                'checkpoint_id': self.current_checkpoint_id,
                'status': 'starting'
            })
        except Exception as e:
            print(f"Error sending initial status: {e}")
        
        # Initialize game count if not resuming
        if self.current_game_result is None:
            self.current_game_count = 0
        
        try:
            while self.is_playing:
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
                else:
                    # Start new game
                    self.current_game_count += 1
                    game_count = self.current_game_count
                    print(f"Playing game #{game_count} for checkpoint {self.current_checkpoint_id}")
                    
                    # Send new game started message
                    try:
                        await websocket_manager.broadcast({
                            'type': 'new_game_started',
                            'checkpoint_id': self.current_checkpoint_id,
                            'game_number': game_count,
                            'message': f'Starting game #{game_count}'
                        })
                    except Exception as e:
                        print(f"Error sending new game started message: {e}")
                    
                    # Play one game
                    game_result = self.play_single_game()
                    
                    if 'error' in game_result:
                        print(f"Game error: {game_result['error']}")
                        await websocket_manager.broadcast({
                            'type': 'playback_error',
                            'error': game_result['error'],
                            'checkpoint_id': self.current_checkpoint_id
                        })
                        break
                    
                    print(f"Game #{game_count} completed: score={game_result.get('final_score', 0)}, steps={game_result.get('steps', 0)}")
                    
                    # Store current game state and start from beginning
                    self.current_game_result = game_result
                    start_step = 0
                
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
                    
                    # Send step data to frontend
                    message = {
                        'type': 'checkpoint_playback',
                        'checkpoint_id': self.current_checkpoint_id,
                        'game_number': game_count,
                        'step_data': step_data,
                        'game_summary': {
                            'final_score': game_result['final_score'],
                            'total_steps': game_result['steps'],
                            'max_tile': game_result['max_tile']
                        }
                    }
                    
                    try:
                        await websocket_manager.broadcast(message)
                        print(f"Broadcasted step {step_idx} of game {game_count}")
                    except Exception as e:
                        print(f"Error broadcasting playback message: {e}")
                        # Continue playing even if broadcast fails
                    
                    # Wait based on playback speed (invert for intuitive speed control)
                    # Speed 2.0x = wait 0.5 seconds, Speed 0.5x = wait 2.0 seconds
                    wait_time = max(0.1, 1.0 / self.playback_speed)
                    await asyncio.sleep(wait_time)
                    
                    # Check if this was the last step
                    if step_idx == len(game_result['game_history']) - 1:
                        game_finished = True
                
                # If we finished the game completely, send completion message and reset
                if game_finished:
                    try:
                        await websocket_manager.broadcast({
                            'type': 'game_completed',
                            'checkpoint_id': self.current_checkpoint_id,
                            'game_number': game_count,
                            'final_score': game_result['final_score'],
                            'total_steps': game_result['steps'],
                            'max_tile': game_result['max_tile']
                        })
                    except Exception as e:
                        print(f"Error sending game completion message: {e}")
                    
                    # Reset game state for next game
                    self.current_game_result = None
                    self.current_step_index = 0
                    
                    # Brief pause between games
                    if self.is_playing:
                        print(f"Waiting {2.0}s before next game...")
                        await asyncio.sleep(2.0)
                
        except asyncio.CancelledError:
            print("Playback cancelled")
        except Exception as e:
            print(f"Error in live playback: {e}")
            try:
                await websocket_manager.broadcast({
                    'type': 'playback_error',
                    'error': str(e),
                    'checkpoint_id': self.current_checkpoint_id
                })
            except:
                pass
        finally:
            self.is_playing = False
            self.is_paused = False
            self.current_game_result = None
            self.current_step_index = 0
            print(f"Playback stopped for checkpoint {self.current_checkpoint_id} after {self.current_game_count} games")
            
            # Send final status message
            try:
                await websocket_manager.broadcast({
                    'type': 'playback_status',
                    'message': f'Playback stopped for checkpoint {self.current_checkpoint_id}',
                    'checkpoint_id': self.current_checkpoint_id,
                    'status': 'stopped',
                    'games_played': self.current_game_count
                })
            except Exception as e:
                print(f"Error sending final status: {e}")
    
    def pause_playback(self):
        """Pause live playback"""
        self.is_paused = True
    
    def resume_playback(self):
        """Resume live playback"""
        self.is_paused = False
    
    def stop_playback(self):
        """Stop live playback"""
        self.is_playing = False
        self.is_paused = False
        # Reset game state when stopping
        self.current_game_result = None
        self.current_step_index = 0
        self.current_game_count = 0
    
    def set_playback_speed(self, speed: float):
        """Set playback speed (seconds between moves)"""
        self.playback_speed = max(0.1, min(5.0, speed))
    
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
            'model_loaded': self.current_model is not None
        } 