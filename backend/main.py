"""
Main FastAPI server for 2048 Transformer Training Visualization
"""
import json
import asyncio
import random
import time
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from app.api.websocket_manager import WebSocketManager
from app.training.training_manager import TrainingManager
from app.training.ppo_trainer import PPOTrainer
from app.models.checkpoint_playback import CheckpointPlayback

app = FastAPI(title="2048 Transformer Training API", version="1.0.0")

# Enable CORS for frontend
import os
# Get CORS origins from environment or use defaults
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# WebSocket manager for real-time updates
websocket_manager = WebSocketManager()
training_manager = TrainingManager(websocket_manager)
checkpoint_playback = CheckpointPlayback(training_manager.checkpoint_manager)

# Refresh checkpoint metadata on startup to infer model sizes
training_manager.checkpoint_manager.refresh_metadata_cache()

class TrainingConfig(BaseModel):
    model_size: str = "medium"
    learning_rate: float = 0.0003
    batch_size: int = 32
    n_experts: int = 6
    n_layers: int = 6

class TrainingStatus(BaseModel):
    is_training: bool = False
    is_paused: bool = False
    current_episode: int = 0
    total_episodes: int = 10000
    start_time: Optional[datetime] = None

# Global training state
training_status = TrainingStatus()
training_config = TrainingConfig()

# Utility to sync Pydantic status model with manager
def _update_training_status():
    training_status.is_training = training_manager.is_training
    training_status.is_paused = training_manager.is_paused
    training_status.current_episode = training_manager.current_episode


@app.get("/")
async def root():
    return {"message": "2048 Transformer Training API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/model/config")
async def get_model_config():
    # Get the actual model configuration being used by the trainer
    if hasattr(training_manager, 'current_config') and training_manager.current_config:
        actual_config = {
            "model_size": training_config.model_size,
            "learning_rate": training_config.learning_rate,
            "batch_size": training_config.batch_size,
            "n_experts": training_config.n_experts,
            "n_layers": training_config.n_layers,
            "actual_d_model": training_manager.current_config.d_model,
            "actual_n_heads": training_manager.current_config.n_heads,
            "actual_n_layers": training_manager.current_config.n_layers,
            "actual_n_experts": training_manager.current_config.n_experts,
            "estimated_params": training_manager.current_config.estimated_params,
        }
    else:
        actual_config = training_config.dict()
        actual_config.update({
            "actual_d_model": None,
            "actual_n_heads": None,
            "actual_n_layers": None,
            "actual_n_experts": None,
            "estimated_params": None,
        })
    
    return actual_config

@app.post("/model/config")
async def update_model_config(config: TrainingConfig):
    global training_config
    training_config = config
    return {"message": "Configuration updated", "config": training_config}

@app.get("/training/status")
async def get_training_status():
    _update_training_status()
    return training_status

@app.get("/mobile-test")
async def mobile_test():
    """Simple endpoint for mobile connectivity testing"""
    return {
        "status": "ok",
        "message": "Mobile connection successful", 
        "timestamp": time.time(),
        "server_ip": "accessible"
    }


# ---------------------------- Control Endpoints -----------------------------

@app.post("/training/start")
async def start_training(request: dict = None):
    # Update training manager with current configuration
    model_size = request.get("model_size", training_config.model_size) if request else training_config.model_size
    
    training_manager.update_config({
        "model_size": model_size,
        "learning_rate": training_config.learning_rate,
        "batch_size": training_config.batch_size,
        "n_experts": training_config.n_experts,
        "n_layers": training_config.n_layers,
    })
    
    training_manager.start()
    _update_training_status()
    return {"message": "Training started", "status": training_status}

@app.get("/training/config")
async def get_training_config():
    return {
        "model_size": training_config.model_size,
        "learning_rate": training_config.learning_rate,
        "batch_size": training_config.batch_size,
        "n_experts": training_config.n_experts,
        "n_layers": training_config.n_layers,
    }

@app.post("/training/pause")
async def pause_training():
    training_manager.pause()
    _update_training_status()
    return {"message": "Training paused", "status": training_status}

@app.post("/training/resume")
async def resume_training():
    training_manager.resume()
    _update_training_status()
    return {"message": "Training resumed", "status": training_status}

@app.post("/training/stop")
async def stop_training():
    await training_manager.stop()
    _update_training_status()
    return {"message": "Training stopped", "status": training_status}

@app.post("/training/reset")
async def reset_training():
    """Reset training state and start fresh with a new model"""
    try:
        # Stop ongoing training if necessary
        if training_manager.is_training:
            await training_manager.stop()
        
        # Reset to fresh model
        training_manager.reset_to_fresh_model()
        
        # Update training status
        _update_training_status()
        
        # Notify all connected clients
        await websocket_manager.broadcast({
            'type': 'training_reset',
            'message': 'Training reset to fresh model',
            'current_episode': 0,
            'is_training': False,
            'is_paused': False
        })
        
        return {
            "message": "Training reset to fresh model",
            "status": training_status
        }
    except Exception as e:
        print(f"Error resetting training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting training: {str(e)}")

@app.get("/checkpoints")
async def list_checkpoints():
    """List all checkpoints with metadata"""
    try:
        checkpoints = training_manager.checkpoint_manager.list_checkpoints()
        return [
            {
                "id": cp.id,
                "nickname": cp.nickname,
                "episode": cp.episode,
                "created_at": cp.created_at,
                "training_duration": cp.training_duration,
                "model_config": cp.model_config,
                "performance_metrics": cp.performance_metrics,
                "file_size": cp.file_size,
                "tags": cp.tags,
            }
            for cp in checkpoints
        ]
    except Exception as e:
        print(f"Error listing checkpoints: {e}")
        return []

@app.get("/checkpoints/stats")
async def get_checkpoint_stats():
    """Get statistics about all checkpoints"""
    try:
        return training_manager.checkpoint_manager.get_checkpoint_stats()
    except Exception as e:
        print(f"Error getting checkpoint stats: {e}")
        return {
            'total_checkpoints': 0,
            'total_size': 0,
            'best_score': 0,
            'latest_episode': 0,
            'total_training_time': 0
        }

@app.post("/checkpoints/refresh")
async def refresh_checkpoint_metadata():
    """Refresh checkpoint metadata cache and update model sizes"""
    try:
        training_manager.checkpoint_manager.refresh_metadata_cache()
        return {"message": "Checkpoint metadata refreshed successfully"}
    except Exception as e:
        print(f"Error refreshing checkpoint metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Error refreshing metadata: {str(e)}")

@app.get("/checkpoints/{checkpoint_id}")
async def get_checkpoint_info(checkpoint_id: str):
    """Get detailed info about a specific checkpoint"""
    metadata = training_manager.checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    
    return {
        "id": metadata.id,
        "nickname": metadata.nickname,
        "episode": metadata.episode,
        "created_at": metadata.created_at,
        "training_duration": metadata.training_duration,
        "model_config": metadata.model_config,
        "performance_metrics": metadata.performance_metrics,
        "file_size": metadata.file_size,
        "parent_checkpoint": metadata.parent_checkpoint,
        "tags": metadata.tags,
    }

@app.post("/checkpoints/{checkpoint_id}/nickname")
async def update_checkpoint_nickname(checkpoint_id: str, request: dict):
    """Update checkpoint nickname"""
    nickname = request.get("nickname", "")
    success = training_manager.checkpoint_manager.update_checkpoint_nickname(checkpoint_id, nickname)
    if not success:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return {"message": "Nickname updated successfully"}

@app.delete("/checkpoints/{checkpoint_id}")
async def delete_checkpoint(checkpoint_id: str):
    """Delete a checkpoint"""
    success = training_manager.checkpoint_manager.delete_checkpoint(checkpoint_id)
    if not success:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return {"message": "Checkpoint deleted successfully"}

@app.post("/checkpoints/{checkpoint_id}/load")
async def load_checkpoint_for_training(checkpoint_id: str):
    """Load a checkpoint and resume training automatically"""
    try:
        # Stop ongoing training if necessary
        if training_manager.is_training:
            await training_manager.stop()

        metadata = training_manager.checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        checkpoint_path = training_manager.checkpoint_manager._get_checkpoint_path(checkpoint_id)
        if not checkpoint_path.exists():
            raise HTTPException(status_code=404, detail="Checkpoint file not found")

        # Load checkpoint data to get the config
        import torch, time as _time
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Get the config from the checkpoint
        config = checkpoint_data.get('config')
        if not config:
            raise HTTPException(status_code=400, detail="Checkpoint does not contain model config")

        # Recreate trainer with the correct config
        training_manager.current_config = config
        training_manager.trainer = PPOTrainer(
            config=config,
            learning_rate=0.0003  # Default learning rate, can be adjusted via UI later
        )

        # Load model/optimiser state
        training_manager.trainer.load_checkpoint(str(checkpoint_path))

        # ------------------------------------------------------------------ Sync manager state
        training_manager.current_episode = training_manager.trainer.episode_count
        training_manager._game_lengths = []
        training_manager._episode_start_times = []
        training_manager._start_time = _time.time()
        training_manager.is_paused = False

        # Automatically resume training from this checkpoint
        training_manager.start()

        print(f"Loaded checkpoint {checkpoint_id} (episode {training_manager.current_episode}) and resumed training.")

        # Update training status and notify frontend
        _update_training_status()
        await websocket_manager.broadcast({
            'type': 'training_status_update',
            'is_training': True,
            'is_paused': False,
            'current_episode': training_manager.current_episode,
            'message': f'Training resumed from checkpoint {checkpoint_id}'
        })

        return {
            "message": f"Checkpoint {checkpoint_id} loaded and training resumed",
            "episode": training_manager.current_episode,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading checkpoint: {str(e)}")

@app.post("/checkpoints/save")
async def save_checkpoint_manual():
    """Manually save current checkpoint"""
    try:
        if not training_manager.trainer:
            raise HTTPException(status_code=400, detail="No training session active")
        
        checkpoint_id = f"checkpoint_manual_{int(time.time())}"
        checkpoint_path = os.path.join(
            training_manager.checkpoint_dir,
            f"{checkpoint_id}.pt"
        )
        
        training_manager.trainer.save_checkpoint(checkpoint_path)
        
        # Create metadata
        training_duration = time.time() - training_manager._start_time if training_manager._start_time else 0
        n_experts = getattr(training_manager.current_config, 'n_experts', 6)
        inferred_size = training_manager.checkpoint_manager._infer_model_size_from_experts(n_experts)
        model_config = {
            'model_size': getattr(training_manager.current_config, 'model_size', inferred_size),
            'learning_rate': 0.0003,
            'n_experts': n_experts,
            'n_layers': getattr(training_manager.current_config, 'n_layers', 6),
            'd_model': getattr(training_manager.current_config, 'd_model', 384),
            'n_heads': getattr(training_manager.current_config, 'n_heads', 8),
        }
        
        performance_metrics = {
            'best_score': training_manager.trainer.best_score,
            'avg_score': 0,  # Would need calculation
            'final_loss': 0.0,  # Would need current loss
            'training_speed': 0.0,  # Would need calculation
        }
        
        metadata = training_manager.checkpoint_manager.create_checkpoint_metadata(
            checkpoint_id=checkpoint_id,
            episode=training_manager.current_episode,
            training_duration=training_duration,
            model_config=model_config,
            performance_metrics=performance_metrics,
            nickname=f"Manual Save - Episode {training_manager.current_episode}",
            tags=['manual']
        )
        
        return {"message": "Checkpoint saved successfully", "checkpoint_id": checkpoint_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving checkpoint: {str(e)}")

# Checkpoint playback endpoints
@app.post("/checkpoints/{checkpoint_id}/playback/start")
async def start_checkpoint_playback(checkpoint_id: str, request: dict = None):
    """Start live playback of a checkpoint"""
    try:
        # Stop any existing playback first
        checkpoint_playback.stop_playback()
        
        # Get speed from request if provided
        speed = 1.0
        if request:
            speed = max(0.5, min(request.get("speed", 1.0), 5.0))  # Clamp between 0.5x and 5x
        
        # Load the checkpoint
        success = checkpoint_playback.load_checkpoint(checkpoint_id)
        if not success:
            raise HTTPException(status_code=404, detail="Checkpoint not found or failed to load")
        
        # Start playback in background
        asyncio.create_task(checkpoint_playback.start_live_playback(websocket_manager, speed))
        
        return {
            "message": f"Playback started for checkpoint {checkpoint_id}",
            "checkpoint_id": checkpoint_id,
            "speed": speed,
            "connected_clients": websocket_manager.get_connection_count()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error starting playback for checkpoint {checkpoint_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting playback: {str(e)}")

@app.post("/checkpoints/{checkpoint_id}/playback/game")
async def play_single_game(checkpoint_id: str):
    """Play a single game with a checkpoint and return the full history"""
    try:
        # Load the checkpoint
        success = checkpoint_playback.load_checkpoint(checkpoint_id)
        if not success:
            raise HTTPException(status_code=404, detail="Checkpoint not found or failed to load")
        
        # Play the game
        game_result = checkpoint_playback.play_single_game()
        
        if 'error' in game_result:
            raise HTTPException(status_code=500, detail=game_result['error'])
        
        # Add metadata to response
        game_result['checkpoint_id'] = checkpoint_id
        game_result['played_at'] = time.time()
        
        return game_result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error playing single game for checkpoint {checkpoint_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error playing game: {str(e)}")

@app.post("/checkpoints/playback/pause")
async def pause_checkpoint_playback():
    """Pause current playback"""
    try:
        checkpoint_playback.pause_playback()
        return {
            "message": "Playback paused",
            "is_paused": True,
            "current_checkpoint": checkpoint_playback.current_checkpoint_id
        }
    except Exception as e:
        print(f"Error pausing playback: {e}")
        raise HTTPException(status_code=500, detail=f"Error pausing playback: {str(e)}")

@app.post("/checkpoints/playback/resume")
async def resume_checkpoint_playback():
    """Resume current playback"""
    try:
        checkpoint_playback.resume_playback()
        return {
            "message": "Playback resumed",
            "is_paused": False,
            "current_checkpoint": checkpoint_playback.current_checkpoint_id
        }
    except Exception as e:
        print(f"Error resuming playback: {e}")
        raise HTTPException(status_code=500, detail=f"Error resuming playback: {str(e)}")

@app.post("/checkpoints/playback/stop")
async def stop_checkpoint_playback():
    """Stop current playback"""
    try:
        checkpoint_playback.stop_playback()
        return {
            "message": "Playback stopped",
            "is_playing": False,
            "is_paused": False
        }
    except Exception as e:
        print(f"Error stopping playback: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping playback: {str(e)}")

@app.post("/checkpoints/playback/speed")
async def set_playback_speed(request: dict):
    """Set playback speed"""
    try:
        speed = request.get("speed", 1.0)
        speed = max(0.5, min(speed, 5.0))
        
        checkpoint_playback.set_playback_speed(speed)
        return {
            "message": f"Playback speed set to {speed}",
            "speed": speed,
            "current_checkpoint": checkpoint_playback.current_checkpoint_id
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error setting playback speed: {e}")
        raise HTTPException(status_code=500, detail=f"Error setting playback speed: {str(e)}")

@app.get("/checkpoints/playback/status")
async def get_playback_status():
    """Get current playback status"""
    try:
        status = checkpoint_playback.get_playback_status()
        # Add additional status information
        status['server_time'] = time.time()
        status['connected_clients'] = websocket_manager.get_connection_count()
        return status
    except Exception as e:
        print(f"Error getting playback status: {e}")
        # Return a safe default status
        return {
            'is_playing': False,
            'is_paused': False,
            'playback_speed': 1.0,
            'current_checkpoint': None,
            'model_loaded': False,
            'server_time': time.time(),
            'connected_clients': websocket_manager.get_connection_count(),
            'error': str(e)
        }

@app.get("/checkpoints/playback/current")
async def get_current_playback_data():
    """Get current playback step data for polling fallback"""
    try:
        status = checkpoint_playback.get_playback_status()
        
        if not status['is_playing'] or not status['model_loaded']:
            return {
                'has_data': False,
                'status': status,
                'error': 'No active playback or model not loaded'
            }
        
        # Get the current step data from the playback system
        current_data = checkpoint_playback.get_current_step_data()
        
        if current_data is None:
            return {
                'has_data': False,
                'status': status,
                'error': 'No current step data available'
            }
        
        return {
            'has_data': True,
            'status': status,
            'playback_data': {
                'type': 'checkpoint_playback',
                'checkpoint_id': status['current_checkpoint'],
                'step_data': current_data['step_data'],
                'game_summary': current_data['game_summary'],
                'timestamp': time.time()
            }
        }
        
    except Exception as e:
        print(f"Error getting current playback data: {e}")
        return {
            'has_data': False,
            'status': {
                'is_playing': False,
                'is_paused': False,
                'current_checkpoint': None,
                'error': str(e)
            },
            'error': str(e)
        }

@app.get("/checkpoints/playback/model")
async def get_playback_model_info():
    """Get information about the currently loaded playback model"""
    try:
        return checkpoint_playback.get_model_info()
    except Exception as e:
        print(f"Error getting playback model info: {e}")
        return {
            "error": "No model loaded or error accessing model info",
            "details": str(e),
            "model_loaded": False
        }

@app.get("/checkpoints/playback/errors")
async def get_playback_errors():
    """Get recent playback error history for debugging"""
    try:
        return {
            "error_history": checkpoint_playback.get_error_history(),
            "error_count": len(checkpoint_playback.get_error_history()),
            "current_status": checkpoint_playback.get_playback_status()
        }
    except Exception as e:
        print(f"Error getting playback errors: {e}")
        return {
            "error": "Failed to get error history",
            "details": str(e),
            "error_history": []
        }

@app.post("/checkpoints/playback/clear-errors")
async def clear_playback_errors():
    """Clear playback error history"""
    try:
        checkpoint_playback.clear_error_history()
        return {
            "message": "Error history cleared",
            "timestamp": time.time()
        }
    except Exception as e:
        print(f"Error clearing playback errors: {e}")
        return {
            "error": "Failed to clear error history",
            "details": str(e)
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Get user agent from headers for mobile detection
    user_agent = websocket.headers.get("user-agent", "")
    
    await websocket_manager.connect(websocket, user_agent)
    
    # Get connection info for adaptive behavior
    conn_info = websocket_manager.get_connection_info(websocket)
    adaptive_timeout = conn_info.get_adaptive_timeout() if conn_info else 1.0
    
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            try:
                # Use adaptive timeout based on device type
                data = await asyncio.wait_for(websocket.receive_text(), timeout=adaptive_timeout)
                
                # Parse the incoming message
                try:
                    message = json.loads(data)
                    
                    # Handle ping for latency measurement
                    if message.get('type') == 'ping':
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": message.get('timestamp'),
                            "server_time": time.time()
                        }))
                        continue
                    
                    # Echo back other messages for testing with mobile-aware response
                    await websocket.send_text(json.dumps({
                        "type": "echo",
                        "message": f"Received: {data}",
                        "timestamp": time.time(),
                        "mobile_optimized": conn_info.is_mobile if conn_info else False
                    }))
                    
                except json.JSONDecodeError:
                    # Handle plain text messages
                    await websocket.send_text(json.dumps({
                        "type": "echo",
                        "message": f"Received: {data}",
                        "timestamp": time.time(),
                        "mobile_optimized": conn_info.is_mobile if conn_info else False
                    }))
                
            except asyncio.TimeoutError:
                # No message received within timeout - adaptive heartbeat is handled by manager
                continue
                
            except WebSocketDisconnect:
                # Client disconnected normally
                break
                
            except Exception as e:
                print(f"WebSocket message error: {e}")
                # Continue trying to receive messages
                continue
                
    except WebSocketDisconnect:
        pass  # Normal disconnection
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        websocket_manager.disconnect(websocket)

# Add endpoint to get connection statistics
@app.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    return websocket_manager.get_connection_stats()

# Remove mock_training_loop in favour of TrainingManager

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    ) 