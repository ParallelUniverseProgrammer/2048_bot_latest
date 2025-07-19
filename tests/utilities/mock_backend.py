#!/usr/bin/env python3
"""
Mock Backend Server for Offline Testing
=====================================

This module provides a mock backend server that can be used for testing
when the real backend is not available. It simulates the main API endpoints
and provides realistic responses for development and testing purposes.

Features:
- Mock checkpoint endpoints with realistic data
- Mock training status endpoints
- Mock WebSocket connections
- Configurable response delays for testing
- Support for error simulation
- Lightweight and fast startup

Usage:
    python mock_backend.py
    
    # Or use as a module
    from mock_backend import MockBackend
    server = MockBackend()
    server.start()
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random
import threading
import queue
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import socketserver

from tests.utilities.test_utils import TestLogger

class MockData:
    """Mock data generator for backend responses"""
    
    def __init__(self):
        self.checkpoints = self._generate_mock_checkpoints()
        self.current_checkpoint = None
        self.is_training = False
        self.is_playing = False
        self.connected_clients = 0
        self.training_status = self._generate_mock_training_status()
        self.playback_status = self._generate_mock_playback_status()
    
    def _generate_mock_checkpoints(self) -> List[Dict[str, Any]]:
        """Generate mock checkpoint data"""
        checkpoints = []
        base_time = datetime.now()
        
        for i in range(5):
            episode = (i + 1) * 300
            score = 1000 + (i * 2000) + random.randint(-500, 500)
            max_tile = 2 ** (5 + i)  # 32, 64, 128, 256, 512
            
            checkpoint = {
                "id": f"mock_checkpoint_episode_{episode}",
                "name": f"Mock Checkpoint Episode {episode}",
                "timestamp": (base_time - timedelta(hours=i)).isoformat(),
                "episode_count": episode,
                "score": score,
                "max_tile": max_tile,
                "training_time": 1800 + (i * 600),  # 30 minutes + 10 min per episode
                "total_steps": episode * 150,
                "model_size": 143000000 + (i * 1000000),  # ~143M parameters
                "file_size": 435000000 + (i * 10000000),  # ~435MB file size
                "metadata": {
                    "model_config": {
                        "d_model": 512,
                        "n_heads": 16,
                        "n_layers": 8,
                        "vocab_size": 16,
                        "max_seq_len": 512
                    },
                    "training_config": {
                        "learning_rate": 0.0001,
                        "batch_size": 32,
                        "optimizer": "Adam",
                        "scheduler": "CosineAnnealingLR"
                    },
                    "performance": {
                        "avg_score": score,
                        "max_score": score + random.randint(500, 1500),
                        "avg_game_length": 150 + random.randint(-30, 30),
                        "win_rate": min(0.85, 0.1 + (i * 0.15))
                    }
                }
            }
            checkpoints.append(checkpoint)
        
        return checkpoints
    
    def _generate_mock_training_status(self) -> Dict[str, Any]:
        """Generate mock training status"""
        return {
            "is_training": self.is_training,
            "is_paused": False,
            "current_episode": 1500,
            "total_episodes": 10000,
            "start_time": (datetime.now() - timedelta(hours=2)).isoformat(),
            "elapsed_time": 7200,
            "estimated_remaining": 14400,
            "current_score": 8500,
            "best_score": 12000,
            "avg_score": 6500,
            "learning_rate": 0.0001,
            "memory_usage": {
                "gpu_memory": 3.2,
                "system_memory": 8.5,
                "gpu_utilization": 0.85
            },
            "performance": {
                "games_per_second": 0.8,
                "steps_per_second": 120,
                "avg_game_length": 150
            }
        }
    
    def _generate_mock_playback_status(self) -> Dict[str, Any]:
        """Generate mock playback status"""
        return {
            "is_playing": self.is_playing,
            "is_paused": False,
            "current_checkpoint": self.current_checkpoint,
            "model_loaded": self.current_checkpoint is not None,
            "current_game": random.randint(0, 100) if self.is_playing else 0,
            "current_step": random.randint(0, 200) if self.is_playing else 0,
            "is_healthy": True,
            "performance_metrics": {
                "total_broadcasts": random.randint(1000, 5000),
                "successful_broadcasts": random.randint(950, 4950),
                "failed_broadcasts": random.randint(0, 50),
                "avg_broadcast_time": 0.1 + random.random() * 0.5,
                "slow_broadcasts": random.randint(0, 100),
                "steps_skipped": random.randint(0, 10),
                "games_completed": random.randint(50, 200)
            },
            "adaptive_settings": {
                "broadcast_interval": 0.5,
                "lightweight_mode": False,
                "adaptive_skip": 1,
                "target_fps": 10
            },
            "server_time": time.time(),
            "connected_clients": self.connected_clients
        }
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics"""
        total_size = sum(cp["file_size"] for cp in self.checkpoints)
        best_score = max(cp["score"] for cp in self.checkpoints)
        latest_episode = max(cp["episode_count"] for cp in self.checkpoints)
        total_training_time = sum(cp["training_time"] for cp in self.checkpoints)
        
        return {
            "total_checkpoints": len(self.checkpoints),
            "total_size": total_size,
            "best_score": best_score,
            "latest_episode": latest_episode,
            "total_training_time": total_training_time,
            "avg_score": sum(cp["score"] for cp in self.checkpoints) / len(self.checkpoints),
            "storage_usage": {
                "used_space": total_size,
                "available_space": 10000000000,  # 10GB
                "usage_percentage": (total_size / 10000000000) * 100
            }
        }
    
    def get_checkpoint_by_id(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get checkpoint by ID"""
        for checkpoint in self.checkpoints:
            if checkpoint["id"] == checkpoint_id:
                return checkpoint
        return None
    
    def simulate_game_result(self) -> Dict[str, Any]:
        """Simulate a single game result"""
        steps = random.randint(50, 300)
        score = random.randint(500, 15000)
        max_tile = 2 ** random.randint(5, 9)  # 32 to 512
        
        # Generate game history
        game_history = []
        for step in range(min(steps, 100)):  # Limit history to 100 steps
            game_history.append({
                "step": step,
                "action": random.randint(0, 3),
                "score": int(score * (step / steps)),
                "board": [[random.randint(0, 11) for _ in range(4)] for _ in range(4)],
                "reward": random.uniform(-0.1, 3.0),
                "done": step == steps - 1
            })
        
        return {
            "game_history": game_history,
            "final_score": score,
            "max_tile": max_tile,
            "steps": steps,
            "total_reward": sum(step["reward"] for step in game_history),
            "completed": True,
            "play_time": steps * 0.5,  # ~0.5 seconds per step
            "performance": {
                "avg_decision_time": 0.1,
                "max_decision_time": 0.5,
                "actions_per_second": 2.0
            }
        }


class MockBackendHandler(BaseHTTPRequestHandler):
    """HTTP request handler for mock backend"""
    
    def __init__(self, request, client_address, server, mock_data: MockData = None):
        self.mock_data = mock_data or MockData()
        super().__init__(request, client_address, server)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        # Add small delay to simulate real backend
        time.sleep(0.1)
        
        if path == "/":
            self._send_response(200, {"status": "ok", "message": "Mock Backend Server"})
        
        elif path == "/checkpoints":
            self._send_response(200, self.mock_data.checkpoints)
        
        elif path == "/checkpoints/stats":
            self._send_response(200, self.mock_data.get_checkpoint_stats())
        
        elif path == "/training/status":
            self._send_response(200, self.mock_data._generate_mock_training_status())
        
        elif path == "/checkpoints/playback/status":
            self._send_response(200, self.mock_data._generate_mock_playback_status())
        
        elif path == "/checkpoints/playback/current":
            if self.mock_data.current_checkpoint:
                current_data = {
                    "has_data": True,
                    "checkpoint_id": self.mock_data.current_checkpoint,
                    "current_game": random.randint(0, 100),
                    "current_step": random.randint(0, 200),
                    "board": [[random.randint(0, 11) for _ in range(4)] for _ in range(4)],
                    "score": random.randint(1000, 10000),
                    "action_probs": [random.random() for _ in range(4)],
                    "attention_weights": [[random.random() for _ in range(4)] for _ in range(4)]
                }
                self._send_response(200, current_data)
            else:
                self._send_response(200, {"has_data": False, "error": "No active playback or model not loaded"})
        
        elif path.startswith("/checkpoints/") and not path.endswith("/"):
            checkpoint_id = path.split("/")[-1]
            checkpoint = self.mock_data.get_checkpoint_by_id(checkpoint_id)
            if checkpoint:
                self._send_response(200, checkpoint)
            else:
                self._send_response(404, {"error": f"Checkpoint {checkpoint_id} not found"})
        
        else:
            self._send_response(404, {"error": "Endpoint not found"})
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        # Add small delay to simulate real backend
        time.sleep(0.2)
        
        if path.startswith("/checkpoints/") and path.endswith("/playback/start"):
            checkpoint_id = path.split("/")[-3]
            checkpoint = self.mock_data.get_checkpoint_by_id(checkpoint_id)
            if checkpoint:
                self.mock_data.current_checkpoint = checkpoint_id
                self.mock_data.is_playing = True
                self._send_response(200, {"status": "started", "checkpoint_id": checkpoint_id})
            else:
                self._send_response(404, {"error": f"Checkpoint {checkpoint_id} not found"})
        
        elif path.startswith("/checkpoints/") and path.endswith("/playback/game"):
            checkpoint_id = path.split("/")[-3]
            checkpoint = self.mock_data.get_checkpoint_by_id(checkpoint_id)
            if checkpoint:
                game_result = self.mock_data.simulate_game_result()
                self._send_response(200, game_result)
            else:
                self._send_response(404, {"error": f"Checkpoint {checkpoint_id} not found"})
        
        elif path == "/checkpoints/playback/pause":
            self.mock_data.is_playing = False
            self._send_response(200, {"status": "paused"})
        
        elif path == "/checkpoints/playback/resume":
            self.mock_data.is_playing = True
            self._send_response(200, {"status": "resumed"})
        
        elif path == "/checkpoints/playback/stop":
            self.mock_data.is_playing = False
            self.mock_data.current_checkpoint = None
            self._send_response(200, {"status": "stopped"})
        
        elif path == "/training/start":
            self.mock_data.is_training = True
            self._send_response(200, {"status": "training_started"})
        
        elif path == "/training/stop":
            self.mock_data.is_training = False
            self._send_response(200, {"status": "training_stopped"})
        
        else:
            self._send_response(404, {"error": "Endpoint not found"})
    
    def _send_response(self, status_code: int, data: Dict[str, Any]):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        response_data = json.dumps(data, indent=2)
        self.wfile.write(response_data.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override log message to use our logger"""
        # Use print for HTTP server logs since TestLogger might not be available in this context
        print(f"Mock Backend: {format % args}")


class MockBackend:
    """Mock backend server"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.mock_data = MockData()
        self.server = None
        self.server_thread = None
        self.is_running = False
        self.logger = TestLogger()
    
    def start(self):
        """Start the mock backend server"""
        try:
            # Create a custom handler class with our mock data
            mock_data = self.mock_data
            
            class CustomHandler(MockBackendHandler):
                def __init__(self, request, client_address, server):
                    super().__init__(request, client_address, server, mock_data)
            
            self.server = HTTPServer((self.host, self.port), CustomHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            self.is_running = True
            
            self.logger.info(f"Mock Backend Server started on http://{self.host}:{self.port}")
            self.logger.info("Available endpoints:")
            self.logger.info("  GET  /")
            self.logger.info("  GET  /checkpoints")
            self.logger.info("  GET  /checkpoints/stats")
            self.logger.info("  GET  /checkpoints/{id}")
            self.logger.info("  GET  /training/status")
            self.logger.info("  GET  /checkpoints/playback/status")
            self.logger.info("  POST /checkpoints/{id}/playback/start")
            self.logger.info("  POST /checkpoints/{id}/playback/game")
            self.logger.info("  POST /checkpoints/playback/pause")
            self.logger.info("  POST /checkpoints/playback/resume")
            self.logger.info("  POST /checkpoints/playback/stop")
            
        except Exception as e:
            self.logger.error(f"Failed to start mock backend: {e}")
            raise
    
    def stop(self):
        """Stop the mock backend server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.is_running = False
            self.logger.info("Mock Backend Server stopped")
    
    def is_alive(self) -> bool:
        """Check if the server is running"""
        return self.is_running and self.server_thread and self.server_thread.is_alive()
    
    def wait_for_shutdown(self):
        """Wait for the server to shutdown"""
        if self.server_thread:
            self.server_thread.join()


def main():
    """Main function to run the mock backend server"""
    import argparse
    
    logger = TestLogger()
    logger.banner("Mock Backend Server for 2048 Bot Testing", 60)
    
    parser = argparse.ArgumentParser(description="Mock Backend Server for 2048 Bot Testing")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Create and start the mock backend
    mock_backend = MockBackend(args.host, args.port)
    
    try:
        mock_backend.start()
        
        logger.info("Mock Backend Server is running. Press Ctrl+C to stop.")
        mock_backend.wait_for_shutdown()
        
        return True
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        mock_backend.stop()
        return True
    except Exception as e:
        logger.error(f"Server error: {e}")
        mock_backend.stop()
        return False


if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1) 