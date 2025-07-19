#!/usr/bin/env python3
"""
Mock Backend Server
==================

This module provides a mock backend server for testing purposes.
It simulates the real backend API endpoints and responses.

Usage:
    from mock_backend import MockBackend
    
    backend = MockBackend()
    backend.start()
    # ... test code ...
    backend.stop()
"""

import json
import time
import threading
import random
from typing import Dict, Any, List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from tests.utilities.test_utils import TestLogger

# Create a logger instance for the mock backend
logger = TestLogger()

class MockData:
    """Mock data generator for the backend"""
    
    def __init__(self):
        self.checkpoints = self._generate_mock_checkpoints()
        self.training_status = self._generate_mock_training_status()
        self.playback_status = self._generate_mock_playback_status()
    
    def _generate_mock_checkpoints(self) -> List[Dict[str, Any]]:
        """Generate mock checkpoint data to match real backend structure"""
        checkpoints = []
        for i in range(5):
            checkpoint = {
                "id": f"mock_checkpoint_{i+1}",
                "nickname": f"Mock Checkpoint {i+1}",
                "episode": (i + 1) * 500,
                "created_at": f"2025-01-{i+1:02d}T00:00:00Z",
                "training_duration": (i + 1) * 3600.0,  # hours in seconds
                "model_config": {
                    "model_size": "medium",
                    "learning_rate": 0.0003,
                    "n_experts": 6,
                    "n_layers": 6,
                    "d_model": 384,
                    "n_heads": 8
                },
                "performance_metrics": {
                    "best_score": (i + 1) * 1000,
                    "avg_score": (i + 1) * 800,
                    "final_loss": 0.15 + i * 0.02,
                    "training_speed": 95.0 + i * 2.0
                },
                "file_size": 1024 * 1024 * (i + 1),  # 1MB, 2MB, etc.
                "parent_checkpoint": None,
                "tags": ["mock", "test"],
                "absolute_path": f"/mock/path/checkpoint_{i+1}.pt"
            }
            checkpoints.append(checkpoint)
        return checkpoints
    
    def _generate_mock_training_status(self) -> Dict[str, Any]:
        """Generate mock training status"""
        return {
            "is_training": False,
            "current_episode": 0,
            "total_episodes": 1000,
            "current_score": 0,
            "best_score": 2048,
            "training_time": 0,
            "model_size": "medium"
        }
    
    def _generate_mock_playback_status(self) -> Dict[str, Any]:
        """Generate mock playback status"""
        return {
            "is_playing": False,
            "current_step": 0,
            "total_steps": 0,
            "current_score": 0,
            "playback_speed": 1.0,
            "checkpoint_id": None
        }
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics to match real backend structure"""
        return {
            "total_checkpoints": len(self.checkpoints),
            "total_size": sum(cp["file_size"] for cp in self.checkpoints),
            "best_score": max(cp["performance_metrics"]["best_score"] for cp in self.checkpoints) if self.checkpoints else 0,
            "latest_episode": max(cp["episode"] for cp in self.checkpoints) if self.checkpoints else 0,
            "total_training_time": sum(cp["training_duration"] for cp in self.checkpoints)
        }
    
    def get_checkpoint_by_id(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific checkpoint by ID"""
        for checkpoint in self.checkpoints:
            if checkpoint["id"] == checkpoint_id:
                return checkpoint
        return None
    
    def simulate_game_result(self) -> Dict[str, Any]:
        """Simulate a game result with the exact structure expected by GameTester"""
        steps = random.randint(50, 200)
        score = random.randint(100, 2048)
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
            "success": True,
            "final_score": score,
            "steps": steps,
            "completed": True,
            "performance_ok": True,
            "steps_per_second": random.uniform(0.5, 2.0),
            "game_history": game_history,
            "max_tile": max_tile
        }

class MockBackendHandler(BaseHTTPRequestHandler):
    """HTTP request handler for mock backend"""
    
    def __init__(self, request, client_address, server, mock_data: Optional[MockData] = None):
        self.mock_data = mock_data or MockData()
        super().__init__(request, client_address, server)
    
    def _send_response(self, status_code: int, data: Any):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        response = json.dumps(data, indent=2)
        self.wfile.write(response.encode('utf-8'))
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == "/":
                self._send_response(200, {"status": "ok", "message": "Mock Backend Server"})
            elif self.path == "/checkpoints":
                self._send_response(200, self.mock_data.checkpoints)
            elif self.path == "/checkpoints/stats":
                self._send_response(200, self.mock_data.get_checkpoint_stats())
            elif self.path == "/training/status":
                self._send_response(200, self.mock_data._generate_mock_training_status())
            elif self.path == "/checkpoints/playback/status":
                self._send_response(200, self.mock_data._generate_mock_playback_status())
            elif self.path.startswith("/checkpoints/"):
                # Extract checkpoint ID from path
                checkpoint_id = self.path.split("/")[-1]
                checkpoint = self.mock_data.get_checkpoint_by_id(checkpoint_id)
                if checkpoint:
                    self._send_response(200, checkpoint)
                else:
                    self._send_response(404, {"error": f"Checkpoint {checkpoint_id} not found"})
            else:
                self._send_response(404, {"error": "Endpoint not found"})
        except Exception as e:
            self._send_response(500, {"error": str(e)})
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            if self.path == "/checkpoints/playback/start":
                # Simulate starting playback
                self._send_response(200, {"success": True, "message": "Playback started"})
            elif self.path.startswith("/checkpoints/") and self.path.endswith("/playback/start"):
                # Simulate starting playback for specific checkpoint
                self._send_response(200, {"success": True, "message": "Playback started"})
            elif self.path == "/checkpoints/playback/stop":
                # Simulate stopping playback
                self._send_response(200, {"success": True, "message": "Playback stopped"})
            elif self.path == "/checkpoints/playback/pause":
                # Simulate pausing playback
                self._send_response(200, {"success": True, "message": "Playback paused"})
            elif self.path == "/checkpoints/playback/resume":
                # Simulate resuming playback
                self._send_response(200, {"success": True, "message": "Playback resumed"})
            elif self.path.endswith("/playback/game"):
                # Simulate game playback
                game_result = self.mock_data.simulate_game_result()
                self._send_response(200, game_result)
            else:
                self._send_response(404, {"error": "Endpoint not found"})
        except Exception as e:
            self._send_response(500, {"error": str(e)})
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"Mock Backend: {format % args}")

class MockBackend:
    """Mock backend server"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.is_running = False
        
    def start(self):
        """Start the mock backend server"""
        try:
            # Create server with custom handler
            class CustomHandler(MockBackendHandler):
                def __init__(self, request, client_address, server):
                    mock_data = MockData()
                    super().__init__(request, client_address, server, mock_data)
            
            # Create threaded server
            class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
                pass
            
            self.server = ThreadedHTTPServer((self.host, self.port), CustomHandler)
            self.is_running = True
            
            # Start server in background thread
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            
            logger.info(f"Mock backend started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start mock backend: {e}")
            return False
    
    def stop(self):
        """Stop the mock backend server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.is_running = False
            logger.info("Mock backend stopped")
    
    def is_alive(self) -> bool:
        """Check if the server is running"""
        return self.is_running and self.server is not None
    
    def wait_for_shutdown(self):
        """Wait for the server to shutdown"""
        if self.server_thread:
            self.server_thread.join()

def main():
    """Main function for testing"""
    logger.info("Starting Mock Backend Server")
    
    backend = MockBackend()
    if backend.start():
        try:
            logger.info("Mock backend is running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping mock backend...")
        finally:
            backend.stop()
    else:
        logger.error("Failed to start mock backend")

if __name__ == "__main__":
    main() 