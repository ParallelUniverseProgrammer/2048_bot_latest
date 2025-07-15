#!/usr/bin/env python3
"""
Shared Test Utilities for Checkpoint System Tests
================================================

This module provides common functionality for all checkpoint system tests:
- Standardized logging with consistent prefixes
- Common backend connectivity checks
- Shared API endpoint testing
- Consistent error handling and reporting

Usage:
    from test_utils import TestLogger, BackendTester
    
    logger = TestLogger()
    backend = BackendTester("http://localhost:8000", logger)
    
    if backend.test_connectivity():
        checkpoints = backend.get_checkpoints()
"""

import requests
import time
import json
from typing import Dict, Any, List, Optional, Tuple

# Configuration
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 30

class TestLogger:
    """Standardized logging for all test scripts with enhanced formatting"""
    
    def __init__(self, show_timestamps: bool = True, use_colors: bool = True):
        self.show_timestamps = show_timestamps
        self.use_colors = use_colors
        self.indent_level = 0
    
    def _get_color_code(self, color: str) -> str:
        """Get ANSI color code if colors are enabled"""
        if not self.use_colors:
            return ""
        
        colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'bold': '\033[1m',
            'reset': '\033[0m'
        }
        return colors.get(color, '')
    
    def _format_message(self, message: str, prefix: str = "", color: str = "") -> str:
        """Format message with optional timestamp, prefix, and color"""
        indent = "  " * self.indent_level
        
        if prefix:
            formatted_prefix = f"{self._get_color_code('bold')}{prefix}{self._get_color_code('reset')}"
            message = f"{formatted_prefix} {message}"
        
        if color:
            message = f"{self._get_color_code(color)}{message}{self._get_color_code('reset')}"
        
        if self.show_timestamps and message.strip() and not message.startswith("="):
            timestamp = time.strftime("%H:%M:%S")
            return f"{indent}[{timestamp}] {message}"
        
        return f"{indent}{message}"
    
    def log(self, message: str = ""):
        """Log a message with standardized formatting"""
        print(self._format_message(message))
    
    def ok(self, message: str):
        """Log a success message"""
        print(self._format_message(message, "OK:", "green"))
    
    def error(self, message: str):
        """Log an error message"""
        print(self._format_message(message, "ERROR:", "red"))
    
    def warning(self, message: str):
        """Log a warning message"""
        print(self._format_message(message, "WARNING:", "yellow"))
    
    def info(self, message: str):
        """Log an info message"""
        print(self._format_message(message, "INFO:", "blue"))
    
    def game(self, message: str):
        """Log a game-related message"""
        print(self._format_message(message, "GAME:", "magenta"))
    
    def starting(self, message: str):
        """Log a starting message"""
        print(self._format_message(message, "STARTING:", "cyan"))
    
    def running(self, message: str):
        """Log a running message"""
        print(self._format_message(message, "RUNNING:", "blue"))
    
    def controls(self, message: str):
        """Log a controls-related message"""
        print(self._format_message(message, "CONTROLS:", "magenta"))
    
    def playback(self, message: str):
        """Log a playback-related message"""
        print(self._format_message(message, "PLAYBACK:", "cyan"))
    
    def testing(self, message: str):
        """Log a testing message"""
        print(self._format_message(message, "TESTING:", "blue"))
    
    def success(self, message: str):
        """Log a success message"""
        print(self._format_message(message, "SUCCESS:", "green"))
    
    def banner(self, message: str, width: int = 60):
        """Print a banner with consistent formatting"""
        print(self._format_message(f"{self._get_color_code('bold')}{message}{self._get_color_code('reset')}"))
        print(self._format_message("=" * width))
    
    def separator(self, width: int = 60):
        """Print a separator line"""
        print(self._format_message("=" * width))
    
    def section(self, message: str, width: int = 60):
        """Print a section header"""
        print(self._format_message(""))
        print(self._format_message(f"{self._get_color_code('bold')}{message}{self._get_color_code('reset')}"))
        print(self._format_message("-" * len(message)))
    
    def indent(self):
        """Increase indentation level"""
        self.indent_level += 1
    
    def dedent(self):
        """Decrease indentation level"""
        self.indent_level = max(0, self.indent_level - 1)
    
    def progress(self, current: int, total: int, message: str = ""):
        """Show progress information"""
        percentage = (current / total) * 100 if total > 0 else 0
        progress_bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
        progress_msg = f"[{progress_bar}] {percentage:.1f}% ({current}/{total})"
        
        if message:
            progress_msg += f" {message}"
        
        print(self._format_message(progress_msg, "PROGRESS:", "cyan"))
    
    def table_header(self, headers: List[str], widths: List[int] = None):
        """Print a table header"""
        if not widths:
            widths = [max(15, len(h)) for h in headers]
        
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        separator_line = "-+-".join("-" * w for w in widths)
        
        print(self._format_message(f"{self._get_color_code('bold')}{header_line}{self._get_color_code('reset')}"))
        print(self._format_message(separator_line))
    
    def table_row(self, values: List[str], widths: List[int] = None):
        """Print a table row"""
        if not widths:
            widths = [max(15, len(str(v))) for v in values]
        
        row_line = " | ".join(str(v).ljust(w) for v, w in zip(values, widths))
        print(self._format_message(row_line))
    
    def summary_box(self, title: str, items: List[tuple], width: int = 60):
        """Print a summary box with title and items"""
        print(self._format_message(""))
        print(self._format_message("┌" + "─" * (width - 2) + "┐"))
        
        # Title
        title_line = f"│ {title.center(width - 4)} │"
        print(self._format_message(f"{self._get_color_code('bold')}{title_line}{self._get_color_code('reset')}"))
        print(self._format_message("├" + "─" * (width - 2) + "┤"))
        
        # Items
        for label, value in items:
            item_line = f"│ {label}: {value}".ljust(width - 1) + "│"
            print(self._format_message(item_line))
        
        print(self._format_message("└" + "─" * (width - 2) + "┘"))
    
    def step(self, step_num: int, total_steps: int, message: str):
        """Log a step in a multi-step process"""
        step_msg = f"Step {step_num}/{total_steps}: {message}"
        print(self._format_message(step_msg, "STEP:", "cyan"))

class BackendTester:
    """Common backend testing functionality"""
    
    def __init__(self, base_url: str = DEFAULT_BASE_URL, logger: TestLogger = None, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url
        self.logger = logger or TestLogger()
        self.timeout = timeout
    
    def test_connectivity(self) -> bool:
        """Test that backend is accessible"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                self.logger.ok("Backend is accessible")
                return True
            else:
                self.logger.error(f"Backend returned HTTP {response.status_code}")
                return False
        except requests.exceptions.Timeout:
            self.logger.error("Backend connectivity timeout")
            return False
        except requests.exceptions.ConnectionError:
            self.logger.error("Backend connection failed - server may not be running")
            return False
        except Exception as e:
            self.logger.error(f"Backend connectivity failed: {e}")
            return False
    
    def get_checkpoints(self) -> List[Dict[str, Any]]:
        """Get list of available checkpoints"""
        try:
            response = requests.get(f"{self.base_url}/checkpoints", timeout=self.timeout)
            if response.status_code != 200:
                self.logger.error(f"Failed to get checkpoints: HTTP {response.status_code}")
                return []
            
            checkpoints = response.json()
            self.logger.ok(f"Found {len(checkpoints)} checkpoints")
            return checkpoints
        except Exception as e:
            self.logger.error(f"Error getting checkpoints: {e}")
            return []
    
    def get_checkpoint_stats(self) -> Optional[Dict[str, Any]]:
        """Get checkpoint statistics"""
        try:
            response = requests.get(f"{self.base_url}/checkpoints/stats", timeout=self.timeout)
            if response.status_code != 200:
                self.logger.error(f"Failed to get checkpoint stats: HTTP {response.status_code}")
                return None
            
            stats = response.json()
            self.logger.ok("Checkpoint stats retrieved successfully")
            return stats
        except Exception as e:
            self.logger.error(f"Error getting checkpoint stats: {e}")
            return None
    
    def get_training_status(self) -> Optional[Dict[str, Any]]:
        """Get training status"""
        try:
            response = requests.get(f"{self.base_url}/training/status", timeout=self.timeout)
            if response.status_code != 200:
                self.logger.error(f"Failed to get training status: HTTP {response.status_code}")
                return None
            
            status = response.json()
            self.logger.ok("Training status retrieved successfully")
            return status
        except Exception as e:
            self.logger.error(f"Error getting training status: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific checkpoint"""
        try:
            response = requests.get(f"{self.base_url}/checkpoints/{checkpoint_id}", timeout=self.timeout)
            if response.status_code != 200:
                self.logger.error(f"Failed to load checkpoint {checkpoint_id}: HTTP {response.status_code}")
                return None
            
            checkpoint_info = response.json()
            self.logger.ok(f"Checkpoint {checkpoint_id} loaded successfully")
            return checkpoint_info
        except Exception as e:
            self.logger.error(f"Error loading checkpoint {checkpoint_id}: {e}")
            return None
    
    def test_basic_endpoints(self) -> Dict[str, bool]:
        """Test all basic endpoints and return results"""
        results = {}
        
        self.logger.info("Testing basic backend endpoints...")
        
        # Test connectivity
        results['connectivity'] = self.test_connectivity()
        
        # Test checkpoints endpoint
        checkpoints = self.get_checkpoints()
        results['checkpoints'] = len(checkpoints) > 0
        
        # Test stats endpoint
        stats = self.get_checkpoint_stats()
        results['stats'] = stats is not None
        
        # Test training status
        training_status = self.get_training_status()
        results['training_status'] = training_status is not None
        
        # Summary
        passed = sum(results.values())
        total = len(results)
        
        if passed == total:
            self.logger.ok(f"All {total} basic endpoints working correctly")
        else:
            self.logger.warning(f"Only {passed}/{total} basic endpoints working correctly")
        
        return results

class GameTester:
    """Common game testing functionality"""
    
    def __init__(self, base_url: str = DEFAULT_BASE_URL, logger: TestLogger = None, timeout: int = 120):
        self.base_url = base_url
        self.logger = logger or TestLogger()
        self.timeout = timeout
    
    def test_single_game_playback(self, checkpoint_id: str) -> Dict[str, Any]:
        """Test playing a complete game from a checkpoint"""
        self.logger.game(f"Testing single game playback for checkpoint {checkpoint_id}")
        
        start_time = time.time()
        
        try:
            # Start single game playback
            response = requests.post(
                f"{self.base_url}/checkpoints/{checkpoint_id}/playback/game",
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Game playback failed: HTTP {response.status_code}",
                    "response_text": response.text
                }
            
            game_result = response.json()
            total_time = time.time() - start_time
            
            # Validate game result structure
            required_fields = ['game_history', 'final_score', 'max_tile', 'steps', 'completed']
            missing_fields = [field for field in required_fields if field not in game_result]
            
            if missing_fields:
                return {
                    "success": False,
                    "error": f"Game result missing required fields: {missing_fields}",
                    "game_result": game_result
                }
            
            # Validate game history
            game_history = game_result['game_history']
            if not game_history:
                return {
                    "success": False,
                    "error": "Game history is empty",
                    "game_result": game_result
                }
            
            # Check game completion
            if not game_result['completed']:
                return {
                    "success": False,
                    "error": "Game did not complete properly",
                    "game_result": game_result
                }
            
            # Performance metrics
            steps_per_second = game_result['steps'] / total_time if total_time > 0 else 0
            
            result = {
                "success": True,
                "checkpoint_id": checkpoint_id,
                "final_score": game_result['final_score'],
                "max_tile": game_result['max_tile'],
                "steps": game_result['steps'],
                "history_length": len(game_history),
                "total_time": total_time,
                "steps_per_second": steps_per_second,
                "performance_ok": steps_per_second > 0.5,  # Should process at least 0.5 steps per second
                "completed": game_result['completed']
            }
            
            self.logger.ok("Game completed successfully!")
            self.logger.log(f"   Final Score: {result['final_score']}")
            self.logger.log(f"   Max Tile: {result['max_tile']}")
            self.logger.log(f"   Steps: {result['steps']}")
            self.logger.log(f"   Time: {total_time:.2f}s")
            self.logger.log(f"   Speed: {steps_per_second:.2f} steps/sec")
            
            return result
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"Game playback timed out after {self.timeout} seconds",
                "checkpoint_id": checkpoint_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error during game playback: {e}",
                "checkpoint_id": checkpoint_id
            }

class PlaybackTester:
    """Common playback testing functionality"""
    
    def __init__(self, base_url: str = DEFAULT_BASE_URL, logger: TestLogger = None, timeout: int = 30):
        self.base_url = base_url
        self.logger = logger or TestLogger()
        self.timeout = timeout
    
    def start_live_playback(self, checkpoint_id: str) -> Dict[str, Any]:
        """Start live playback for a checkpoint"""
        self.logger.playback(f"Testing live playback start for checkpoint {checkpoint_id}")
        
        try:
            # Start live playback
            response = requests.post(
                f"{self.base_url}/checkpoints/{checkpoint_id}/playback/start",
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Live playback start failed: HTTP {response.status_code}",
                    "response_text": response.text
                }
            
            start_result = response.json()
            
            # Wait a moment for playback to initialize
            time.sleep(2)
            
            # Check playback status
            status_response = requests.get(f"{self.base_url}/checkpoints/playback/status", timeout=10)
            if status_response.status_code != 200:
                return {
                    "success": False,
                    "error": "Failed to get playback status after start"
                }
            
            status = status_response.json()
            
            result = {
                "success": True,
                "checkpoint_id": checkpoint_id,
                "is_playing": status.get('is_playing', False),
                "model_loaded": status.get('model_loaded', False),
                "current_checkpoint": status.get('current_checkpoint'),
                "start_result": start_result
            }
            
            self.logger.ok("Live playback started successfully!")
            self.logger.log(f"   Playing: {result['is_playing']}")
            self.logger.log(f"   Model Loaded: {result['model_loaded']}")
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error starting live playback: {e}",
                "checkpoint_id": checkpoint_id
            }
    
    def test_playback_controls(self, checkpoint_id: str) -> Dict[str, Any]:
        """Test playback pause/resume/stop controls"""
        self.logger.controls("Testing playback controls")
        
        try:
            # Start playback first
            start_result = self.start_live_playback(checkpoint_id)
            if not start_result["success"]:
                return {"success": False, "error": "Failed to start playback for control testing"}
            
            time.sleep(1)
            
            # Test pause
            pause_response = requests.post(f"{self.base_url}/checkpoints/playback/pause", timeout=10)
            pause_ok = pause_response.status_code == 200
            
            time.sleep(1)
            
            # Test resume
            resume_response = requests.post(f"{self.base_url}/checkpoints/playback/resume", timeout=10)
            resume_ok = resume_response.status_code == 200
            
            time.sleep(1)
            
            # Test stop
            stop_response = requests.post(f"{self.base_url}/checkpoints/playback/stop", timeout=10)
            stop_ok = stop_response.status_code == 200
            
            result = {
                "success": all([pause_ok, resume_ok, stop_ok]),
                "pause_ok": pause_ok,
                "resume_ok": resume_ok,
                "stop_ok": stop_ok
            }
            
            if result["success"]:
                self.logger.ok("Playback controls working correctly")
            else:
                self.logger.error("Some playback controls failed")
                if not pause_ok:
                    self.logger.log("   Pause failed")
                if not resume_ok:
                    self.logger.log("   Resume failed")
                if not stop_ok:
                    self.logger.log("   Stop failed")
            
            return result
            
        except Exception as e:
            return {"success": False, "error": f"Error testing playback controls: {e}"} 