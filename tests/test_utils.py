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
            'bold': '\033[1m',
            'reset': '\033[0m'
        }
        
        return colors.get(color, "")
    
    def _format_message(self, message: str, prefix: str = "", color: str = ""):
        """Format message with timestamp and color"""
        indent = "  " * self.indent_level
        
        if self.show_timestamps:
            timestamp = time.strftime("[%H:%M:%S]")
        else:
            timestamp = ""
        
        color_code = self._get_color_code(color)
        reset_code = self._get_color_code('reset')
        
        if prefix:
            return f"{color_code}{timestamp} {prefix} {indent}{message}{reset_code}"
        else:
            return f"{color_code}{timestamp} {indent}{message}{reset_code}"
    
    def info(self, message: str):
        """Log an info message"""
        print(self._format_message(message, "INFO:", "blue"))
    
    def ok(self, message: str):
        """Log a success message"""
        print(self._format_message(message, "OK:", "green"))
    
    def warning(self, message: str):
        """Log a warning message"""
        print(self._format_message(message, "WARNING:", "yellow"))
    
    def error(self, message: str):
        """Log an error message"""
        print(self._format_message(message, "ERROR:", "red"))
    
    def log(self, message: str):
        """Log a plain message"""
        print(self._format_message(message))
    
    def debug(self, message: str):
        """Log a debug message"""
        print(self._format_message(message, "DEBUG:", "cyan"))
    
    def game(self, message: str):
        """Log a game-related message"""
        print(self._format_message(message, "GAME:", "magenta"))
    
    def find(self, message: str):
        """Log a find-related message"""
        print(self._format_message(message, "FIND:", "cyan"))
    
    def status(self, message: str):
        """Log a status message"""
        print(self._format_message(message, "STATUS:", "blue"))
    
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
    """Common backend testing functionality with enhanced availability checking"""
    
    def __init__(self, base_url: str = DEFAULT_BASE_URL, logger: TestLogger = None, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url
        self.logger = logger or TestLogger()
        self.timeout = timeout
        self._connectivity_cache = None
        self._last_connectivity_check = 0
        self._cache_duration = 30  # Cache connectivity result for 30 seconds
    
    def is_backend_available(self, use_cache: bool = True) -> bool:
        """Check if backend is available with caching and retry logic"""
        current_time = time.time()
        
        # Use cached result if available and fresh
        if use_cache and self._connectivity_cache is not None:
            if current_time - self._last_connectivity_check < self._cache_duration:
                return self._connectivity_cache
        
        # Try to connect with retry logic
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.base_url}/", timeout=5)
                if response.status_code == 200:
                    self._connectivity_cache = True
                    self._last_connectivity_check = current_time
                    if attempt > 0:
                        self.logger.ok(f"Backend is accessible (attempt {attempt + 1})")
                    return True
                else:
                    self.logger.warning(f"Backend returned HTTP {response.status_code} (attempt {attempt + 1})")
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Backend connectivity timeout (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError:
                self.logger.warning(f"Backend connection failed (attempt {attempt + 1})")
            except Exception as e:
                self.logger.warning(f"Backend connectivity error: {e} (attempt {attempt + 1})")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        # All attempts failed
        self._connectivity_cache = False
        self._last_connectivity_check = current_time
        self.logger.error("Backend is not available after multiple attempts")
        return False
    
    def test_connectivity(self) -> bool:
        """Test that backend is accessible (legacy method - calls is_backend_available)"""
        return self.is_backend_available()
    
    def wait_for_backend(self, max_wait_time: int = 120, check_interval: int = 5) -> bool:
        """Wait for backend to become available"""
        self.logger.info(f"Waiting for backend to become available (max {max_wait_time}s)...")
        
        start_time = time.time()
        attempt = 0
        
        while time.time() - start_time < max_wait_time:
            attempt += 1
            if self.is_backend_available(use_cache=False):
                self.logger.ok(f"Backend became available after {time.time() - start_time:.1f}s")
                return True
            
            self.logger.info(f"Attempt {attempt}: Backend not ready, waiting {check_interval}s...")
            time.sleep(check_interval)
        
        self.logger.error(f"Backend did not become available within {max_wait_time}s")
        return False
    
    def require_backend_or_skip(self, test_name: str = "test") -> bool:
        """Check if backend is available, skip test if not"""
        if not self.is_backend_available():
            self.logger.warning(f"SKIPPED: {test_name} - Backend not available")
            self.logger.info("To run this test, ensure the backend server is running:")
            self.logger.info("  cd backend && python main.py")
            return False
        return True
    
    def get_checkpoints_with_fallback(self) -> List[Dict[str, Any]]:
        """Get list of available checkpoints with fallback for offline testing"""
        if not self.is_backend_available():
            self.logger.warning("Backend not available, using mock checkpoint data")
            return [
                {
                    "id": "mock_checkpoint_1",
                    "name": "Mock Checkpoint 1",
                    "episode": 1000,
                    "score": 5000,
                    "timestamp": "2025-01-01T00:00:00Z"
                },
                {
                    "id": "mock_checkpoint_2", 
                    "name": "Mock Checkpoint 2",
                    "episode": 2000,
                    "score": 10000,
                    "timestamp": "2025-01-02T00:00:00Z"
                }
            ]
        
        return self.get_checkpoints()
    
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
    
    def start_playback(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Start playback for a checkpoint"""
        try:
            response = requests.post(
                f"{self.base_url}/checkpoints/{checkpoint_id}/playback/start",
                timeout=self.timeout
            )
            if response.status_code != 200:
                self.logger.error(f"Failed to start playback for {checkpoint_id}: HTTP {response.status_code}")
                return None
            
            result = response.json()
            self.logger.ok(f"Playback started for checkpoint {checkpoint_id}")
            return result
        except Exception as e:
            self.logger.error(f"Error starting playback for {checkpoint_id}: {e}")
            return None
    
    def set_playback_speed(self, speed: float) -> Optional[Dict[str, Any]]:
        """Set playback speed multiplier"""
        try:
            response = requests.post(
                f"{self.base_url}/checkpoints/playback/speed",
                json={"speed": speed},
                timeout=self.timeout
            )
            if response.status_code != 200:
                self.logger.error(f"Failed to set playback speed {speed}: HTTP {response.status_code}")
                return None
            
            result = response.json()
            self.logger.ok(f"Playback speed set to {speed}x")
            return result
        except Exception as e:
            self.logger.error(f"Error setting playback speed {speed}: {e}")
            return None
    
    def get_playback_status(self) -> Optional[Dict[str, Any]]:
        """Get current playback status"""
        try:
            response = requests.get(f"{self.base_url}/checkpoints/playback/status", timeout=self.timeout)
            if response.status_code != 200:
                self.logger.error(f"Failed to get playback status: HTTP {response.status_code}")
                return None
            
            status = response.json()
            return status
        except Exception as e:
            self.logger.error(f"Error getting playback status: {e}")
            return None
    
    def get_playback_speed(self) -> Optional[Dict[str, Any]]:
        """Get current playback speed"""
        try:
            response = requests.get(f"{self.base_url}/checkpoints/playback/speed", timeout=self.timeout)
            if response.status_code != 200:
                self.logger.error(f"Failed to get playback speed: HTTP {response.status_code}")
                return None
            
            result = response.json()
            return result
        except Exception as e:
            self.logger.error(f"Error getting playback speed: {e}")
            return None
    
    def test_basic_endpoints(self) -> Dict[str, bool]:
        """Test all basic endpoints and return results"""
        results = {}
        
        self.logger.info("Testing basic backend endpoints...")
        
        # Test connectivity first
        results['connectivity'] = self.is_backend_available()
        
        if not results['connectivity']:
            self.logger.warning("Backend not available, skipping endpoint tests")
            return results
        
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


# Global backend tester instance for shared use
_global_backend_tester = None

def get_backend_tester(base_url: str = DEFAULT_BASE_URL, logger: 'TestLogger' = None) -> 'BackendTester':
    """Get a global backend tester instance"""
    global _global_backend_tester
    if _global_backend_tester is None:
        _global_backend_tester = BackendTester(base_url, logger)
    return _global_backend_tester

def requires_backend(test_name: str = None):
    """Decorator to skip tests if backend is not available"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            backend_tester = get_backend_tester()
            actual_test_name = test_name or func.__name__
            
            if not backend_tester.require_backend_or_skip(actual_test_name):
                return {"skipped": True, "reason": "Backend not available"}
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def with_backend_fallback(fallback_result=None):
    """Decorator to provide fallback results when backend is not available"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            backend_tester = get_backend_tester()
            
            if not backend_tester.is_backend_available():
                logger = backend_tester.logger
                logger.warning(f"Backend not available for {func.__name__}, using fallback")
                return fallback_result
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def check_backend_or_exit(exit_code: int = 1, wait_time: int = 30):
    """Check if backend is available or exit with error code"""
    backend_tester = get_backend_tester()
    
    if not backend_tester.is_backend_available():
        backend_tester.logger.error("Backend is not available!")
        backend_tester.logger.info("Starting backend server...")
        backend_tester.logger.info("  cd backend && python main.py")
        backend_tester.logger.info("")
        backend_tester.logger.info(f"Waiting {wait_time}s for backend to start...")
        
        if not backend_tester.wait_for_backend(wait_time):
            backend_tester.logger.error("Backend failed to start. Exiting.")
            exit(exit_code)
    
    return backend_tester 

# Add mock backend import
try:
    from tests.mock_backend import MockBackend
    MOCK_BACKEND_AVAILABLE = True
except ImportError:
    MOCK_BACKEND_AVAILABLE = False 

def start_mock_backend_if_needed(port: int = 8000, wait_time: int = 30) -> bool:
    """Start mock backend if real backend is not available"""
    if not MOCK_BACKEND_AVAILABLE:
        return False
    
    backend_tester = get_backend_tester()
    
    if backend_tester.is_backend_available():
        backend_tester.logger.info("Real backend is available, not starting mock backend")
        return True
    
    backend_tester.logger.info("Real backend not available, starting mock backend...")
    
    try:
        mock_backend = MockBackend("localhost", port)
        mock_backend.start()
        
        # Wait for mock backend to be ready
        import time
        time.sleep(2)
        
        # Check if mock backend is working
        if mock_backend.is_alive():
            backend_tester.logger.ok(f"Mock backend started on port {port}")
            return True
        else:
            backend_tester.logger.error("Mock backend failed to start")
            return False
            
    except Exception as e:
        backend_tester.logger.error(f"Failed to start mock backend: {e}")
        return False

def check_backend_or_start_mock(port: int = 8000, wait_time: int = 30) -> bool:
    """Check if backend is available or start mock backend"""
    backend_tester = get_backend_tester()
    
    if backend_tester.is_backend_available():
        backend_tester.logger.ok("Real backend is available")
        return True
    
    backend_tester.logger.warning("Real backend not available, attempting to start mock backend...")
    
    if start_mock_backend_if_needed(port, wait_time):
        backend_tester.logger.ok("Mock backend started successfully")
        return True
    else:
        backend_tester.logger.error("Neither real nor mock backend is available")
        return False 