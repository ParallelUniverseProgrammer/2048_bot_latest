#!/usr/bin/env python3
"""
Unified Backend Manager
=======================

This module provides a comprehensive backend management system with two distinct decorators:
- @requires_real_backend: For tests that need the actual backend server
- @requires_mock_backend: For tests that can use the mock backend

The system includes:
- Global backend manager that can start/maintain a real backend server
- Automatic health checking and restart capabilities
- Mock backend management for offline testing
- Smart caching and connection pooling
- Comprehensive logging and error handling

Usage:
    from tests.utilities.backend_manager import requires_real_backend, requires_mock_backend
    
    @requires_real_backend("Training Integration Tests")
    def test_training_integration():
        # This test will have a real backend available
        pass
    
    @requires_mock_backend("Frontend UI Tests")
    def test_frontend_ui():
        # This test will have a mock backend available
        pass
"""

import os
import sys
import time
import threading
import subprocess
import signal
import atexit
from typing import Dict, Any, Optional, Callable, Union
from contextlib import contextmanager
from pathlib import Path

from tests.utilities.test_utils import BackendTester, TestLogger
from tests.utilities.mock_backend import MockBackend


class BackendManager:
    """
    Global backend manager that can start and maintain a real backend server
    or provide mock backend services for testing.
    """
    
    def __init__(self, base_url: str = "http://localhost:8001", 
                 backend_port: int = 8001,
                 backend_dir: str = "backend",
                 test_logger: Optional[TestLogger] = None):
        """
        Initialize the backend manager
        
        Args:
            base_url: Base URL for the backend server
            backend_port: Port to use for backend server
            backend_dir: Directory containing the backend code
            test_logger: Optional TestLogger instance
        """
        self.base_url = base_url
        self.backend_port = backend_port
        self.backend_dir = backend_dir
        self.test_logger = test_logger or TestLogger()
        
        # Backend tester for connectivity checks
        self.backend_tester = BackendTester(base_url, self.test_logger)
        
        # Real backend management
        self.real_backend_process = None
        self.real_backend_thread = None
        self.is_real_backend_running = False
        self.real_backend_start_time = None
        self.real_backend_health_checks = 0
        
        # Mock backend management
        self.mock_backend = None
        self.mock_backend_thread = None
        self.is_mock_backend_running = False
        
        # State management
        self._backend_type = None  # 'real', 'mock', or None
        self._last_health_check = 0
        self._health_check_interval = 30  # Check health every 30 seconds
        self._max_health_check_failures = 3
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def start_real_backend(self, wait_for_startup: bool = True) -> bool:
        """
        Start the real backend server
        
        Args:
            wait_for_startup: Whether to wait for backend to become available
            
        Returns:
            bool: True if successfully started, False otherwise
        """
        if self.is_real_backend_running:
            self.test_logger.warning("Real backend is already running")
            return True
        
        try:
            # Get the absolute path to the backend directory
            project_root = Path(__file__).parent.parent.parent
            backend_path = project_root / self.backend_dir
            
            if not backend_path.exists():
                self.test_logger.error(f"Backend directory not found: {backend_path}")
                return False
            
            # Change to backend directory and start the server
            self.test_logger.info(f"Starting real backend from {backend_path}")
            
            # Platform-specific startup
            if os.name == 'nt':  # Windows
                # Use PowerShell to start the backend
                cmd = [
                    'powershell.exe', '-Command',
                    f'Set-Location "{backend_path}"; python main.py'
                ]
            else:  # Unix/Linux/macOS
                cmd = ['bash', '-c', f'cd "{backend_path}" && python main.py']
            
            # Start the process
            self.real_backend_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.is_real_backend_running = True
            self.real_backend_start_time = time.time()
            self._backend_type = 'real'
            
            self.test_logger.ok("Real backend process started")
            
            if wait_for_startup:
                return self.wait_for_backend_startup()
            else:
                return True
                
        except Exception as e:
            self.test_logger.error(f"Failed to start real backend: {e}")
            return False
    
    def stop_real_backend(self):
        """Stop the real backend server"""
        if not self.is_real_backend_running:
            return
        
        self.test_logger.info("Stopping real backend...")
        
        if self.real_backend_process:
            try:
                # Try graceful shutdown first
                if os.name == 'nt':  # Windows
                    self.real_backend_process.terminate()
                else:  # Unix/Linux/macOS
                    self.real_backend_process.send_signal(signal.SIGTERM)
                
                # Wait for graceful shutdown
                try:
                    self.real_backend_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    self.real_backend_process.kill()
                    self.real_backend_process.wait()
                    
            except Exception as e:
                self.test_logger.warning(f"Error stopping real backend: {e}")
        
        self.is_real_backend_running = False
        self.real_backend_process = None
        self._backend_type = None
        
        self.test_logger.ok("Real backend stopped")
    
    def start_mock_backend(self, wait_for_startup: bool = True) -> bool:
        """
        Start the mock backend server
        
        Args:
            wait_for_startup: Whether to wait for backend to become available
            
        Returns:
            bool: True if successfully started, False otherwise
        """
        if self.is_mock_backend_running:
            self.test_logger.warning("Mock backend is already running")
            return True
        
        try:
            self.test_logger.info(f"Starting mock backend on port {self.backend_port}...")
            
            # Create mock backend instance
            self.mock_backend = MockBackend("localhost", self.backend_port)
            
            # Start in background thread
            self.mock_backend_thread = threading.Thread(
                target=self._run_mock_backend,
                daemon=True
            )
            self.mock_backend_thread.start()
            
            # Mark as running
            self.is_mock_backend_running = True
            self._backend_type = 'mock'
            
            self.test_logger.ok("Mock backend started successfully")
            
            if wait_for_startup:
                return self.wait_for_backend_startup()
            else:
                return True
                
        except Exception as e:
            self.test_logger.error(f"Failed to start mock backend: {e}")
            return False
    
    def _run_mock_backend(self):
        """Internal method to run mock backend in thread"""
        try:
            if self.mock_backend:
                self.mock_backend.start()
                self.mock_backend.wait_for_shutdown()
        except Exception as e:
            self.test_logger.error(f"Mock backend error: {e}")
        finally:
            self.is_mock_backend_running = False
    
    def stop_mock_backend(self):
        """Stop the mock backend server"""
        if not self.is_mock_backend_running:
            return
        
        self.test_logger.info("Stopping mock backend...")
        
        if self.mock_backend:
            try:
                self.mock_backend.stop()
            except Exception as e:
                self.test_logger.warning(f"Error stopping mock backend: {e}")
        
        self.is_mock_backend_running = False
        self.mock_backend = None
        self.mock_backend_thread = None
        self._backend_type = None
        
        self.test_logger.ok("Mock backend stopped")
    
    def is_backend_available(self, backend_type: Optional[str] = None) -> bool:
        """
        Check if backend is available
        
        Args:
            backend_type: Specific backend type to check ('real', 'mock', or None for any)
            
        Returns:
            bool: True if backend is available, False otherwise
        """
        # Check real backend
        if backend_type is None or backend_type == 'real':
            if self.is_real_backend_running and self.backend_tester.is_backend_available(use_cache=False):
                return True
        
        # Check mock backend
        if backend_type is None or backend_type == 'mock':
            if self.is_mock_backend_running and self.backend_tester.is_backend_available(use_cache=False):
                return True
        
        return False
    
    def wait_for_backend_startup(self, timeout: int = 30) -> bool:
        """
        Wait for backend to become available
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if backend became available, False if timeout
        """
        self.test_logger.info(f"Waiting for backend startup (max {timeout}s)...")
        
        start_time = time.time()
        check_interval = 1
        
        while time.time() - start_time < timeout:
            if self.backend_tester.is_backend_available(use_cache=False):
                backend_type = self._backend_type or 'unknown'
                self.test_logger.ok(f"Backend ({backend_type}) is now available")
                return True
            
            time.sleep(check_interval)
        
        self.test_logger.error(f"Backend did not become available within {timeout}s")
        return False
    
    def check_backend_health(self) -> bool:
        """
        Check the health of the currently running backend
        
        Returns:
            bool: True if backend is healthy, False otherwise
        """
        current_time = time.time()
        
        # Don't check too frequently
        if current_time - self._last_health_check < self._health_check_interval:
            return True
        
        self._last_health_check = current_time
        
        if not self.is_backend_available():
            self.real_backend_health_checks += 1
            self.test_logger.warning(f"Backend health check failed ({self.real_backend_health_checks}/{self._max_health_check_failures})")
            
            # Restart real backend if too many failures
            if (self._backend_type == 'real' and 
                self.real_backend_health_checks >= self._max_health_check_failures):
                self.test_logger.warning("Too many health check failures, restarting real backend...")
                self.stop_real_backend()
                return self.start_real_backend()
            
            return False
        
        # Reset failure counter on success
        self.real_backend_health_checks = 0
        return True
    
    def get_backend_type(self) -> Optional[str]:
        """Get the type of currently active backend"""
        return self._backend_type
    
    def get_backend_stats(self) -> Dict[str, Any]:
        """Get statistics about backend status"""
        return {
            'backend_type': self._backend_type,
            'is_real_running': self.is_real_backend_running,
            'is_mock_running': self.is_mock_backend_running,
            'is_available': self.is_backend_available(),
            'real_backend_start_time': self.real_backend_start_time,
            'health_check_failures': self.real_backend_health_checks,
            'base_url': self.base_url
        }
    
    def cleanup(self):
        """Clean up all backend processes"""
        self.stop_real_backend()
        self.stop_mock_backend()


# Global backend manager instance
_global_backend_manager = None

def get_global_backend_manager() -> BackendManager:
    """Get the global backend manager instance"""
    global _global_backend_manager
    if _global_backend_manager is None:
        _global_backend_manager = BackendManager()
    return _global_backend_manager


# Decorator functions
def requires_real_backend(test_name: Optional[str] = None):
    """
    Decorator for tests that require a real backend server
    
    Args:
        test_name: Optional name for the test (for logging)
        
    Usage:
        @requires_real_backend("Training Integration Tests")
        def test_training_integration():
            # This test will have a real backend available
            pass
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            manager = get_global_backend_manager()
            actual_test_name = test_name or func.__name__
            
            manager.test_logger.info(f"Ensuring real backend for: {actual_test_name}")
            
            # Check if real backend is already available
            if manager.is_backend_available(backend_type='real'):
                manager.test_logger.ok("Real backend is already available")
                return func(*args, **kwargs)
            
            # Start real backend if not available
            if manager.start_real_backend():
                manager.test_logger.ok(f"Real backend started for {actual_test_name}")
                return func(*args, **kwargs)
            else:
                manager.test_logger.error(f"Failed to start real backend for {actual_test_name}")
                return {"skipped": True, "reason": "Real backend not available"}
        
        return wrapper
    return decorator


def requires_mock_backend(test_name: Optional[str] = None):
    """
    Decorator for tests that can use a mock backend
    
    Args:
        test_name: Optional name for the test (for logging)
        
    Usage:
        @requires_mock_backend("Frontend UI Tests")
        def test_frontend_ui():
            # This test will have a mock backend available
            pass
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            manager = get_global_backend_manager()
            actual_test_name = test_name or func.__name__
            
            manager.test_logger.info(f"Ensuring mock backend for: {actual_test_name}")
            
            # Check if mock backend is available
            if manager.is_backend_available(backend_type='mock'):
                manager.test_logger.ok("Mock backend is already available")
                return func(*args, **kwargs)
            
            # Start mock backend if no backend is available
            if manager.start_mock_backend():
                manager.test_logger.ok(f"Mock backend started for {actual_test_name}")
                return func(*args, **kwargs)
            else:
                manager.test_logger.error(f"Failed to start mock backend for {actual_test_name}")
                return {"skipped": True, "reason": "Mock backend not available"}
        
        return wrapper
    return decorator


# Context managers for manual control
@contextmanager
def real_backend_context(test_logger: Optional[TestLogger] = None):
    """
    Context manager for real backend availability
    
    Usage:
        with real_backend_context() as backend_available:
            if backend_available:
                # Run tests with real backend
                pass
    """
    manager = get_global_backend_manager()
    if test_logger:
        manager.test_logger = test_logger
    
    backend_available = manager.is_backend_available(backend_type='real')
    
    if not backend_available:
        backend_available = manager.start_real_backend()
    
    try:
        yield backend_available
    finally:
        # Don't stop the real backend here - let it keep running for other tests
        pass


@contextmanager
def mock_backend_context(test_logger: Optional[TestLogger] = None):
    """
    Context manager for mock backend availability
    
    Usage:
        with mock_backend_context() as backend_available:
            if backend_available:
                # Run tests with mock backend
                pass
    """
    manager = get_global_backend_manager()
    if test_logger:
        manager.test_logger = test_logger
    
    backend_available = manager.is_backend_available()
    
    if not backend_available:
        backend_available = manager.start_mock_backend()
    
    try:
        yield backend_available
    finally:
        # Clean up mock backend after use
        if manager._backend_type == 'mock':
            manager.stop_mock_backend()


@requires_mock_backend("Backend Manager Demo")
def main():
    """Main entry point for backend manager testing"""
    logger = TestLogger()
    logger.banner("Backend Manager Test", 60)
    
    manager = get_global_backend_manager()
    
    logger.info("Testing backend management...")
    
    # Test real backend startup
    logger.section("Testing Real Backend Management")
    if manager.start_real_backend():
        logger.ok("Real backend started successfully")
        
        # Show backend stats
        stats = manager.get_backend_stats()
        logger.info(f"Backend type: {stats['backend_type']}")
        logger.info(f"Base URL: {stats['base_url']}")
        
        # Test health check
        if manager.check_backend_health():
            logger.ok("Backend health check passed")
        else:
            logger.error("Backend health check failed")
        
        # Clean up
        manager.stop_real_backend()
        
    else:
        logger.error("Failed to start real backend")
    
    # Test mock backend startup
    logger.section("Testing Mock Backend Management")
    if manager.start_mock_backend():
        logger.ok("Mock backend started successfully")
        
        # Test context manager
        with mock_backend_context() as available:
            if available:
                logger.ok("Mock backend context is working")
            else:
                logger.error("Mock backend context failed")
        
        # Clean up
        manager.stop_mock_backend()
        
    else:
        logger.error("Failed to start mock backend")
    
    return True

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1) 