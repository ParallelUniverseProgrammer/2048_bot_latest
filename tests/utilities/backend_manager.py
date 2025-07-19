#!/usr/bin/env python3
"""
Backend Availability Manager
============================

This module provides a comprehensive backend availability management system
that can detect backend status, automatically start mock backends for testing,
and provide a unified interface for all backend-dependent tests.

Features:
- Automatic backend availability detection with exponential backoff
- Seamless mock backend integration for offline testing
- Unified testing interface for all backend-dependent tests
- Smart caching to avoid repeated connectivity checks
- Graceful fallback mechanisms
- Comprehensive logging and error handling

Usage:
    from backend_availability_manager import BackendAvailabilityManager
    
    manager = BackendAvailabilityManager()
    
    # Ensure backend is available (real or mock)
    if manager.ensure_backend_available():
        # Run tests
        pass
    else:
        # Handle failure
        pass
"""

import time
import threading
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager

from tests.utilities.test_utils import BackendTester, TestLogger


class BackendAvailabilityManager:
    """
    Comprehensive backend availability management system
    
    This class provides a unified interface for managing backend availability,
    including automatic detection, mock backend startup, and graceful fallback
    mechanisms for testing scenarios.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", 
                 mock_backend_port: int = 8000,
                 test_logger: Optional[TestLogger] = None):
        """
        Initialize the backend availability manager
        
        Args:
            base_url: Base URL for the backend server
            mock_backend_port: Port to use for mock backend
            test_logger: Optional TestLogger instance for formatted output
        """
        self.base_url = base_url
        self.mock_backend_port = mock_backend_port
        self.test_logger = test_logger or TestLogger()
        
        # Backend tester for connectivity checks
        self.backend_tester = BackendTester(base_url, self.test_logger)
        
        # Mock backend management
        self.mock_backend = None
        self.mock_backend_thread = None
        self.is_mock_backend_running = False
        
        # Caching and state management
        self._last_availability_check = 0
        self._cache_duration = 30  # Cache results for 30 seconds
        self._cached_availability = None
        self._backend_type = None  # 'real', 'mock', or None
        
        # Configuration
        self.max_startup_wait = 30  # Maximum time to wait for backend startup
        self.connectivity_timeout = 5  # Timeout for individual connectivity checks
        
    def is_backend_available(self, use_cache: bool = True, 
                           prefer_real_backend: bool = True) -> bool:
        """
        Check if any backend (real or mock) is available
        
        Args:
            use_cache: Whether to use cached results if available
            prefer_real_backend: Whether to prefer real backend over mock
            
        Returns:
            bool: True if backend is available, False otherwise
        """
        current_time = time.time()
        
        # Use cached result if available and fresh
        if use_cache and self._cached_availability is not None:
            if current_time - self._last_availability_check < self._cache_duration:
                return self._cached_availability
        
        # Check real backend first if preferred
        if prefer_real_backend:
            if self.backend_tester.is_backend_available(use_cache=False):
                self._cached_availability = True
                self._backend_type = 'real'
                self._last_availability_check = current_time
                return True
        
        # Check mock backend if running
        if self.is_mock_backend_running:
            if self.backend_tester.is_backend_available(use_cache=False):
                self._cached_availability = True
                self._backend_type = 'mock'
                self._last_availability_check = current_time
                return True
        
        # No backend available
        self._cached_availability = False
        self._backend_type = None
        self._last_availability_check = current_time
        return False
    
    def ensure_backend_available(self, auto_start_mock: bool = True, 
                               wait_for_startup: bool = True) -> bool:
        """
        Ensure that a backend (real or mock) is available
        
        Args:
            auto_start_mock: Whether to automatically start mock backend if real backend unavailable
            wait_for_startup: Whether to wait for backend startup before returning
            
        Returns:
            bool: True if backend is available, False otherwise
        """
        self.test_logger.info("Ensuring backend availability...")
        
        # Check if real backend is available
        if self.backend_tester.is_backend_available(use_cache=False):
            self.test_logger.ok("Real backend is available")
            self._backend_type = 'real'
            return True
        
        # Check if mock backend is already running
        if self.is_mock_backend_running:
            if self.backend_tester.is_backend_available(use_cache=False):
                self.test_logger.ok("Mock backend is already running")
                self._backend_type = 'mock'
                return True
        
        # Try to start mock backend if requested
        if auto_start_mock:
            self.test_logger.info("Real backend not available - starting mock backend...")
            
            if self.start_mock_backend():
                if wait_for_startup:
                    return self.wait_for_backend_startup()
                else:
                    return True
            else:
                self.test_logger.error("Failed to start mock backend")
                return False
        
        self.test_logger.error("No backend available and auto_start_mock is disabled")
        return False
    
    def start_mock_backend(self) -> bool:
        """
        Start the mock backend server
        
        Returns:
            bool: True if successfully started, False otherwise
        """
        if self.is_mock_backend_running:
            self.test_logger.warning("Mock backend is already running")
            return True
        
        try:
            from mock_backend import MockBackend
            
            self.test_logger.info(f"Starting mock backend on port {self.mock_backend_port}...")
            
            # Create mock backend instance
            self.mock_backend = MockBackend(port=self.mock_backend_port)
            
            # Start in background thread
            self.mock_backend_thread = threading.Thread(
                target=self._run_mock_backend,
                daemon=True
            )
            self.mock_backend_thread.start()
            
            # Mark as running
            self.is_mock_backend_running = True
            
            self.test_logger.ok("Mock backend started successfully")
            return True
            
        except ImportError:
            self.test_logger.error("Mock backend module not available")
            return False
        except Exception as e:
            self.test_logger.error(f"Failed to start mock backend: {e}")
            return False
    
    def _run_mock_backend(self):
        """Internal method to run mock backend in thread"""
        try:
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
        
        self.test_logger.ok("Mock backend stopped")
    
    def wait_for_backend_startup(self, timeout: int = None) -> bool:
        """
        Wait for backend to become available
        
        Args:
            timeout: Maximum time to wait (uses instance default if None)
            
        Returns:
            bool: True if backend became available, False if timeout
        """
        timeout = timeout or self.max_startup_wait
        
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
    
    def get_backend_type(self) -> Optional[str]:
        """Get the type of currently active backend"""
        return self._backend_type
    
    def get_backend_stats(self) -> Dict[str, Any]:
        """Get statistics about backend availability"""
        return {
            'backend_type': self._backend_type,
            'is_available': self.is_backend_available(),
            'is_mock_running': self.is_mock_backend_running,
            'last_check': self._last_availability_check,
            'cache_duration': self._cache_duration,
            'base_url': self.base_url
        }
    
    @contextmanager
    def backend_context(self, auto_start_mock: bool = True, 
                       auto_cleanup: bool = True):
        """
        Context manager for backend availability
        
        Args:
            auto_start_mock: Whether to automatically start mock backend if needed
            auto_cleanup: Whether to automatically clean up mock backend on exit
            
        Usage:
            with manager.backend_context() as backend_available:
                if backend_available:
                    # Run tests
                    pass
        """
        # Ensure backend is available
        backend_available = self.ensure_backend_available(auto_start_mock=auto_start_mock)
        
        try:
            yield backend_available
        finally:
            # Clean up mock backend if requested and it was started by us
            if auto_cleanup and self.is_mock_backend_running and self._backend_type == 'mock':
                self.stop_mock_backend()
    
    def run_with_backend(self, test_func: Callable, *args, **kwargs) -> Any:
        """
        Run a test function with backend availability guaranteed
        
        Args:
            test_func: Function to run
            *args: Arguments to pass to test_func
            **kwargs: Keyword arguments to pass to test_func
            
        Returns:
            Result of test_func or None if backend not available
        """
        with self.backend_context() as backend_available:
            if backend_available:
                return test_func(*args, **kwargs)
            else:
                self.test_logger.error("Cannot run test - backend not available")
                return None


# Convenience functions for common use cases
def ensure_backend_available(base_url: str = "http://localhost:8000", 
                           test_logger: Optional[TestLogger] = None) -> bool:
    """
    Convenience function to ensure backend is available
    
    Args:
        base_url: Base URL for the backend server
        test_logger: Optional TestLogger instance
        
    Returns:
        bool: True if backend is available, False otherwise
    """
    manager = BackendAvailabilityManager(base_url=base_url, test_logger=test_logger)
    return manager.ensure_backend_available()


def with_backend(test_func: Callable, base_url: str = "http://localhost:8000", 
                test_logger: Optional[TestLogger] = None):
    """
    Decorator to ensure backend is available before running test
    
    Args:
        test_func: Function to decorate
        base_url: Base URL for the backend server
        test_logger: Optional TestLogger instance
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        manager = BackendAvailabilityManager(base_url=base_url, test_logger=test_logger)
        return manager.run_with_backend(test_func, *args, **kwargs)
    
    return wrapper


# Global manager instance for convenience
_global_manager = None

def get_global_manager() -> BackendAvailabilityManager:
    """Get the global backend availability manager instance"""
    global _global_manager
    if _global_manager is None:
        _global_manager = BackendAvailabilityManager()
    return _global_manager


def main():
    """Main entry point for backend availability testing"""
    logger = TestLogger()
    logger.banner("Backend Availability Manager Test", 60)
    
    manager = BackendAvailabilityManager()
    
    logger.info("Testing backend availability management...")
    
    # Test backend availability
    if manager.ensure_backend_available():
        logger.ok("Backend is available")
        
        # Show backend stats
        stats = manager.get_backend_stats()
        logger.info(f"Backend type: {stats['backend_type']}")
        logger.info(f"Base URL: {stats['base_url']}")
        
        # Test context manager
        with manager.backend_context() as available:
            if available:
                logger.ok("Backend context is working")
            else:
                logger.error("Backend context failed")
        
        return True
    else:
        logger.error("Backend is not available")
        return False

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1) 