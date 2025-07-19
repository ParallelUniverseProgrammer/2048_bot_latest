#!/usr/bin/env python3
"""
Integration Test for Checkpoint Loading
======================================

This test verifies that checkpoint loading works correctly in a real system
environment, including:
- Checkpoint metadata validation
- Model loading and compatibility
- Performance benchmarks
- Error handling scenarios
- Memory management
"""

import sys
import os
import time
import json
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))

from tests.utilities.test_utils import TestLogger, BackendTester, get_backend_tester
from tests.utilities.backend_manager import requires_mock_backend

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 60

class CheckpointLoadingTester:
    """Test checkpoint loading functionality"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.logger = TestLogger()
        self.backend = get_backend_tester(base_url, self.logger)
        self.test_results = []
        
    def setup_test_environment(self):
        """Setup test environment"""
        self.logger.info("Setting up test environment...")
        
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        self.logger.info("Cleaning up test environment...")
        
    def log_test_start(self, test_name: str):
        """Log test start"""
        self.logger.section(f"Test: {test_name}")
        
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        if success:
            self.logger.ok(f"{test_name}: PASS")
        else:
            self.logger.error(f"{test_name}: FAIL")
        if details:
            self.logger.log(f"Details: {details}")
        self.test_results.append((test_name, success, details))
    
    def test_metadata_validation(self) -> bool:
        """Test checkpoint metadata validation"""
        self.log_test_start("Metadata Validation")
        
        try:
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Metadata Validation", False, "No checkpoints available")
                return False
            
            # Validate first checkpoint metadata
            checkpoint = checkpoints[0]
            # Updated to match real backend structure
            required_fields = ['id', 'nickname', 'episode', 'created_at', 'model_config', 'performance_metrics', 'file_size', 'absolute_path']
            
            missing_fields = [field for field in required_fields if field not in checkpoint]
            if missing_fields:
                self.log_test_result("Metadata Validation", False, f"Missing fields: {missing_fields}")
                return False
            
            # Validate data types
            if not isinstance(checkpoint['episode'], int):
                self.log_test_result("Metadata Validation", False, "Episode should be integer")
                return False
                
            # Check that performance_metrics contains best_score
            if 'performance_metrics' not in checkpoint or 'best_score' not in checkpoint['performance_metrics']:
                self.log_test_result("Metadata Validation", False, "Missing best_score in performance_metrics")
                return False
                
            if not isinstance(checkpoint['performance_metrics']['best_score'], (int, float)):
                self.log_test_result("Metadata Validation", False, "best_score should be numeric")
                return False
            
            self.log_test_result("Metadata Validation", True, f"Validated {len(checkpoints)} checkpoints")
            return True
            
        except Exception as e:
            self.log_test_result("Metadata Validation", False, f"Exception: {str(e)}")
            return False
    
    def test_model_loading(self) -> bool:
        """Test model loading functionality"""
        self.log_test_start("Model Loading")
        
        try:
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Model Loading", False, "No checkpoints available")
                return False
            
            # Test loading first checkpoint
            checkpoint_id = checkpoints[0]['id']
            start_time = time.time()
            
            checkpoint_info = self.backend.load_checkpoint(checkpoint_id)
            load_time = time.time() - start_time
            
            if not checkpoint_info:
                self.log_test_result("Model Loading", False, "Failed to load checkpoint")
                return False
            
            # Validate load time
            if load_time > 10:  # Should load within 10 seconds
                self.log_test_result("Model Loading", False, f"Slow loading time: {load_time:.2f}s")
                return False
            
            self.log_test_result("Model Loading", True, f"Loaded in {load_time:.2f}s")
            return True
            
        except Exception as e:
            self.log_test_result("Model Loading", False, f"Exception: {str(e)}")
            return False
    
    def test_compatibility_check(self) -> bool:
        """Test checkpoint compatibility"""
        self.log_test_start("Compatibility Check")
        
        try:
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Compatibility Check", False, "No checkpoints available")
                return False
            
            # Check if checkpoints have model configuration
            for checkpoint in checkpoints[:3]:  # Check first 3 checkpoints
                checkpoint_info = self.backend.load_checkpoint(checkpoint['id'])
                if checkpoint_info and 'model_config' in checkpoint_info:
                    self.log_test_result("Compatibility Check", True, "Model configuration present")
                    return True
            
            self.log_test_result("Compatibility Check", False, "No model configuration found")
            return False
            
        except Exception as e:
            self.log_test_result("Compatibility Check", False, f"Exception: {str(e)}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling for invalid checkpoints"""
        self.log_test_start("Error Handling")
        
        try:
            # Test with invalid checkpoint ID
            invalid_checkpoint_info = self.backend.load_checkpoint("invalid_id_12345")
            
            # Should handle gracefully (return None or error response)
            if invalid_checkpoint_info is None:
                self.log_test_result("Error Handling", True, "Gracefully handled invalid checkpoint")
                return True
            else:
                self.log_test_result("Error Handling", False, "Did not handle invalid checkpoint properly")
                return False
                
        except Exception as e:
            self.log_test_result("Error Handling", False, f"Exception: {str(e)}")
            return False
    
    def test_performance(self) -> bool:
        """Test checkpoint loading performance"""
        self.log_test_start("Performance Test")
        
        try:
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Performance Test", False, "No checkpoints available")
                return False
            
            # Test loading multiple checkpoints
            load_times = []
            for checkpoint in checkpoints[:3]:  # Test first 3 checkpoints
                start_time = time.time()
                checkpoint_info = self.backend.load_checkpoint(checkpoint['id'])
                load_time = time.time() - start_time
                
                if checkpoint_info:
                    load_times.append(load_time)
            
            if not load_times:
                self.log_test_result("Performance Test", False, "No checkpoints loaded successfully")
                return False
            
            avg_load_time = sum(load_times) / len(load_times)
            max_load_time = max(load_times)
            
            if avg_load_time > 5:  # Average should be under 5 seconds
                self.log_test_result("Performance Test", False, f"Slow average load time: {avg_load_time:.2f}s")
                return False
            
            if max_load_time > 10:  # Max should be under 10 seconds
                self.log_test_result("Performance Test", False, f"Slow max load time: {max_load_time:.2f}s")
                return False
            
            self.log_test_result("Performance Test", True, f"Avg: {avg_load_time:.2f}s, Max: {max_load_time:.2f}s")
            return True
            
        except Exception as e:
            self.log_test_result("Performance Test", False, f"Exception: {str(e)}")
            return False
    
    def test_memory_management(self) -> bool:
        """Test memory management during checkpoint loading"""
        self.log_test_start("Memory Management")
        
        try:
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Memory Management", False, "No checkpoints available")
                return False
            
            # Load multiple checkpoints to test memory management
            for i, checkpoint in enumerate(checkpoints[:5]):  # Test first 5 checkpoints
                checkpoint_info = self.backend.load_checkpoint(checkpoint['id'])
                if not checkpoint_info:
                    self.log_test_result("Memory Management", False, f"Failed to load checkpoint {i+1}")
                    return False
            
            self.log_test_result("Memory Management", True, "Successfully loaded multiple checkpoints")
            return True
            
        except Exception as e:
            self.log_test_result("Memory Management", False, f"Exception: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all checkpoint loading tests"""
        self.logger.banner("Checkpoint Loading Integration Test Suite", 60)
        
        self.setup_test_environment()
        
        # Run all tests
        tests = [
            self.test_metadata_validation,
            self.test_model_loading,
            self.test_compatibility_check,
            self.test_error_handling,
            self.test_performance,
            self.test_memory_management
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                self.logger.error(f"Test {test.__name__} failed with exception: {str(e)}")
        
        self.cleanup_test_environment()
        
        # Generate summary
        self.generate_report()
        
        return passed == total
    
    def generate_report(self):
        """Generate test report"""
        self.logger.separator(60)
        self.logger.success("Checkpoint Loading Integration Test Report")
        
        # Create summary table
        self.logger.table_header(["Test", "Status", "Details"], [25, 10, 25])
        for test_name, success, details in self.test_results:
            status = "PASS" if success else "FAIL"
            self.logger.table_row([test_name, status, details[:23] + "..." if len(details) > 25 else details], [25, 10, 25])
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        self.logger.log(f"\nSummary: {passed}/{total} tests passed")
        
        if passed == total:
            self.logger.success("All checkpoint loading tests passed!")
        else:
            self.logger.error(f"{total - passed} tests failed")

@requires_mock_backend("Checkpoint Loading Integration Tests")
def main():
    """Main entry point"""
    logger = TestLogger()
    
    logger.banner("Checkpoint Loading Integration Tests", 60)
    
    tester = CheckpointLoadingTester()
    success = tester.run_all_tests()
    
    if success:
        logger.success("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 