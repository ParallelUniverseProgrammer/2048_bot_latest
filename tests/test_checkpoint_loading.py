#!/usr/bin/env python3
"""
Checkpoint Loading Test Suite
=============================

Tests for checkpoint loading functionality including:
- Checkpoint metadata validation
- Model state loading and verification
- Compatibility checks between checkpoint and current system
- Error handling for corrupted or invalid checkpoints
- Performance monitoring during loading operations
- Memory management during checkpoint operations

Usage:
    python test_checkpoint_loading.py

Expected outcomes:
- Successful loading of valid checkpoints
- Proper error handling for invalid checkpoints
- Memory-efficient loading operations
- Metadata integrity verification
- Performance within acceptable bounds
"""

import sys
import os
import time
import traceback
from pathlib import Path
import json
from typing import Dict, Any, List, Optional, Tuple
import tempfile
import shutil

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from test_utils import TestLogger, BackendTester

class CheckpointLoadingTester:
    """Test suite for checkpoint loading functionality"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.logger = TestLogger()
        self.backend = BackendTester(base_url, self.logger)
        self.test_results = {
            'metadata_validation': False,
            'model_loading': False,
            'compatibility_check': False,
            'error_handling': False,
            'performance_test': False,
            'memory_management': False
        }
        self.temp_dir = None
        self.loading_times = []
    
    def setup_test_environment(self):
        """Setup temporary directory for test operations"""
        self.temp_dir = tempfile.mkdtemp(prefix="checkpoint_test_")
        self.logger.info(f"🔧 Test environment setup in: {self.temp_dir}")
    
    def cleanup_test_environment(self):
        """Cleanup temporary test files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.logger.info("🧹 Test environment cleaned up")
    
    def log_test_start(self, test_name: str):
        """Log the start of a test"""
        self.logger.info(f"🧪 Starting {test_name}...")
    
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Log the result of a test"""
        status = "✅ PASS" if success else "❌ FAIL"
        self.logger.info(f"{status} {test_name}")
        if details:
            self.logger.info(f"   Details: {details}")
    
    def test_metadata_validation(self) -> bool:
        """Test checkpoint metadata validation"""
        self.log_test_start("Metadata Validation Test")
        
        try:
            # Get available checkpoints
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Metadata Validation Test", False, "No checkpoints available")
                return False
            
            valid_checkpoints = 0
            for checkpoint in checkpoints:
                # Validate required metadata fields
                required_fields = ['id', 'name', 'timestamp', 'episode_count', 'score']
                missing_fields = [field for field in required_fields if field not in checkpoint]
                
                if not missing_fields:
                    valid_checkpoints += 1
                else:
                    self.logger.warning(f"   Checkpoint {checkpoint.get('name', 'unknown')} missing: {missing_fields}")
            
            if valid_checkpoints > 0:
                self.log_test_result("Metadata Validation Test", True, 
                                   f"{valid_checkpoints}/{len(checkpoints)} checkpoints valid")
                return True
            else:
                self.log_test_result("Metadata Validation Test", False, "No valid checkpoints found")
                return False
                
        except Exception as e:
            self.log_test_result("Metadata Validation Test", False, f"Exception: {str(e)}")
            return False
    
    def test_model_loading(self) -> bool:
        """Test model state loading from checkpoints"""
        self.log_test_start("Model Loading Test")
        
        try:
            # Get available checkpoints
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Model Loading Test", False, "No checkpoints available")
                return False
            
            # Test loading first checkpoint
            checkpoint = checkpoints[0]
            checkpoint_id = checkpoint.get('id', checkpoint.get('name', 'unknown'))
            
            # Measure loading time
            start_time = time.time()
            
            # Load checkpoint
            response = self.backend.load_checkpoint(checkpoint_id)
            
            loading_time = time.time() - start_time
            self.loading_times.append(loading_time)
            
            if response and response.get('success', False):
                self.log_test_result("Model Loading Test", True, 
                                   f"Loaded {checkpoint_id} in {loading_time:.2f}s")
                return True
            else:
                error_msg = response.get('error', 'Unknown error') if response else 'No response'
                self.log_test_result("Model Loading Test", False, f"Failed to load: {error_msg}")
                return False
                
        except Exception as e:
            self.log_test_result("Model Loading Test", False, f"Exception: {str(e)}")
            return False
    
    def test_compatibility_check(self) -> bool:
        """Test compatibility between checkpoints and current system"""
        self.log_test_start("Compatibility Check Test")
        
        try:
            # Get system info
            system_info = self.backend.get_system_info()
            if not system_info:
                self.log_test_result("Compatibility Check Test", False, "Cannot get system info")
                return False
            
            # Get available checkpoints
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Compatibility Check Test", False, "No checkpoints available")
                return False
            
            compatible_count = 0
            for checkpoint in checkpoints:
                # Check compatibility
                compatibility = self.backend.check_compatibility(checkpoint.get('id', checkpoint.get('name')))
                
                if compatibility and compatibility.get('compatible', False):
                    compatible_count += 1
                else:
                    issues = compatibility.get('issues', []) if compatibility else ['No compatibility info']
                    self.logger.warning(f"   Incompatible: {checkpoint.get('name', 'unknown')} - {issues}")
            
            if compatible_count > 0:
                self.log_test_result("Compatibility Check Test", True, 
                                   f"{compatible_count}/{len(checkpoints)} checkpoints compatible")
                return True
            else:
                self.log_test_result("Compatibility Check Test", False, "No compatible checkpoints found")
                return False
                
        except Exception as e:
            self.log_test_result("Compatibility Check Test", False, f"Exception: {str(e)}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling for invalid checkpoints"""
        self.log_test_start("Error Handling Test")
        
        try:
            # Test loading non-existent checkpoint
            response = self.backend.load_checkpoint("non_existent_checkpoint")
            if response and response.get('success', False):
                self.log_test_result("Error Handling Test", False, "Should have failed for non-existent checkpoint")
                return False
            
            # Test with invalid checkpoint ID format
            response = self.backend.load_checkpoint("")
            if response and response.get('success', False):
                self.log_test_result("Error Handling Test", False, "Should have failed for empty checkpoint ID")
                return False
            
            # Test with malformed checkpoint ID
            response = self.backend.load_checkpoint("../../../etc/passwd")
            if response and response.get('success', False):
                self.log_test_result("Error Handling Test", False, "Should have failed for malformed checkpoint ID")
                return False
            
            # Verify system is still responsive after errors
            checkpoints = self.backend.get_checkpoints()
            if not isinstance(checkpoints, list):
                self.log_test_result("Error Handling Test", False, "System unresponsive after errors")
                return False
            
            self.log_test_result("Error Handling Test", True, "Error handling working correctly")
            return True
            
        except Exception as e:
            self.log_test_result("Error Handling Test", False, f"Exception: {str(e)}")
            return False
    
    def test_performance(self) -> bool:
        """Test checkpoint loading performance"""
        self.log_test_start("Performance Test")
        
        try:
            # Get available checkpoints
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Performance Test", False, "No checkpoints available")
                return False
            
            # Test loading multiple checkpoints and measure performance
            performance_times = []
            
            for i, checkpoint in enumerate(checkpoints[:3]):  # Test first 3 checkpoints
                checkpoint_id = checkpoint.get('id', checkpoint.get('name', f'checkpoint_{i}'))
                
                start_time = time.time()
                response = self.backend.load_checkpoint(checkpoint_id)
                end_time = time.time()
                
                loading_time = end_time - start_time
                performance_times.append(loading_time)
                
                if not response or not response.get('success', False):
                    self.logger.warning(f"   Failed to load {checkpoint_id}")
                    continue
                    
                # Brief pause between loads
                time.sleep(0.5)
            
            if performance_times:
                avg_time = sum(performance_times) / len(performance_times)
                max_time = max(performance_times)
                min_time = min(performance_times)
                
                # Performance threshold: loading should be under 10 seconds
                if avg_time < 10.0:
                    self.log_test_result("Performance Test", True, 
                                       f"Avg: {avg_time:.2f}s, Range: {min_time:.2f}s - {max_time:.2f}s")
                    return True
                else:
                    self.log_test_result("Performance Test", False, 
                                       f"Loading too slow: {avg_time:.2f}s average")
                    return False
            else:
                self.log_test_result("Performance Test", False, "No performance data collected")
                return False
                
        except Exception as e:
            self.log_test_result("Performance Test", False, f"Exception: {str(e)}")
            return False
    
    def test_memory_management(self) -> bool:
        """Test memory usage during checkpoint operations"""
        self.log_test_start("Memory Management Test")
        
        try:
            import psutil
            process = psutil.Process()
            
            # Record initial memory usage
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Get available checkpoints
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Memory Management Test", False, "No checkpoints available")
                return False
            
            # Load multiple checkpoints and monitor memory
            memory_readings = [initial_memory]
            
            for checkpoint in checkpoints[:2]:  # Test first 2 checkpoints
                checkpoint_id = checkpoint.get('id', checkpoint.get('name', 'unknown'))
                
                # Load checkpoint
                response = self.backend.load_checkpoint(checkpoint_id)
                
                # Record memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_readings.append(current_memory)
                
                if not response or not response.get('success', False):
                    self.logger.warning(f"   Failed to load {checkpoint_id}")
                    continue
                
                time.sleep(1)  # Allow memory to stabilize
            
            # Analyze memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            max_memory = max(memory_readings)
            memory_increase = final_memory - initial_memory
            
            # Memory threshold: should not increase by more than 500MB
            if memory_increase < 500:
                self.log_test_result("Memory Management Test", True, 
                                   f"Memory: {initial_memory:.1f}MB -> {final_memory:.1f}MB (peak: {max_memory:.1f}MB)")
                return True
            else:
                self.log_test_result("Memory Management Test", False, 
                                   f"Memory increase too large: {memory_increase:.1f}MB")
                return False
                
        except ImportError:
            self.log_test_result("Memory Management Test", False, "psutil not available")
            return False
        except Exception as e:
            self.log_test_result("Memory Management Test", False, f"Exception: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all checkpoint loading tests"""
        self.logger.info("🚀 Starting Checkpoint Loading Test Suite")
        self.logger.info("=" * 50)
        
        try:
            # Setup test environment
            self.setup_test_environment()
            
            # Test backend connectivity first
            if not self.backend.test_connectivity():
                self.logger.error("❌ Backend connectivity failed - aborting tests")
                return
            
            # Run all tests
            self.test_results['metadata_validation'] = self.test_metadata_validation()
            self.test_results['model_loading'] = self.test_model_loading()
            self.test_results['compatibility_check'] = self.test_compatibility_check()
            self.test_results['error_handling'] = self.test_error_handling()
            self.test_results['performance_test'] = self.test_performance()
            self.test_results['memory_management'] = self.test_memory_management()
            
            # Generate final report
            self.generate_report()
            
        finally:
            # Cleanup test environment
            self.cleanup_test_environment()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        self.logger.info("📊 Checkpoint Loading Test Results")
        self.logger.info("=" * 50)
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            self.logger.info(f"{status} {test_name}")
        
        self.logger.info("-" * 50)
        self.logger.info(f"📈 Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            self.logger.info("🎉 All checkpoint loading tests passed!")
        else:
            self.logger.warning(f"⚠️  {total - passed} tests failed")
        
        # Performance summary
        if self.loading_times:
            avg_loading_time = sum(self.loading_times) / len(self.loading_times)
            self.logger.info(f"⏱️  Average loading time: {avg_loading_time:.2f}s")


def main():
    """Main test execution function"""
    try:
        tester = CheckpointLoadingTester()
        tester.run_all_tests()
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"❌ Test suite failed: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 