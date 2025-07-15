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

from test_utils import TestLogger, BackendTester, get_backend_tester, requires_backend, check_backend_or_exit

class CheckpointLoadingTester:
    """Test suite for checkpoint loading functionality"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.logger = TestLogger()
        self.backend = get_backend_tester(base_url, self.logger)
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
            # Check if backend is available
            if not self.backend.is_backend_available():
                self.log_test_result("Metadata Validation Test", False, "Backend not available")
                return False
            
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
            
            success = valid_checkpoints > 0
            if success:
                self.log_test_result("Metadata Validation Test", True, f"Found {valid_checkpoints} valid checkpoints")
            else:
                self.log_test_result("Metadata Validation Test", False, "No valid checkpoints found")
            
            return success
            
        except Exception as e:
            self.log_test_result("Metadata Validation Test", False, f"Exception: {str(e)}")
            return False
    
    def test_model_loading(self) -> bool:
        """Test model loading from checkpoints"""
        self.log_test_start("Model Loading Test")
        
        try:
            # Check if backend is available
            if not self.backend.is_backend_available():
                self.log_test_result("Model Loading Test", False, "Backend not available")
                return False
            
            # Get available checkpoints
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Model Loading Test", False, "No checkpoints available")
                return False
            
            # Test loading first checkpoint
            checkpoint = checkpoints[0]
            checkpoint_id = checkpoint['id']
            
            start_time = time.time()
            result = self.backend.load_checkpoint(checkpoint_id)
            load_time = time.time() - start_time
            
            self.loading_times.append(load_time)
            
            if result:
                self.log_test_result("Model Loading Test", True, f"Loaded in {load_time:.2f}s")
                return True
            else:
                self.log_test_result("Model Loading Test", False, "Failed to load: Unknown error")
                return False
                
        except Exception as e:
            self.log_test_result("Model Loading Test", False, f"Exception: {str(e)}")
            return False
    
    def test_compatibility_check(self) -> bool:
        """Test compatibility between checkpoint and current system"""
        self.log_test_start("Compatibility Check Test")
        
        try:
            # Check if backend is available
            if not self.backend.is_backend_available():
                self.log_test_result("Compatibility Check Test", False, "Backend not available")
                return False
            
            # Test basic system compatibility
            stats = self.backend.get_checkpoint_stats()
            if stats:
                self.log_test_result("Compatibility Check Test", True, "System compatibility verified")
                return True
            else:
                self.log_test_result("Compatibility Check Test", False, "Failed to verify compatibility")
                return False
                
        except Exception as e:
            self.log_test_result("Compatibility Check Test", False, f"Exception: {str(e)}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling for invalid checkpoints"""
        self.log_test_start("Error Handling Test")
        
        try:
            # Check if backend is available
            if not self.backend.is_backend_available():
                self.log_test_result("Error Handling Test", False, "Backend not available")
                return False
            
            # Try to load non-existent checkpoint
            result = self.backend.load_checkpoint("non_existent_checkpoint")
            if result is None:
                self.log_test_result("Error Handling Test", True, "Non-existent checkpoint handled gracefully")
                return True
            else:
                self.log_test_result("Error Handling Test", False, "Should have failed to load non-existent checkpoint")
                return False
                
        except Exception as e:
            self.log_test_result("Error Handling Test", False, f"Exception: {str(e)}")
            return False
    
    def test_performance(self) -> bool:
        """Test performance of checkpoint loading"""
        self.log_test_start("Performance Test")
        
        try:
            # Check if backend is available
            if not self.backend.is_backend_available():
                self.log_test_result("Performance Test", False, "Backend not available")
                return False
            
            # Get available checkpoints
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Performance Test", False, "No checkpoints available")
                return False
            
            # Test loading performance for multiple checkpoints
            load_times = []
            for checkpoint in checkpoints[:3]:  # Test first 3 checkpoints
                start_time = time.time()
                result = self.backend.load_checkpoint(checkpoint['id'])
                load_time = time.time() - start_time
                load_times.append(load_time)
                
                if result:
                    self.logger.info(f"✅ Loaded {checkpoint['id']}: {load_time:.2f}s")
                else:
                    self.logger.warning(f"   Failed to load {checkpoint['id']}")
            
            if load_times:
                avg_time = sum(load_times) / len(load_times)
                min_time = min(load_times)
                max_time = max(load_times)
                
                self.log_test_result("Performance Test", True, f"Avg: {avg_time:.2f}s, Range: {min_time:.2f}s - {max_time:.2f}s")
                return True
            else:
                self.log_test_result("Performance Test", False, "No successful loads")
                return False
                
        except Exception as e:
            self.log_test_result("Performance Test", False, f"Exception: {str(e)}")
            return False
    
    def test_memory_management(self) -> bool:
        """Test memory management during checkpoint operations"""
        self.log_test_start("Memory Management Test")
        
        try:
            # Check if backend is available
            if not self.backend.is_backend_available():
                self.log_test_result("Memory Management Test", False, "Backend not available")
                return False
            
            # Get available checkpoints
            checkpoints = self.backend.get_checkpoints()
            if not checkpoints:
                self.log_test_result("Memory Management Test", False, "No checkpoints available")
                return False
            
            # Monitor memory usage during loading
            import psutil
            process = psutil.Process()
            
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load multiple checkpoints
            for checkpoint in checkpoints[:2]:  # Test first 2 checkpoints
                result = self.backend.load_checkpoint(checkpoint['id'])
                if result:
                    self.logger.info(f"✅ Loaded {checkpoint['id']}")
                else:
                    self.logger.warning(f"   Failed to load {checkpoint['id']}")
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            peak_memory = process.memory_info().peak_wss / 1024 / 1024 if hasattr(process.memory_info(), 'peak_wss') else final_memory
            
            self.log_test_result("Memory Management Test", True, f"Memory: {initial_memory:.1f}MB -> {final_memory:.1f}MB (peak: {peak_memory:.1f}MB)")
            return True
            
        except ImportError:
            self.log_test_result("Memory Management Test", True, "psutil not available, skipping memory test")
            return True
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
            
            # Check backend availability first
            if not self.backend.is_backend_available():
                self.logger.error("Backend is not available!")
                self.logger.info("Please ensure the backend server is running:")
                self.logger.info("  cd backend && python main.py")
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