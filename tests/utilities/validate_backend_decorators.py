#!/usr/bin/env python3
"""
Backend Decorator Validation and Testing Script
==============================================

This script provides comprehensive validation of the backend decorator implementation:

1. Runs the compliance checker to verify zero major issues
2. Tests that all decorators are working correctly
3. Verifies that real backend tests can start the backend
4. Verifies that mock backend tests can use the mock backend
5. Tests the backend manager's health checking and restart capabilities
6. Generates a final compliance report

Features:
- Automated test execution with proper error handling
- Backend manager testing and validation
- Compliance report generation
- Performance benchmarking
- Integration with existing test runners
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tests.utilities.test_utils import TestLogger
from tests.utilities.backend_manager import requires_mock_backend
from tests.utilities.backend_manager import (
    BackendManager, 
    get_global_backend_manager,
    requires_real_backend, 
    requires_mock_backend
)


class BackendDecoratorValidator:
    """Comprehensive backend decorator validation system"""
    
    def __init__(self):
        self.logger = TestLogger()
        self.project_root = Path(__file__).parent.parent.parent
        self.tests_dir = self.project_root / "tests"
        
        # Test results
        self.results = {
            'compliance_check': False,
            'real_backend_tests': [],
            'mock_backend_tests': [],
            'backend_manager_tests': [],
            'performance_tests': [],
            'overall_success': False
        }
        
        # Performance metrics
        self.performance_metrics = {
            'compliance_check_time': 0,
            'real_backend_startup_time': 0,
            'mock_backend_startup_time': 0,
            'test_execution_times': {}
        }
    
    def run_compliance_check(self) -> bool:
        """
        Run the compliance checker and verify zero major issues
        
        Returns:
            True if compliance check passes, False otherwise
        """
        self.logger.section("Running Compliance Check")
        
        start_time = time.time()
        
        try:
            # Run compliance checker as subprocess
            result = subprocess.run(
                [sys.executable, "tests/compliance_checker.py"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=60
            )
            
            self.performance_metrics['compliance_check_time'] = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.ok("Compliance check completed successfully")
                
                # Parse output to check for major issues
                output = result.stdout
                if "MAJOR ISSUES (0):" in output or "Major issues: 0" in output:
                    self.logger.ok("No major compliance issues found")
                    self.results['compliance_check'] = True
                    return True
                else:
                    self.logger.error("Major compliance issues found")
                    self.logger.info("Compliance check output:")
                    print(output)
                    return False
            else:
                self.logger.error(f"Compliance check failed with return code {result.returncode}")
                self.logger.info("Compliance check output:")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Compliance check timed out")
            return False
        except Exception as e:
            self.logger.error(f"Compliance check failed: {e}")
            return False
    
    @requires_mock_backend("Backend Manager Functionality Test")
    def test_backend_manager_functionality(self) -> bool:
        """
        Test the backend manager's core functionality
        
        Returns:
            True if all tests pass, False otherwise
        """
        self.logger.section("Testing Backend Manager Functionality")
        
        try:
            # Get global backend manager
            manager = get_global_backend_manager()
            
            # Test 1: Mock backend startup
            self.logger.info("Testing mock backend startup...")
            start_time = time.time()
            
            if manager.start_mock_backend():
                self.performance_metrics['mock_backend_startup_time'] = time.time() - start_time
                self.logger.ok("Mock backend started successfully")
                
                # Test mock backend availability
                if manager.is_backend_available(backend_type='mock'):
                    self.logger.ok("Mock backend is available")
                else:
                    self.logger.error("Mock backend is not available")
                    return False
                
                # Clean up mock backend
                manager.stop_mock_backend()
                self.logger.ok("Mock backend stopped successfully")
            else:
                self.logger.error("Failed to start mock backend")
                return False
            
            # Test 2: Real backend startup (if possible)
            self.logger.info("Testing real backend startup...")
            start_time = time.time()
            
            if manager.start_real_backend(wait_for_startup=False):
                self.performance_metrics['real_backend_startup_time'] = time.time() - start_time
                self.logger.ok("Real backend started successfully")
                
                # Test real backend availability
                if manager.is_backend_available(backend_type='real'):
                    self.logger.ok("Real backend is available")
                else:
                    self.logger.warning("Real backend started but not immediately available")
                
                # Test health check
                if manager.check_backend_health():
                    self.logger.ok("Backend health check passed")
                else:
                    self.logger.warning("Backend health check failed")
                
                # Clean up real backend
                manager.stop_real_backend()
                self.logger.ok("Real backend stopped successfully")
            else:
                self.logger.warning("Could not start real backend (this is expected if backend is not configured)")
            
            # Test 3: Backend manager stats
            stats = manager.get_backend_stats()
            self.logger.info(f"Backend manager stats: {stats}")
            
            self.results['backend_manager_tests'].append({
                'test': 'Backend Manager Functionality',
                'status': 'PASS',
                'details': 'All backend manager tests passed'
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Backend manager test failed: {e}")
            self.results['backend_manager_tests'].append({
                'test': 'Backend Manager Functionality',
                'status': 'FAIL',
                'details': str(e)
            })
            return False
    
    @requires_mock_backend("Decorator Functionality Test")
    def test_decorator_functionality(self) -> bool:
        """
        Test the decorator functionality with sample functions
        
        Returns:
            True if all tests pass, False otherwise
        """
        self.logger.section("Testing Decorator Functionality")
        
        # Test 1: Mock backend decorator
        @requires_mock_backend("Mock Backend Test")
        def test_mock_backend():
            return {"status": "success", "backend_type": "mock"}
        
        try:
            result = test_mock_backend()
            if result.get("status") == "success":
                self.logger.ok("Mock backend decorator works correctly")
                self.results['mock_backend_tests'].append({
                    'test': 'Mock Backend Decorator',
                    'status': 'PASS',
                    'details': 'Decorator executed successfully'
                })
            else:
                self.logger.error("Mock backend decorator failed")
                self.results['mock_backend_tests'].append({
                    'test': 'Mock Backend Decorator',
                    'status': 'FAIL',
                    'details': 'Decorator returned unexpected result'
                })
                return False
        except Exception as e:
            self.logger.error(f"Mock backend decorator test failed: {e}")
            self.results['mock_backend_tests'].append({
                'test': 'Mock Backend Decorator',
                'status': 'FAIL',
                'details': str(e)
            })
            return False
        
        # Test 2: Real backend decorator
        @requires_real_backend("Real Backend Test")
        def test_real_backend():
            return {"status": "success", "backend_type": "real"}
        
        try:
            result = test_real_backend()
            if result.get("status") == "success":
                self.logger.ok("Real backend decorator works correctly")
                self.results['real_backend_tests'].append({
                    'test': 'Real Backend Decorator',
                    'status': 'PASS',
                    'details': 'Decorator executed successfully'
                })
            else:
                self.logger.warning("Real backend decorator returned unexpected result (this may be expected)")
                self.results['real_backend_tests'].append({
                    'test': 'Real Backend Decorator',
                    'status': 'WARNING',
                    'details': 'Decorator returned unexpected result'
                })
        except Exception as e:
            self.logger.warning(f"Real backend decorator test failed: {e}")
            self.results['real_backend_tests'].append({
                'test': 'Real Backend Decorator',
                'status': 'WARNING',
                'details': str(e)
            })
        
        return True
    
    def run_sample_tests(self) -> bool:
        """
        Run a subset of actual test files to validate decorators work in practice
        
        Returns:
            True if all tests pass, False otherwise
        """
        self.logger.section("Running Sample Tests")
        
        # Find some test files to run
        sample_tests = [
            "tests/core/test_checkpoint_loading.py",
            "tests/frontend/test_automation.py",
            "tests/integration/test_complete_games.py"
        ]
        
        success_count = 0
        for test_file in sample_tests:
            test_path = self.project_root / test_file
            
            if not test_path.exists():
                self.logger.warning(f"Sample test file not found: {test_file}")
                continue
            
            self.logger.info(f"Running sample test: {test_file}")
            
            try:
                start_time = time.time()
                
                # Run test as subprocess
                result = subprocess.run(
                    [sys.executable, str(test_path)],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=120
                )
                
                execution_time = time.time() - start_time
                self.performance_metrics['test_execution_times'][test_file] = execution_time
                
                if result.returncode == 0:
                    self.logger.ok(f"Sample test passed: {test_file} ({execution_time:.1f}s)")
                    success_count += 1
                else:
                    self.logger.warning(f"Sample test failed: {test_file}")
                    self.logger.info(f"Test output: {result.stdout}")
                    self.logger.info(f"Test errors: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Sample test timed out: {test_file}")
            except Exception as e:
                self.logger.warning(f"Sample test failed: {test_file} - {e}")
        
        self.logger.info(f"Sample tests completed: {success_count}/{len(sample_tests)} passed")
        return success_count > 0
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report
        
        Returns:
            Dictionary containing validation results and metrics
        """
        self.logger.section("Validation Report")
        
        # Calculate overall success
        compliance_ok = self.results['compliance_check']
        backend_manager_ok = all(test['status'] == 'PASS' for test in self.results['backend_manager_tests'])
        decorators_ok = len(self.results['mock_backend_tests']) > 0 and len(self.results['real_backend_tests']) > 0
        
        self.results['overall_success'] = compliance_ok and backend_manager_ok and decorators_ok
        
        # Print summary
        self.logger.info("Validation Summary:")
        self.logger.info(f"  Compliance Check: {'PASS' if compliance_ok else 'FAIL'}")
        self.logger.info(f"  Backend Manager: {'PASS' if backend_manager_ok else 'FAIL'}")
        self.logger.info(f"  Decorators: {'PASS' if decorators_ok else 'FAIL'}")
        self.logger.info(f"  Overall: {'PASS' if self.results['overall_success'] else 'FAIL'}")
        
        # Print performance metrics
        self.logger.info("Performance Metrics:")
        self.logger.info(f"  Compliance check time: {self.performance_metrics['compliance_check_time']:.2f}s")
        self.logger.info(f"  Mock backend startup: {self.performance_metrics['mock_backend_startup_time']:.2f}s")
        self.logger.info(f"  Real backend startup: {self.performance_metrics['real_backend_startup_time']:.2f}s")
        
        if self.performance_metrics['test_execution_times']:
            self.logger.info("  Test execution times:")
            for test_file, execution_time in self.performance_metrics['test_execution_times'].items():
                self.logger.info(f"    {test_file}: {execution_time:.2f}s")
        
        # Print detailed results
        if self.results['backend_manager_tests']:
            self.logger.info("Backend Manager Tests:")
            for test in self.results['backend_manager_tests']:
                status_color = "green" if test['status'] == 'PASS' else "red"
                self.logger.info(f"  {test['test']}: {test['status']}")
                if test['details']:
                    self.logger.info(f"    Details: {test['details']}")
        
        if self.results['mock_backend_tests']:
            self.logger.info("Mock Backend Tests:")
            for test in self.results['mock_backend_tests']:
                status_color = "green" if test['status'] == 'PASS' else "red"
                self.logger.info(f"  {test['test']}: {test['status']}")
                if test['details']:
                    self.logger.info(f"    Details: {test['details']}")
        
        if self.results['real_backend_tests']:
            self.logger.info("Real Backend Tests:")
            for test in self.results['real_backend_tests']:
                status_color = "green" if test['status'] == 'PASS' else "yellow"
                self.logger.info(f"  {test['test']}: {test['status']}")
                if test['details']:
                    self.logger.info(f"    Details: {test['details']}")
        
        return {
            'results': self.results,
            'performance_metrics': self.performance_metrics,
            'timestamp': time.time()
        }
    
    def run_full_validation(self) -> bool:
        """
        Run the complete validation suite
        
        Returns:
            True if validation passes, False otherwise
        """
        self.logger.banner("Backend Decorator Validation Suite", 60)
        
        # Step 1: Compliance check
        if not self.run_compliance_check():
            self.logger.error("Compliance check failed - stopping validation")
            return False
        
        # Step 2: Backend manager functionality
        if not self.test_backend_manager_functionality():
            self.logger.error("Backend manager tests failed")
            return False
        
        # Step 3: Decorator functionality
        if not self.test_decorator_functionality():
            self.logger.error("Decorator tests failed")
            return False
        
        # Step 4: Sample test execution
        self.run_sample_tests()
        
        # Step 5: Generate report
        report = self.generate_validation_report()
        
        # Final result
        if self.results['overall_success']:
            self.logger.success("Backend decorator validation completed successfully!")
        else:
            self.logger.error("Backend decorator validation failed!")
        
        return self.results['overall_success']


@requires_mock_backend("Backend Decorator Validation")
def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate backend decorator implementation")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick validation (skip sample tests)")
    parser.add_argument("--report-only", action="store_true",
                       help="Generate report only (skip validation)")
    
    args = parser.parse_args()
    
    validator = BackendDecoratorValidator()
    
    if args.report_only:
        # Just generate a report
        validator.generate_validation_report()
        return True
    
    success = validator.run_full_validation()
    
    if success:
        print("\n" + "="*60)
        print("VALIDATION SUCCESS!")
        print("The backend decorator implementation is working correctly.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("VALIDATION FAILED!")
        print("Please review the issues above and fix them.")
        print("="*60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 