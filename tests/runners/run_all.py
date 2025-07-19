#!/usr/bin/env python3
"""
Master Test Runner for 2048 AI Bot
==================================

This script provides a comprehensive test runner for all freeze diagnostic tests.
It runs tests in the correct order and provides comprehensive testing coverage.

The runner supports different test levels:
- Quick tests: Basic functionality tests
- Comprehensive tests: Full diagnostic suite
- Real system tests: Tests against running backend

Usage:
    python tests/runners/run_all.py --quick
    python tests/runners/run_all.py --comprehensive
    python tests/runners/run_all.py --real-system
"""

import sys
import os
import asyncio
import time
import subprocess
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Add backend to path for internal imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from tests.utilities.test_utils import TestLogger
from tests.utilities.backend_manager import requires_mock_backend

class TestRunner:
    """Master test runner for all freeze diagnostic tests"""
    
    def __init__(self):
        self.logger = TestLogger()
        self.tests_dir = Path(__file__).parent
        self.results = {}
        self.start_time = None
        
    def run_test(self, test_name: str, timeout: int = 300) -> bool:
        """Run a single test file with timeout"""
        test_path = self.tests_dir / test_name
        
        if not test_path.exists():
            self.logger.info(f"SKIP: {test_name} - File not found")
            return True
        
        self.logger.banner(f"Running: {test_name}", 60)
        
        try:
            result = subprocess.run(
                [sys.executable, str(test_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.tests_dir.parent
            )
            
            duration = time.time() - self.start_time if self.start_time else 0
            
            if result.returncode == 0:
                self.logger.ok(f"{test_name} PASSED ({duration:.1f}s)")
                if result.stdout:
                    self.logger.info("STDOUT:")
                    self.logger.info(result.stdout)
                self.results[test_name] = 'PASSED'
                return True
            else:
                self.logger.error(f"{test_name} FAILED ({duration:.1f}s)")
                self.logger.error(f"Exit code: {result.returncode}")
                if result.stdout:
                    self.logger.info("STDOUT:")
                    self.logger.info(result.stdout)
                if result.stderr:
                    self.logger.error("STDERR:")
                    self.logger.error(result.stderr)
                self.results[test_name] = 'FAILED'
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"{test_name} timed out after {timeout}s")
            self.results[test_name] = 'TIMEOUT'
            return False
        except Exception as e:
            self.logger.error(f"{test_name} error: {e}")
            self.results[test_name] = 'ERROR'
            return False
    
    def run_quick_tests(self) -> bool:
        """Run quick diagnostic tests"""
        self.logger.banner("2048 AI Bot - Quick Freeze Diagnostic Tests", 80)
        
        quick_tests = [
            'test_checkpoint_loading.py',
            'test_game_simulation.py',
        ]
        
        success = True
        
        for test_name in quick_tests:
            self.logger.info(f"{test_name.replace('_', ' ').replace('.py', '').title()} Tests")
            self.start_time = time.time()
            success &= self.run_test(test_name, timeout=120)
        
        return success
    
    def run_comprehensive_tests(self) -> bool:
        """Run comprehensive diagnostic tests"""
        self.logger.banner("2048 AI Bot - Comprehensive Freeze Diagnostic Tests", 80)
        
        comprehensive_tests = [
            'test_checkpoint_loading.py',
            'test_game_simulation.py',
            'test_live_playback.py',
            'test_freeze_diagnostics.py',
        ]
        
        success = True
        
        for test_name in comprehensive_tests:
            self.logger.info(f"{test_name.replace('_', ' ').replace('.py', '').title()} Tests")
            self.start_time = time.time()
            success &= self.run_test(test_name, timeout=600)
        
        return success
    
    def run_real_system_tests(self) -> bool:
        """Run tests against the real system (requires backend running)"""
        self.logger.banner("2048 AI Bot - Real System Diagnostic Tests", 80)
        
        real_system_tests = [
            'test_freeze_diagnostics.py',
        ]
        
        success = True
        
        for test_name in real_system_tests:
            self.logger.info(f"{test_name.replace('_', ' ').replace('.py', '').title()} Tests")
            self.start_time = time.time()
            success &= self.run_test(test_name, timeout=1200)
        
        return success
    
    def print_summary(self):
        """Print test results summary"""
        self.logger.separator()
        self.logger.info("TEST RESULTS SUMMARY")
        self.logger.separator()
        
        passed = sum(1 for result in self.results.values() if result == 'PASSED')
        failed = sum(1 for result in self.results.values() if result == 'FAILED')
        timeout = sum(1 for result in self.results.values() if result == 'TIMEOUT')
        error = sum(1 for result in self.results.values() if result == 'ERROR')
        total = len(self.results)
        
        self.logger.info(f"Total tests: {total}")
        self.logger.info(f"Passed: {passed}")
        self.logger.error(f"Failed: {failed}")
        self.logger.warning(f"Timeout: {timeout}")
        self.logger.error(f"Error: {error}")
        
        if failed > 0 or timeout > 0 or error > 0:
            self.logger.error("FAILED TESTS:")
            for test_name, result in self.results.items():
                if result != 'PASSED':
                    self.logger.error(f"  {test_name}: {result}")
        
        if passed == total:
            self.logger.success("ALL TESTS PASSED!")
        else:
            self.logger.error(f"SOME TESTS FAILED! ({passed}/{total} passed)")
        
        self.logger.info(f"Quick test results: {passed} passed, {failed} failed")
@requires_mock_backend("Test Runner")

def main():
    """Main entry point"""
    logger = TestLogger()
    logger.banner("2048 AI Bot Test Runner", 60)
    
    parser = argparse.ArgumentParser(description='Run 2048 AI Bot freeze diagnostic tests')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive tests')
    parser.add_argument('--real-system', action='store_true', help='Run real system tests (requires backend running)')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    try:
        if args.quick:
            success = runner.run_quick_tests()
        elif args.comprehensive:
            success = runner.run_comprehensive_tests()
        elif args.real_system:
            success = runner.run_real_system_tests()
        else:
            # Default to quick tests
            success = runner.run_quick_tests()
        
        runner.print_summary()
        
        if success:
            logger.success("Test runner completed successfully")
        else:
            logger.error("Test runner completed with failures")
        
        return success
        
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 