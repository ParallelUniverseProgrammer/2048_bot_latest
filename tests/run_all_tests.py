#!/usr/bin/env python3
"""
Master test runner for 2048 AI Bot freeze diagnostics
Runs all tests in the correct order and provides comprehensive testing

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

class TestRunner:
    """Master test runner for all freeze diagnostic tests"""
    
    def __init__(self):
        self.tests_dir = Path(__file__).parent
        self.results = {}
        self.start_time = None
        
    def run_test(self, test_name: str, timeout: int = 300) -> bool:
        """Run a single test file with timeout"""
        test_path = self.tests_dir / test_name
        
        if not test_path.exists():
            print(f"[SKIP] {test_name} - File not found")
            return True
        
        print(f"============================================================")
        print(f"Running: {test_name}")
        print(f"============================================================")
        
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
                print(f"[OK] {test_name} PASSED ({duration:.1f}s)")
                if result.stdout:
                    print("STDOUT:")
                    print(result.stdout)
                self.results[test_name] = 'PASSED'
                return True
            else:
                print(f"[FAIL] {test_name} FAILED ({duration:.1f}s)")
                print(f"Exit code: {result.returncode}")
                if result.stdout:
                    print("STDOUT:")
                    print(result.stdout)
                if result.stderr:
                    print("STDERR:")
                    print(result.stderr)
                self.results[test_name] = 'FAILED'
                return False
                
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {test_name} timed out after {timeout}s")
            self.results[test_name] = 'TIMEOUT'
            return False
        except Exception as e:
            print(f"[ERROR] {test_name} error: {e}")
            self.results[test_name] = 'ERROR'
            return False
    
    def run_quick_tests(self) -> bool:
        """Run quick diagnostic tests"""
        print("=" * 80)
        print("2048 AI Bot - Quick Freeze Diagnostic Tests")
        print("=" * 80)
        
        quick_tests = [
            'test_checkpoint_loading.py',
            'test_game_simulation.py',
        ]
        
        success = True
        
        for test_name in quick_tests:
            print(f"\n{test_name.replace('_', ' ').replace('.py', '').title()} Tests")
            print()
            self.start_time = time.time()
            success &= self.run_test(test_name, timeout=120)
        
        return success
    
    def run_comprehensive_tests(self) -> bool:
        """Run comprehensive diagnostic tests"""
        print("=" * 80)
        print("2048 AI Bot - Comprehensive Freeze Diagnostic Tests")
        print("=" * 80)
        
        comprehensive_tests = [
            'test_checkpoint_loading.py',
            'test_game_simulation.py',
            'test_live_playback.py',
            'test_freeze_diagnostics.py',
        ]
        
        success = True
        
        for test_name in comprehensive_tests:
            print(f"\n{test_name.replace('_', ' ').replace('.py', '').title()} Tests")
            print()
            self.start_time = time.time()
            success &= self.run_test(test_name, timeout=600)
        
        return success
    
    def run_real_system_tests(self) -> bool:
        """Run tests against the real system (requires backend running)"""
        print("=" * 80)
        print("2048 AI Bot - Real System Diagnostic Tests")
        print("=" * 80)
        
        real_system_tests = [
            'test_freeze_diagnostics.py',
        ]
        
        success = True
        
        for test_name in real_system_tests:
            print(f"\n{test_name.replace('_', ' ').replace('.py', '').title()} Tests")
            print()
            self.start_time = time.time()
            success &= self.run_test(test_name, timeout=1200)
        
        return success
    
    def print_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for result in self.results.values() if result == 'PASSED')
        failed = sum(1 for result in self.results.values() if result == 'FAILED')
        timeout = sum(1 for result in self.results.values() if result == 'TIMEOUT')
        error = sum(1 for result in self.results.values() if result == 'ERROR')
        total = len(self.results)
        
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Timeout: {timeout}")
        print(f"Error: {error}")
        print()
        
        if failed > 0 or timeout > 0 or error > 0:
            print("FAILED TESTS:")
            for test_name, result in self.results.items():
                if result != 'PASSED':
                    print(f"  {test_name}: {result}")
            print()
        
        if passed == total:
            print("ALL TESTS PASSED!")
        else:
            print(f"SOME TESTS FAILED! ({passed}/{total} passed)")
        
        print(f"\nQuick test results: {passed} passed, {failed} failed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run 2048 AI Bot freeze diagnostic tests')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive tests')
    parser.add_argument('--real-system', action='store_true', help='Run real system tests (requires backend running)')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
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
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 