#!/usr/bin/env python3
"""
Simple test runner for working tests
"""

import subprocess
import sys
import time

def run_test(test_path):
    """Run a single test and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {test_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, test_path],
            capture_output=True,
            text=True,
            timeout=300
        )
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ PASSED ({duration:.1f}s)")
            print(result.stdout)
            return True
        else:
            print(f"❌ FAILED ({duration:.1f}s)")
            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT after 300s")
        return False
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False

def main():
    """Run all working tests"""
    # Tests that are marked as compliant and should work
    working_tests = [
        "tests/integration/test_checkpoint_loading.py",
        "tests/integration/test_connection_stability.py", 
        "tests/mobile/test_connection_issues.py",
        "tests/mobile/test_training_disconnection.py",
        "tests/playback/test_checkpoint_failure.py",
        "tests/playback/test_failure_comprehensive.py",
        "tests/playback/test_failure_simple.py",
        "tests/playback/test_playback_failure.py",
        "tests/training/test_reconnection.py",
        "tests/training/test_reconnection_failure.py",
        "tests/training/test_status_sync.py",
    ]
    
    print("2048 AI Test Suite - Working Tests")
    print("="*60)
    print(f"Running {len(working_tests)} tests...")
    
    passed = 0
    failed = 0
    
    for test_path in working_tests:
        if run_test(test_path):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"💥 {failed} TESTS FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 