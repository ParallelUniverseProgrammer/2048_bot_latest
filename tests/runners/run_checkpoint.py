#!/usr/bin/env python3
"""
Checkpoint System Test Runner
============================

This script runs different levels of checkpoint system testing:

1. Basic Tests: Quick connectivity and API tests
2. Core Tests: Checkpoint loading and single game playback
3. Full Tests: Complete functionality including live playback
4. Comprehensive Tests: All tests including performance and error handling

Usage:
    python run_checkpoint_tests.py --level basic
    python run_checkpoint_tests.py --level core
    python run_checkpoint_tests.py --level full
    python run_checkpoint_tests.py --level comprehensive
"""

import sys
import os
import time
import argparse
import subprocess
from pathlib import Path
from test_utils import TestLogger

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_test_script(script_name: str, timeout: int = 300, logger: TestLogger = None) -> bool:
    """Run a test script and return success status"""
    if logger is None:
        logger = TestLogger()
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        logger.error(f"Test script not found: {script_name}")
        return False
    
    logger.running(f"Running: {script_name}")
    logger.separator(50)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            logger.ok(f"{script_name} PASSED")
            if result.stdout:
                logger.log("Output:")
                logger.log(result.stdout)
            return True
        else:
            logger.error(f"{script_name} FAILED")
            logger.log(f"Exit code: {result.returncode}")
            if result.stdout:
                logger.log("STDOUT:")
                logger.log(result.stdout)
            if result.stderr:
                logger.log("STDERR:")
                logger.log(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"{script_name} TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"{script_name} ERROR: {e}")
        return False

def run_basic_tests(logger: TestLogger = None) -> bool:
    """Run basic connectivity and API tests"""
    if logger is None:
        logger = TestLogger()
    
    logger.starting("Running Basic Checkpoint Tests")
    logger.log("Testing: Backend connectivity, API endpoints, checkpoint listing")
    logger.separator(60)
    
    tests = [
        ("test_checkpoint_loading_issue.py", 60),
    ]
    
    success = True
    for test_name, timeout in tests:
        success &= run_test_script(test_name, timeout, logger)
    
    return success

def run_core_tests(logger: TestLogger = None) -> bool:
    """Run core functionality tests"""
    if logger is None:
        logger = TestLogger()
    
    logger.starting("Running Core Checkpoint Tests")
    logger.log("Testing: Checkpoint loading, single game playback, basic validation")
    logger.separator(60)
    
    tests = [
        ("test_checkpoint_loading_fix.py", 60),
        ("test_checkpoint_complete_games.py", 180),  # 3 minutes for game playback
    ]
    
    success = True
    for test_name, timeout in tests:
        success &= run_test_script(test_name, timeout, logger)
    
    return success

def run_full_tests(logger: TestLogger = None) -> bool:
    """Run full functionality tests including live playback"""
    if logger is None:
        logger = TestLogger()
    
    logger.starting("Running Full Checkpoint Tests")
    logger.log("Testing: All core functionality + live playback + controls")
    logger.separator(60)
    
    tests = [
        ("test_checkpoint_loading_fix.py", 60),
        ("test_checkpoint_complete_games.py", 180),
        ("test_live_playback.py", 300),  # 5 minutes for live playback
    ]
    
    success = True
    for test_name, timeout in tests:
        success &= run_test_script(test_name, timeout, logger)
    
    return success

def run_comprehensive_tests(logger: TestLogger = None) -> bool:
    """Run comprehensive tests including performance and error handling"""
    if logger is None:
        logger = TestLogger()
    
    logger.starting("Running Comprehensive Checkpoint Tests")
    logger.log("Testing: All functionality + performance + error handling + edge cases")
    logger.separator(60)
    
    tests = [
        ("test_checkpoint_loading_fix.py", 60),
        ("test_checkpoint_complete_games.py", 180),
        ("test_live_playback.py", 300),
        ("test_game_simulation.py", 120),
        ("test_real_playback_freeze.py", 240),
        ("test_frontend_automation.py", 120),
        ("test_edge_cases.py", 300),
    ]
    
    success = True
    for test_name, timeout in tests:
        success &= run_test_script(test_name, timeout, logger)
    
    return success

def check_backend_running(logger: TestLogger = None) -> bool:
    """Check if the backend server is running"""
    if logger is None:
        logger = TestLogger()
    
    try:
        import requests
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            logger.ok("Backend server is running")
            return True
        else:
            logger.error(f"Backend server returned HTTP {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Backend server connectivity failed: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Checkpoint System Test Runner')
    parser.add_argument('--level', choices=['basic', 'core', 'full', 'comprehensive'], 
                       default='core', help='Test level to run')
    parser.add_argument('--timeout', type=int, default=300, help='Default timeout in seconds')
    
    args = parser.parse_args()
    
    logger = TestLogger()
    
    logger.game("Checkpoint System Test Runner")
    logger.separator(60)
    
    # Check if backend is running
    logger.testing("Checking backend connectivity...")
    if not check_backend_running(logger):
        logger.error("Backend server is not running!")
        logger.log("Please start the backend server first:")
        logger.log("   cd backend")
        logger.log("   python main.py")
        sys.exit(1)
    
    # Run tests based on level
    start_time = time.time()
    
    if args.level == 'basic':
        success = run_basic_tests(logger)
    elif args.level == 'core':
        success = run_core_tests(logger)
    elif args.level == 'full':
        success = run_full_tests(logger)
    elif args.level == 'comprehensive':
        success = run_comprehensive_tests(logger)
    else:
        logger.error(f"Unknown test level: {args.level}")
        sys.exit(1)
    
    total_time = time.time() - start_time
    
    # Print summary
    logger.separator(60)
    logger.banner("TEST SUMMARY", 60)
    logger.log(f"Level: {args.level.upper()}")
    logger.log(f"Total Time: {total_time:.1f}s")
    
    if success:
        logger.ok("ALL TESTS PASSED!")
        logger.log("The checkpoint system is working correctly.")
        logger.log("Complete games can be played back from checkpoints.")
        sys.exit(0)
    else:
        logger.error("SOME TESTS FAILED!")
        logger.log("The checkpoint system needs attention.")
        logger.log("Check the output above for specific issues.")
        sys.exit(1)

if __name__ == "__main__":
    main() 