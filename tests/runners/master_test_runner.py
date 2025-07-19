#!/usr/bin/env python3
"""
Master Test Runner for 2048 AI Checkpoint System
===============================================

This is the main test runner that implements the 4 test levels as outlined in the README:
- basic: Quick connectivity and API tests (~1 minute)
- core: Checkpoint loading and game playback (~5 minutes)  
- full: All core functionality plus live playback (~10 minutes)
- comprehensive: All tests including performance and edge cases (~15 minutes)

Usage:
    python tests/runners/master_test_runner.py --level basic
    python tests/runners/master_test_runner.py --level core
    python tests/runners/master_test_runner.py --level full
    python tests/runners/master_test_runner.py --level comprehensive
"""

import sys
import os
import time
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.utilities.test_utils import TestLogger
from tests.utilities.backend_manager import requires_mock_backend

class TestLevel(Enum):
    """Test levels as defined in the README"""
    BASIC = "basic"
    CORE = "core"
    FULL = "full"
    COMPREHENSIVE = "comprehensive"

@dataclass
class TestResult:
    """Result of a test execution"""
    name: str
    success: bool
    duration: float
    output: str
    error: Optional[str] = None

class MasterTestRunner:
    """Master test runner implementing the 4 test levels"""
    
    def __init__(self):
        self.logger = TestLogger()
        self.tests_dir = Path(__file__).parent.parent
        self.results: List[TestResult] = []
        self.start_time = None
        
        # Define test suites for each level
        self.test_suites = {
            TestLevel.BASIC: {
                "description": "Quick connectivity and API tests (~1 minute)",
                "tests": [
                    "tests/integration/test_checkpoint_loading.py",
                    "tests/integration/test_mock_backend_basic.py",
                ]
            },
            TestLevel.CORE: {
                "description": "Checkpoint loading and game playback (~5 minutes)",
                "tests": [
                    "tests/integration/test_checkpoint_loading.py",
                    "tests/integration/test_complete_games.py",
                    "tests/core/test_checkpoint_loading.py",
                    "tests/core/test_json_serialization.py",
                ]
            },
            TestLevel.FULL: {
                "description": "All core functionality plus live playback (~10 minutes)",
                "tests": [
                    "tests/integration/test_checkpoint_loading.py",
                    "tests/integration/test_complete_games.py",
                    "tests/integration/test_live_playback.py",
                    "tests/integration/test_game_simulation.py",
                    "tests/integration/test_model_studio.py",
                    "tests/integration/test_backend_parity.py",
                    "tests/core/test_checkpoint_loading.py",
                    "tests/core/test_json_serialization.py",
                    "tests/playback/test_controls.py",
                    "tests/playback/test_simulation.py",
                    "tests/frontend/test_pwa_origin_unification.py",
                ]
            },
            TestLevel.COMPREHENSIVE: {
                "description": "All tests including performance and edge cases (~15 minutes)",
                "tests": [
                    # Integration tests - Core functionality
                    "tests/integration/test_checkpoint_loading.py",
                    "tests/integration/test_complete_games.py",
                    "tests/integration/test_live_playback.py",
                    "tests/integration/test_game_simulation.py",
                    "tests/integration/test_edge_cases.py",
                    "tests/integration/test_websocket_broadcast.py",
                    "tests/integration/test_model_studio.py",
                    "tests/integration/test_tunnel_demo.py",
                    "tests/integration/test_mock_backend.py",
                    "tests/integration/test_connection_stability.py",
                    "tests/integration/test_checkpoint_loading_issue.py",
                    "tests/integration/test_backend_parity.py",
                    
                    # Core tests - Backend functionality
                    "tests/core/test_checkpoint_loading.py",
                    "tests/core/test_json_serialization.py",
                    "tests/core/test_training_manager.py",
                    "tests/core/test_training_issue.py",
                    "tests/core/test_training_fix.py",
                    "tests/core/test_tiny_model.py",
                    "tests/core/test_minimal_crash.py",
                    "tests/core/test_json_serialization_fix.py",
                    "tests/core/test_checkpoint_loading_verification.py",
                    "tests/core/test_checkpoint_loading_fixes.py",
                    
                    # Playback tests - Game playback functionality
                    "tests/playback/test_controls.py",
                    "tests/playback/test_simulation.py",
                    "tests/playback/test_freeze_detection.py",
                    "tests/playback/test_freeze_diagnostics.py",
                    "tests/playback/test_freeze_reproduction.py",
                    "tests/playback/test_failure_simple.py",
                    "tests/playback/test_failure_comprehensive.py",
                    "tests/playback/test_checkpoint_failure.py",
                    "tests/playback/test_failure_final.py",
                    "tests/playback/test_playback_failure.py",
                    
                    # Performance tests - System performance
                    "tests/performance/test_performance.py",
                    "tests/performance/test_speed_control.py",
                    "tests/performance/test_training_speed.py",
                    "tests/performance/test_load_balancing.py",
                    "tests/performance/test_gpu_usage.py",
                    
                    # Mobile tests - Device compatibility
                    "tests/mobile/test_connection_issues.py",
                    "tests/mobile/test_device_compatibility.py",
                    "tests/mobile/test_device_fix.py",
                    "tests/mobile/test_device_error.py",
                    "tests/mobile/test_comprehensive_device.py",
                    "tests/mobile/test_mobile_basic.py",
                    "tests/mobile/test_training_disconnection.py",
                    
                    # Training tests - Training functionality
                    "tests/training/test_status_sync.py",
                    "tests/training/test_status_sync_simple.py",
                    "tests/training/test_reconnection.py",
                    "tests/training/test_reconnection_failure.py",
                    
                    # Frontend tests - Frontend functionality
                    "tests/frontend/test_automation.py",
                    "tests/frontend/test_pwa_origin_unification.py",
                    "tests/frontend/test_pwa_install.py",
                    "tests/frontend/test_persistence_fix.py",
                    "tests/frontend/test_browser_enhanced.py",
                    "tests/frontend/test_browser_simulation.py",
                ]
            }
        }
    
    def run_test(self, test_path: str, timeout: int = 300) -> TestResult:
        """Run a single test file with timeout"""
        test_file = Path(test_path)
        
        if not test_file.exists():
            return TestResult(
                name=test_path,
                success=False,
                duration=0.0,
                output="",
                error=f"Test file not found: {test_path}"
            )
        
        self.logger.log(f"Running: {test_path}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.tests_dir.parent  # Run from project root
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.logger.ok(f"✅ {test_path} PASSED ({duration:.1f}s)")
                return TestResult(
                    name=test_path,
                    success=True,
                    duration=duration,
                    output=result.stdout
                )
            else:
                self.logger.error(f"❌ {test_path} FAILED ({duration:.1f}s)")
                return TestResult(
                    name=test_path,
                    success=False,
                    duration=duration,
                    output=result.stdout,
                    error=result.stderr
                )
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            self.logger.error(f"⏰ {test_path} TIMEOUT after {timeout}s")
            return TestResult(
                name=test_path,
                success=False,
                duration=duration,
                output="",
                error=f"Test timed out after {timeout} seconds"
            )
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"💥 {test_path} ERROR: {e}")
            return TestResult(
                name=test_path,
                success=False,
                duration=duration,
                output="",
                error=str(e)
            )
    
    def run_level(self, level: TestLevel) -> bool:
        """Run tests for a specific level"""
        suite = self.test_suites[level]
        
        self.logger.banner(f"2048 AI Test Suite - {level.value.upper()} Level", 80)
        self.logger.log(f"Description: {suite['description']}")
        self.logger.log(f"Tests to run: {len(suite['tests'])}")
        self.logger.separator(80)
        
        self.start_time = time.time()
        self.results = []
        
        # Run each test in the suite
        for test_path in suite['tests']:
            result = self.run_test(test_path)
            self.results.append(result)
            
            # Add some spacing between tests
            self.logger.log("")
        
        # Print summary
        self.print_summary(level)
        
        return all(result.success for result in self.results)
    
    def print_summary(self, level: TestLevel):
        """Print test results summary"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        self.logger.separator(80)
        self.logger.banner("TEST RESULTS SUMMARY", 80)
        
        passed = sum(1 for result in self.results if result.success)
        failed = sum(1 for result in self.results if not result.success)
        total = len(self.results)
        
        self.logger.log(f"Level: {level.value.upper()}")
        self.logger.log(f"Total tests: {total}")
        self.logger.log(f"Passed: {passed}")
        self.logger.log(f"Failed: {failed}")
        self.logger.log(f"Total time: {total_time:.1f}s")
        self.logger.log("")
        
        if failed > 0:
            self.logger.error("FAILED TESTS:")
            for result in self.results:
                if not result.success:
                    self.logger.error(f"  {result.name}: {result.error or 'Unknown error'}")
            self.logger.log("")
        
        if passed == total:
            self.logger.success("🎉 ALL TESTS PASSED!")
        else:
            self.logger.error(f"💥 SOME TESTS FAILED! ({passed}/{total} passed)")
        
        self.logger.separator(80)
    
    def list_levels(self):
        """List all available test levels with descriptions"""
        self.logger.banner("Available Test Levels", 60)
        
        for level in TestLevel:
            suite = self.test_suites[level]
            self.logger.log(f"\n{level.value.upper()}:")
            self.logger.log(f"  {suite['description']}")
            self.logger.log(f"  Tests: {len(suite['tests'])}")
            self.logger.log("  Files:")
            for test in suite['tests']:
                self.logger.log(f"    - {test}")

@requires_mock_backend("Master Test Runner")
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Master test runner for 2048 AI checkpoint system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/runners/master_test_runner.py --level basic
  python tests/runners/master_test_runner.py --level core
  python tests/runners/master_test_runner.py --level full
  python tests/runners/master_test_runner.py --level comprehensive
  python tests/runners/master_test_runner.py --list
        """
    )
    
    parser.add_argument(
        '--level', 
        choices=[level.value for level in TestLevel],
        help='Test level to run'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available test levels'
    )
    
    args = parser.parse_args()
    
    runner = MasterTestRunner()
    
    if args.list:
        runner.list_levels()
        return
    
    if not args.level:
        parser.print_help()
        return
    
    try:
        level = TestLevel(args.level)
        success = runner.run_level(level)
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        runner.logger.error("\nTest run interrupted by user")
        sys.exit(1)
    except Exception as e:
        runner.logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 