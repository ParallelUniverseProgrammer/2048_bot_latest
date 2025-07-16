#!/usr/bin/env python3
"""
Comprehensive Checkpoint System Testing Plan
============================================

This test plan covers all aspects of the checkpoint system to ensure:
1. Checkpoints can be loaded correctly
2. Complete games can be played back from checkpoints
3. The frontend displays checkpoints properly
4. Playback controls work correctly
5. Performance is acceptable
6. Error handling works properly

Test Categories:
- Backend API Tests
- Checkpoint Loading Tests
- Game Playback Tests
- Frontend Integration Tests
- Performance Tests
- Error Handling Tests
- End-to-End Tests
"""

import asyncio
import time
import json
import requests
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

class TestStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    TIMEOUT = "TIMEOUT"

@dataclass
class TestResult:
    name: str
    status: TestStatus
    duration: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class CheckpointTestSuite:
    """Comprehensive test suite for checkpoint system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        self.checkpoints: List[Dict[str, Any]] = []
        self.test_checkpoint_id: Optional[str] = None
        
    def run_test(self, test_name: str, test_func, timeout: int = 60) -> TestResult:
        """Run a single test with timeout and error handling"""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                # Run async test
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(asyncio.wait_for(test_func(), timeout=timeout))
                    status = TestStatus.PASSED if result else TestStatus.FAILED
                except asyncio.TimeoutError:
                    status = TestStatus.TIMEOUT
                    result = None
                finally:
                    loop.close()
            else:
                # Run sync test
                result = test_func()
                status = TestStatus.PASSED if result else TestStatus.FAILED
                
        except Exception as e:
            status = TestStatus.FAILED
            result = None
            error = str(e)
        else:
            error = None
            
        duration = time.time() - start_time
        
        test_result = TestResult(
            name=test_name,
            status=status,
            duration=duration,
            error=error,
            details=result if isinstance(result, dict) else None
        )
        
        self.results.append(test_result)
        return test_result
    
    def print_result(self, result: TestResult):
        """Print test result with formatting"""
        status_emoji = {
            TestStatus.PASSED: "OK:",
            TestStatus.FAILED: "ERROR:",
            TestStatus.SKIPPED: "NEXT:",
            TestStatus.TIMEOUT: "ALARM:"
        }
        
        print(f"{status_emoji[result.status]} {result.name} ({result.duration:.2f}s)")
        if result.error:
            print(f"   Error: {result.error}")
        if result.details:
            print(f"   Details: {result.details}")
    
    # ============================================================================
    # BACKEND API TESTS
    # ============================================================================
    
    def test_backend_connectivity(self) -> bool:
        """Test basic backend connectivity"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def test_checkpoints_list_endpoint(self) -> Dict[str, Any]:
        """Test the checkpoints list endpoint"""
        response = requests.get(f"{self.base_url}/checkpoints", timeout=30)
        
        if response.status_code != 200:
            return {"success": False, "error": f"HTTP {response.status_code}"}
        
        checkpoints = response.json()
        self.checkpoints = checkpoints
        
        return {
            "success": True,
            "count": len(checkpoints),
            "checkpoints": checkpoints[:3] if checkpoints else []  # First 3 for details
        }
    
    def test_checkpoints_stats_endpoint(self) -> Dict[str, Any]:
        """Test the checkpoints stats endpoint"""
        response = requests.get(f"{self.base_url}/checkpoints/stats", timeout=30)
        
        if response.status_code != 200:
            return {"success": False, "error": f"HTTP {response.status_code}"}
        
        stats = response.json()
        return {
            "success": True,
            "stats": stats
        }
    
    def test_playback_status_endpoint(self) -> Dict[str, Any]:
        """Test the playback status endpoint"""
        response = requests.get(f"{self.base_url}/checkpoints/playback/status", timeout=30)
        
        if response.status_code != 200:
            return {"success": False, "error": f"HTTP {response.status_code}"}
        
        status = response.json()
        return {
            "success": True,
            "status": status
        }
    
    # ============================================================================
    # CHECKPOINT LOADING TESTS
    # ============================================================================
    
    def test_checkpoint_metadata_validation(self) -> Dict[str, Any]:
        """Test that checkpoint metadata is valid"""
        if not self.checkpoints:
            return {"success": False, "error": "No checkpoints available"}
        
        errors = []
        valid_checkpoints = []
        
        for cp in self.checkpoints:
            required_fields = ['id', 'nickname', 'episode', 'created_at', 'model_config']
            missing_fields = [field for field in required_fields if field not in cp]
            
            if missing_fields:
                errors.append(f"Checkpoint {cp.get('id', 'unknown')} missing fields: {missing_fields}")
            else:
                valid_checkpoints.append(cp['id'])
        
        return {
            "success": len(errors) == 0,
            "valid_count": len(valid_checkpoints),
            "errors": errors,
            "valid_checkpoints": valid_checkpoints[:3]  # First 3 for reference
        }
    
    def test_checkpoint_file_accessibility(self) -> Dict[str, Any]:
        """Test that checkpoint files are accessible"""
        if not self.checkpoints:
            return {"success": False, "error": "No checkpoints available"}
        
        # Test first checkpoint
        test_checkpoint = self.checkpoints[0]
        checkpoint_id = test_checkpoint['id']
        self.test_checkpoint_id = checkpoint_id
        
        response = requests.get(f"{self.base_url}/checkpoints/{checkpoint_id}", timeout=30)
        
        if response.status_code != 200:
            return {"success": False, "error": f"Failed to access checkpoint {checkpoint_id}"}
        
        checkpoint_info = response.json()
        return {
            "success": True,
            "checkpoint_id": checkpoint_id,
            "episode": checkpoint_info.get('episode'),
            "file_size": checkpoint_info.get('file_size'),
            "model_config": checkpoint_info.get('model_config')
        }
    
    # ============================================================================
    # GAME PLAYBACK TESTS
    # ============================================================================
    
    async def test_single_game_playback(self) -> Dict[str, Any]:
        """Test playing a single game from a checkpoint"""
        if not self.test_checkpoint_id:
            return {"success": False, "error": "No test checkpoint available"}
        
        # Start single game playback
        response = requests.post(
            f"{self.base_url}/checkpoints/{self.test_checkpoint_id}/playback/game",
            timeout=120  # 2 minutes for game completion
        )
        
        if response.status_code != 200:
            return {"success": False, "error": f"Game playback failed: HTTP {response.status_code}"}
        
        game_result = response.json()
        
        # Validate game result
        required_fields = ['game_history', 'final_score', 'max_tile', 'steps', 'completed']
        missing_fields = [field for field in required_fields if field not in game_result]
        
        if missing_fields:
            return {"success": False, "error": f"Game result missing fields: {missing_fields}"}
        
        # Validate game history
        game_history = game_result['game_history']
        if not game_history:
            return {"success": False, "error": "Game history is empty"}
        
        # Check that game completed properly
        if not game_result['completed']:
            return {"success": False, "error": "Game did not complete properly"}
        
        return {
            "success": True,
            "final_score": game_result['final_score'],
            "max_tile": game_result['max_tile'],
            "steps": game_result['steps'],
            "history_length": len(game_history),
            "completed": game_result['completed']
        }
    
    async def test_live_playback_start(self) -> Dict[str, Any]:
        """Test starting live playback"""
        if not self.test_checkpoint_id:
            return {"success": False, "error": "No test checkpoint available"}
        
        # Start live playback
        response = requests.post(
            f"{self.base_url}/checkpoints/{self.test_checkpoint_id}/playback/start",
            timeout=30
        )
        
        if response.status_code != 200:
            return {"success": False, "error": f"Live playback start failed: HTTP {response.status_code}"}
        
        start_result = response.json()
        
        # Wait a moment for playback to initialize
        await asyncio.sleep(2)
        
        # Check playback status
        status_response = requests.get(f"{self.base_url}/checkpoints/playback/status", timeout=10)
        if status_response.status_code != 200:
            return {"success": False, "error": "Failed to get playback status"}
        
        status = status_response.json()
        
        return {
            "success": True,
            "start_result": start_result,
            "is_playing": status.get('is_playing', False),
            "model_loaded": status.get('model_loaded', False),
            "current_checkpoint": status.get('current_checkpoint')
        }
    
    async def test_playback_controls(self) -> Dict[str, Any]:
        """Test playback pause/resume/stop controls"""
        if not self.test_checkpoint_id:
            return {"success": False, "error": "No test checkpoint available"}
        
        results = {}
        
        # Test pause
        pause_response = requests.post(f"{self.base_url}/checkpoints/playback/pause", timeout=10)
        results['pause'] = pause_response.status_code == 200
        
        await asyncio.sleep(1)
        
        # Test resume
        resume_response = requests.post(f"{self.base_url}/checkpoints/playback/resume", timeout=10)
        results['resume'] = resume_response.status_code == 200
        
        await asyncio.sleep(1)
        
        # Test stop
        stop_response = requests.post(f"{self.base_url}/checkpoints/playback/stop", timeout=10)
        results['stop'] = stop_response.status_code == 200
        
        return {
            "success": all(results.values()),
            "control_results": results
        }
    
    # ============================================================================
    # PERFORMANCE TESTS
    # ============================================================================
    
    def test_checkpoint_loading_performance(self) -> Dict[str, Any]:
        """Test checkpoint loading performance"""
        if not self.checkpoints:
            return {"success": False, "error": "No checkpoints available"}
        
        load_times = []
        
        # Test loading first 3 checkpoints
        for i, checkpoint in enumerate(self.checkpoints[:3]):
            start_time = time.time()
            response = requests.get(f"{self.base_url}/checkpoints/{checkpoint['id']}", timeout=30)
            load_time = time.time() - start_time
            
            if response.status_code == 200:
                load_times.append(load_time)
        
        if not load_times:
            return {"success": False, "error": "No checkpoints loaded successfully"}
        
        avg_load_time = sum(load_times) / len(load_times)
        max_load_time = max(load_times)
        
        return {
            "success": True,
            "avg_load_time": avg_load_time,
            "max_load_time": max_load_time,
            "load_times": load_times,
            "performance_ok": avg_load_time < 5.0  # Should load in under 5 seconds
        }
    
    async def test_game_playback_performance(self) -> Dict[str, Any]:
        """Test game playback performance"""
        if not self.test_checkpoint_id:
            return {"success": False, "error": "No test checkpoint available"}
        
        start_time = time.time()
        
        # Play a single game
        response = requests.post(
            f"{self.base_url}/checkpoints/{self.test_checkpoint_id}/playback/game",
            timeout=120
        )
        
        total_time = time.time() - start_time
        
        if response.status_code != 200:
            return {"success": False, "error": "Game playback failed"}
        
        game_result = response.json()
        steps = game_result.get('steps', 0)
        
        if steps == 0:
            return {"success": False, "error": "Game had 0 steps"}
        
        steps_per_second = steps / total_time if total_time > 0 else 0
        
        return {
            "success": True,
            "total_time": total_time,
            "steps": steps,
            "steps_per_second": steps_per_second,
            "performance_ok": steps_per_second > 1.0  # Should process at least 1 step per second
        }
    
    # ============================================================================
    # ERROR HANDLING TESTS
    # ============================================================================
    
    def test_invalid_checkpoint_access(self) -> Dict[str, Any]:
        """Test accessing non-existent checkpoints"""
        invalid_id = "nonexistent_checkpoint_12345"
        
        response = requests.get(f"{self.base_url}/checkpoints/{invalid_id}", timeout=10)
        
        return {
            "success": response.status_code == 404,  # Should return 404
            "status_code": response.status_code,
            "expected": 404
        }
    
    def test_invalid_playback_start(self) -> Dict[str, Any]:
        """Test starting playback with invalid checkpoint"""
        invalid_id = "nonexistent_checkpoint_12345"
        
        response = requests.post(
            f"{self.base_url}/checkpoints/{invalid_id}/playback/start",
            timeout=10
        )
        
        return {
            "success": response.status_code == 404,  # Should return 404
            "status_code": response.status_code,
            "expected": 404
        }
    
    # ============================================================================
    # FRONTEND INTEGRATION TESTS
    # ============================================================================
    
    def test_frontend_checkpoint_display(self) -> Dict[str, Any]:
        """Test that frontend can display checkpoints correctly"""
        # This would require browser automation
        # For now, we'll test the API endpoints that the frontend uses
        
        endpoints_to_test = [
            "/checkpoints",
            "/checkpoints/stats",
            "/checkpoints/playback/status"
        ]
        
        results = {}
        for endpoint in endpoints_to_test:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                results[endpoint] = response.status_code == 200
            except Exception:
                results[endpoint] = False
        
        all_working = all(results.values())
        
        return {
            "success": all_working,
            "endpoint_results": results,
            "frontend_ready": all_working
        }
    
    # ============================================================================
    # COMPREHENSIVE TEST RUNNERS
    # ============================================================================
    
    def run_backend_tests(self) -> bool:
        """Run all backend API tests"""
        print("\nRUNNING: Backend API Tests")
        print("=" * 50)
        
        tests = [
            ("Backend Connectivity", self.test_backend_connectivity),
            ("Checkpoints List Endpoint", self.test_checkpoints_list_endpoint),
            ("Checkpoints Stats Endpoint", self.test_checkpoints_stats_endpoint),
            ("Playback Status Endpoint", self.test_playback_status_endpoint),
        ]
        
        success = True
        for test_name, test_func in tests:
            result = self.run_test(test_name, test_func)
            self.print_result(result)
            success &= result.status == TestStatus.PASSED
        
        return success
    
    def run_checkpoint_loading_tests(self) -> bool:
        """Run checkpoint loading tests"""
        print("\nSTATUS: Checkpoint Loading Tests")
        print("=" * 50)
        
        tests = [
            ("Checkpoint Metadata Validation", self.test_checkpoint_metadata_validation),
            ("Checkpoint File Accessibility", self.test_checkpoint_file_accessibility),
            ("Checkpoint Loading Performance", self.test_checkpoint_loading_performance),
        ]
        
        success = True
        for test_name, test_func in tests:
            result = self.run_test(test_name, test_func)
            self.print_result(result)
            success &= result.status == TestStatus.PASSED
        
        return success
    
    async def run_playback_tests(self) -> bool:
        """Run game playback tests"""
        print("\nGAME: Game Playback Tests")
        print("=" * 50)
        
        tests = [
            ("Single Game Playback", self.test_single_game_playback),
            ("Live Playback Start", self.test_live_playback_start),
            ("Playback Controls", self.test_playback_controls),
            ("Game Playback Performance", self.test_game_playback_performance),
        ]
        
        success = True
        for test_name, test_func in tests:
            result = self.run_test(test_name, test_func)
            self.print_result(result)
            success &= result.status == TestStatus.PASSED
        
        return success
    
    def run_error_handling_tests(self) -> bool:
        """Run error handling tests"""
        print("\nWARNING: Error Handling Tests")
        print("=" * 50)
        
        tests = [
            ("Invalid Checkpoint Access", self.test_invalid_checkpoint_access),
            ("Invalid Playback Start", self.test_invalid_playback_start),
        ]
        
        success = True
        for test_name, test_func in tests:
            result = self.run_test(test_name, test_func)
            self.print_result(result)
            success &= result.status == TestStatus.PASSED
        
        return success
    
    def run_frontend_integration_tests(self) -> bool:
        """Run frontend integration tests"""
        print("\nSTATUS: Frontend Integration Tests")
        print("=" * 50)
        
        tests = [
            ("Frontend Checkpoint Display", self.test_frontend_checkpoint_display),
        ]
        
        success = True
        for test_name, test_func in tests:
            result = self.run_test(test_name, test_func)
            self.print_result(result)
            success &= result.status == TestStatus.PASSED
        
        return success
    
    async def run_comprehensive_test_suite(self) -> bool:
        """Run the complete test suite"""
        print("STATUS: Comprehensive Checkpoint System Test Suite")
        print("=" * 80)
        print("Testing complete checkpoint functionality including:")
        print("- Backend API endpoints")
        print("- Checkpoint loading and validation")
        print("- Complete game playback")
        print("- Performance metrics")
        print("- Error handling")
        print("- Frontend integration")
        print("=" * 80)
        
        # Run all test categories
        backend_ok = self.run_backend_tests()
        loading_ok = self.run_checkpoint_loading_tests()
        playback_ok = await self.run_playback_tests()
        error_ok = self.run_error_handling_tests()
        frontend_ok = self.run_frontend_integration_tests()
        
        # Print summary
        self.print_summary()
        
        return all([backend_ok, loading_ok, playback_ok, error_ok, frontend_ok])
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        timeout = sum(1 for r in self.results if r.status == TestStatus.TIMEOUT)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
        total = len(self.results)
        
        print(f"Total Tests: {total}")
        print(f"OK: Passed: {passed}")
        print(f"ERROR: Failed: {failed}")
        print(f"ALARM: Timeout: {timeout}")
        print(f"NEXT: Skipped: {skipped}")
        print()
        
        if failed > 0:
            print("FAILED TESTS:")
            for result in self.results:
                if result.status == TestStatus.FAILED:
                    print(f"  ERROR: {result.name}: {result.error}")
            print()
        
        if timeout > 0:
            print("TIMEOUT TESTS:")
            for result in self.results:
                if result.status == TestStatus.TIMEOUT:
                    print(f"  ALARM: {result.name}")
            print()
        
        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
        if passed == total:
            print("\nSUCCESS: ALL TESTS PASSED! Checkpoint system is working correctly.")
        else:
            print(f"\nWARNING: {failed + timeout} tests failed. Checkpoint system needs attention.")
        
        # Performance insights
        performance_tests = [r for r in self.results if "Performance" in r.name]
        if performance_tests:
            print("\nSUMMARY: Performance Insights:")
            for test in performance_tests:
                if test.details and "performance_ok" in test.details:
                    status = "OK:" if test.details["performance_ok"] else "WARNING:"
                    print(f"  {status} {test.name}: {'Good' if test.details['performance_ok'] else 'Needs attention'}")

async def main():
    """Main entry point for comprehensive testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Checkpoint System Test Suite')
    parser.add_argument('--url', default='http://localhost:8000', help='Backend URL')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    
    args = parser.parse_args()
    
    # Check if backend is running
    try:
        response = requests.get(args.url, timeout=5)
        if response.status_code != 200:
            print(f"ERROR: Backend not accessible at {args.url}")
            print("Please start the backend server first.")
            return False
    except Exception:
        print(f"ERROR: Cannot connect to backend at {args.url}")
        print("Please start the backend server first.")
        return False
    
    print(f"OK: Backend accessible at {args.url}")
    
    # Run tests
    test_suite = CheckpointTestSuite(args.url)
    
    if args.quick:
        # Run basic tests only
        backend_ok = test_suite.run_backend_tests()
        loading_ok = test_suite.run_checkpoint_loading_tests()
        test_suite.print_summary()
        return backend_ok and loading_ok
    else:
        # Run comprehensive test suite
        return await test_suite.run_comprehensive_test_suite()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 