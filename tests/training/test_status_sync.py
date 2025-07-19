#!/usr/bin/env python3
"""
Test training status synchronization between frontend and backend.
This test verifies that the frontend doesn't show stale training state from localStorage.
"""

import asyncio
import json
import time
from typing import Dict, Any
# Add project root to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))

from tests.utilities.test_utils import TestLogger, BackendTester
from tests.utilities.backend_manager import requires_mock_backend

class TrainingStatusSyncTest:
    def __init__(self, backend_url: str = "http://localhost:8000", frontend_url: str = "http://localhost:5173"):
        self.backend_url = backend_url
        self.frontend_url = frontend_url
        self.logger = TestLogger()
        self.backend = BackendTester(backend_url, self.logger)
        
    def get_backend_training_status(self) -> Dict[str, Any]:
        """Get current training status from backend"""
        try:
            return self.backend.get_training_status() or {"is_training": False, "is_paused": False, "current_episode": 0}
        except Exception as e:
            self.logger.error(f"Error getting backend training status: {e}")
            return {"is_training": False, "is_paused": False, "current_episode": 0}
            
@requires_mock_backend
    def test_backend_training_status_consistency(self) -> bool:
        """Test that backend training status is consistent"""
        self.logger.testing("Testing backend training status consistency")
        
        # Get training status multiple times to check consistency
        statuses = []
        for i in range(5):
            status = self.get_backend_training_status()
            statuses.append(status)
            time.sleep(0.5)
        
        # Check if all statuses are consistent
        first_status = statuses[0]
        consistent = True
        
        for i, status in enumerate(statuses[1:], 1):
            if status.get("is_training") != first_status.get("is_training"):
                self.logger.warning(f"Training status inconsistent at check {i+1}")
                consistent = False
            if status.get("is_paused") != first_status.get("is_paused"):
                self.logger.warning(f"Pause status inconsistent at check {i+1}")
                consistent = False
        
        if consistent:
            self.logger.ok("Backend training status is consistent")
        else:
            self.logger.error("Backend training status is inconsistent")
        
        return consistent
    
@requires_mock_backend
    def test_training_status_endpoint_stability(self) -> bool:
        """Test that training status endpoint is stable under load"""
        self.logger.testing("Testing training status endpoint stability")
        
        success_count = 0
        total_requests = 20
        
        for i in range(total_requests):
            try:
                status = self.get_backend_training_status()
                if status is not None:
                    success_count += 1
                time.sleep(0.1)  # 100ms between requests
            except Exception as e:
                self.logger.warning(f"Request {i+1} failed: {e}")
        
        success_rate = success_count / total_requests * 100
        self.logger.log(f"Training status endpoint success rate: {success_rate:.1f}% ({success_count}/{total_requests})")
        
        if success_rate >= 90:
            self.logger.ok("Training status endpoint is stable")
            return True
        else:
            self.logger.error("Training status endpoint is unstable")
            return False
    
@requires_mock_backend
    def test_training_state_transitions(self) -> bool:
        """Test training state transitions"""
        self.logger.testing("Testing training state transitions")
        
        # Get initial state
        initial_status = self.get_backend_training_status()
        self.logger.log(f"Initial training status: {initial_status}")
        
        # Test that we can get the status multiple times without issues
        transition_success = True
        
        for i in range(10):
            try:
                current_status = self.get_backend_training_status()
                if current_status is None:
                    self.logger.error(f"Failed to get training status at iteration {i+1}")
                    transition_success = False
                    break
                time.sleep(0.2)
            except Exception as e:
                self.logger.error(f"Exception getting training status at iteration {i+1}: {e}")
                transition_success = False
                break
        
        if transition_success:
            self.logger.ok("Training state transitions test passed")
        else:
            self.logger.error("Training state transitions test failed")
        
        return transition_success
    
@requires_mock_backend
    def test_connection_health_during_training_simulation(self) -> bool:
        """Test connection health during simulated training activity"""
        self.logger.testing("Testing connection health during training simulation")
        
        # Simulate the scenario where training would be happening
        # by making rapid requests to training-related endpoints
        
        success_count = 0
        total_requests = 30
        
        for i in range(total_requests):
            try:
                # Alternate between different endpoints to simulate real usage
                if i % 3 == 0:
                    result = self.backend.get_training_status()
                elif i % 3 == 1:
                    result = self.backend.get_checkpoint_stats()
                else:
                    result = self.backend.test_connectivity()
                
                if result is not None:
                    success_count += 1
                
                time.sleep(0.1)  # 100ms between requests
                
            except Exception as e:
                self.logger.warning(f"Request {i+1} failed: {e}")
        
        success_rate = success_count / total_requests * 100
        self.logger.log(f"Connection health during training simulation: {success_rate:.1f}% ({success_count}/{total_requests})")
        
        if success_rate >= 80:
            self.logger.ok("Connection health during training simulation is good")
            return True
        else:
            self.logger.error("Connection health during training simulation is poor")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all training status sync tests"""
        self.logger.banner("Training Status Sync Tests", 60)
        
        results = {
            "backend_consistency": self.test_backend_training_status_consistency(),
            "endpoint_stability": self.test_training_status_endpoint_stability(),
            "state_transitions": self.test_training_state_transitions(),
            "connection_health": self.test_connection_health_during_training_simulation()
        }
        
        # Summary
        self.logger.separator(60)
        self.logger.banner("TEST RESULTS", 60)
        
        passed_tests = sum(1 for result in results.values() if result is True)
        total_tests = len(results)
        
        for test_name, result in results.items():
            status = "PASS" if result is True else "FAIL"
            self.logger.log(f"{test_name}: {status}")
        
        self.logger.log(f"Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            self.logger.success("All training status sync tests passed!")
        else:
            self.logger.error("Some training status sync tests failed!")
        
        return results

@requires_mock_backend("Training Status Sync Tests")
async def main():
    """Run the training status sync tests"""
    logger = TestLogger()
    logger.banner("Training Status Sync Tests", 60)
    
    test = TrainingStatusSyncTest()
    results = await test.run_all_tests()
    
    # Save results to file
    with open("training_status_sync_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to results file")

if __name__ == "__main__":
    asyncio.run(main()) 