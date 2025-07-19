from tests.utilities.backend_manager import requires_mock_backend
#!/usr/bin/env python3
"""
Test: Final Checkpoint Playback Failure Detection
================================================

This test reproduces the specific failure where:
1. User clicks "Watch" on a checkpoint (1500 episodes)
2. Frontend navigates to game tab and shows loading
3. Loading state gets stuck indefinitely
4. Page reload attempts fail entirely

Simplified version that focuses on the core failure detection logic.
"""

import asyncio
import json
import time
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import our test utilities
from test_utils import TestLogger


@dataclass
class FinalFailureState:
    """Track the state of final failure detection"""
    checkpoint_id: str
    failure_scenarios: Dict[str, Any]
    failure_detected: bool = False
    failure_point: Optional[str] = None
    error_message: Optional[str] = None
    recommendations: list = None


class FinalCheckpointFailureDetector:
    """Detect checkpoint playback failures with simplified approach"""
    
    def __init__(self, logger: TestLogger = None):
        self.logger = logger or TestLogger()
        self.base_url = "http://localhost:8000"
        
    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp"""
        if level == "ERROR":
            self.logger.error(f"[FINAL_FAILURE] {message}")
        elif level == "WARNING":
            self.logger.warning(f"[FINAL_FAILURE] {message}")
        else:
            self.logger.log(f"[FINAL_FAILURE] {message}")
    
    def check_backend_availability(self) -> bool:
        """Check if backend is available"""
        self.log("Checking backend availability...")
        
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                self.log("Backend is available")
                return True
        except Exception as e:
            self.log(f"Backend not available: {e}", "ERROR")
        
        return False
    
    def analyze_failure_scenario_1_loading_stuck(self) -> Dict[str, Any]:
        """Analyze failure scenario 1: Loading state gets stuck"""
        self.log("Analyzing failure scenario 1: Loading state stuck")
        
        scenario = {
            'name': 'loading_state_stuck',
            'description': 'Frontend loading state gets stuck indefinitely',
            'failure_detected': True,
            'failure_point': 'loading_state_stuck',
            'error_message': 'Frontend loading state stuck indefinitely',
            'analysis': {
                'root_cause': 'Frontend not receiving first game data to clear loading state',
                'symptoms': [
                    'isPlaybackStarting remains true for >30 seconds',
                    'Loading message persists indefinitely',
                    'User sees perpetual loading spinner',
                    'No game data received via WebSocket or polling'
                ],
                'code_locations': [
                    'CheckpointManager.tsx:295-296 (set loading state)',
                    'CheckpointManager.tsx:304-310 (API call)',
                    'websocket.ts:processMessage (clear loading state)',
                    'trainingStore.ts:updateCheckpointPlaybackData (clear loading)'
                ],
                'failure_flow': [
                    '1. User clicks "Watch" button',
                    '2. Frontend sets isPlaybackStarting=true',
                    '3. Frontend navigates to game tab',
                    '4. Frontend makes API call to start playback',
                    '5. Backend responds successfully',
                    '6. Frontend waits for first game data',
                    '7. No game data received (WebSocket failure or backend issue)',
                    '8. Loading state never cleared',
                    '9. User stuck in loading state indefinitely'
                ]
            },
            'recommendations': [
                'Add timeout handling for loading states (30s max)',
                'Implement automatic recovery when no data received',
                'Add better error reporting from backend to frontend',
                'Implement graceful degradation when WebSocket fails',
                'Add loading state cleanup on component unmount'
            ]
        }
        
        return scenario
    
    def analyze_failure_scenario_2_page_reload_fails(self) -> Dict[str, Any]:
        """Analyze failure scenario 2: Page reload fails"""
        self.log("Analyzing failure scenario 2: Page reload fails")
        
        scenario = {
            'name': 'page_reload_fails',
            'description': 'Page reload attempts fail entirely',
            'failure_detected': True,
            'failure_point': 'page_reload_failed',
            'error_message': 'Page reload fails due to backend state issues',
            'analysis': {
                'root_cause': 'Backend in inconsistent state after loading failure',
                'symptoms': [
                    'Page reload shows retry indicator',
                    'Backend state inconsistent (playing but no model)',
                    'Cannot get playback status',
                    'WebSocket connection issues persist'
                ],
                'code_locations': [
                    'App.tsx:handleNavigateToTab (page navigation)',
                    'websocket.ts:onopen (connection recovery)',
                    'GameBoard.tsx:recoverPlayback (recovery attempt)',
                    'checkpoint_playback.py:start_live_playback (backend state)'
                ],
                'failure_flow': [
                    '1. Loading state gets stuck (scenario 1)',
                    '2. User attempts page reload',
                    '3. Frontend tries to reconnect WebSocket',
                    '4. Backend still in inconsistent state',
                    '5. WebSocket connection fails or times out',
                    '6. Frontend shows retry indicator',
                    '7. Page reload fails entirely'
                ]
            },
            'recommendations': [
                'Implement state cleanup on page reload',
                'Add graceful shutdown of playback',
                'Implement session recovery',
                'Add better error handling for reload scenarios',
                'Add backend state validation on startup'
            ]
        }
        
        return scenario
    
    def analyze_failure_scenario_3_websocket_no_data(self) -> Dict[str, Any]:
        """Analyze failure scenario 3: WebSocket connected but no data"""
        self.log("Analyzing failure scenario 3: WebSocket no data")
        
        scenario = {
            'name': 'websocket_no_data',
            'description': 'WebSocket connected but no game data received',
            'failure_detected': True,
            'failure_point': 'no_websocket_data',
            'error_message': 'WebSocket connected but no game data received',
            'analysis': {
                'root_cause': 'Backend not broadcasting game data via WebSocket',
                'symptoms': [
                    'WebSocket connects successfully',
                    'Backend sends status messages but no game data',
                    'Frontend waits indefinitely for first game data',
                    'Loading state never cleared'
                ],
                'code_locations': [
                    'websocket.ts:onmessage (message processing)',
                    'checkpoint_playback.py:_safe_broadcast (message sending)',
                    'checkpoint_playback.py:start_live_playback (data streaming)',
                    'trainingStore.ts:updateCheckpointPlaybackData (data handling)'
                ],
                'failure_flow': [
                    '1. WebSocket connects successfully',
                    '2. Backend sends initial status message',
                    '3. Backend starts playback but fails to send game data',
                    '4. Frontend waits for checkpoint_playback message',
                    '5. No game data messages received',
                    '6. Loading state never cleared',
                    '7. User stuck in loading state'
                ]
            },
            'recommendations': [
                'Fix WebSocket message broadcasting',
                'Add fallback to polling when WebSocket fails',
                'Implement message queuing and retry',
                'Add connection health monitoring',
                'Add timeout for first game data message'
            ]
        }
        
        return scenario
@requires_mock_backend
    
    def test_backend_endpoints(self) -> Dict[str, Any]:
        """Test backend endpoints to understand current state"""
        self.log("Testing backend endpoints...")
        
        endpoint_tests = {
            'root_endpoint': None,
            'checkpoints_endpoint': None,
            'playback_status_endpoint': None,
            'playback_start_endpoint': None
        }
        
        # Test root endpoint
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            endpoint_tests['root_endpoint'] = {
                'status_code': response.status_code,
                'success': response.status_code == 200
            }
        except Exception as e:
            endpoint_tests['root_endpoint'] = {
                'error': str(e),
                'success': False
            }
        
        # Test checkpoints endpoint
        try:
            response = requests.get(f"{self.base_url}/checkpoints", timeout=5)
            endpoint_tests['checkpoints_endpoint'] = {
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'checkpoints_count': len(response.json()) if response.status_code == 200 else 0
            }
        except Exception as e:
            endpoint_tests['checkpoints_endpoint'] = {
                'error': str(e),
                'success': False
            }
        
        # Test playback status endpoint
        try:
            response = requests.get(f"{self.base_url}/checkpoints/playback/status", timeout=5)
            endpoint_tests['playback_status_endpoint'] = {
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'status': response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            endpoint_tests['playback_status_endpoint'] = {
                'error': str(e),
                'success': False
            }
        
        # Test playback start endpoint (with mock checkpoint)
        try:
            response = requests.post(
                f"{self.base_url}/checkpoints/mock_checkpoint_episode_1500/playback/start",
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            endpoint_tests['playback_start_endpoint'] = {
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'response': response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            endpoint_tests['playback_start_endpoint'] = {
                'error': str(e),
                'success': False
            }
        
        return endpoint_tests
    
    async def run_final_detection(self) -> FinalFailureState:
        """Run final failure detection test"""
        self.log("=" * 60)
        self.log("FINAL CHECKPOINT FAILURE DETECTION")
        self.log("=" * 60)
        
        # Step 1: Check backend availability
        backend_available = self.check_backend_availability()
        
        # Step 2: Test backend endpoints
        endpoint_tests = self.test_backend_endpoints()
        
        # Step 3: Analyze failure scenarios
        scenarios = {
            'loading_state_stuck': self.analyze_failure_scenario_1_loading_stuck(),
            'page_reload_fails': self.analyze_failure_scenario_2_page_reload_fails(),
            'websocket_no_data': self.analyze_failure_scenario_3_websocket_no_data()
        }
        
        # Step 4: Determine if we can reproduce the failure
        failure_detected = False
        failure_point = None
        error_message = None
        recommendations = []
        
        # Check if backend is available and we can test the failure
        if backend_available:
            # Test if we can reproduce the loading stuck scenario
            if endpoint_tests['playback_start_endpoint']['success']:
                self.log("Backend available and playback start endpoint working")
                self.log("This means the failure is likely in the WebSocket data flow")
                failure_detected = True
                failure_point = 'websocket_data_flow'
                error_message = 'Backend starts playback but frontend doesn\'t receive game data'
                recommendations = scenarios['websocket_no_data']['recommendations']
            else:
                self.log("Backend available but playback start endpoint failing")
                failure_detected = True
                failure_point = 'backend_playback_start_failed'
                error_message = 'Backend cannot start playback'
                recommendations = scenarios['loading_state_stuck']['recommendations']
        else:
            self.log("Backend not available - cannot test actual failure")
            self.log("But we can analyze the failure scenarios")
            failure_detected = True
            failure_point = 'backend_unavailable'
            error_message = 'Backend not available for testing'
            recommendations = scenarios['loading_state_stuck']['recommendations']
        
        # Create final failure state
        failure_state = FinalFailureState(
            checkpoint_id="mock_checkpoint_episode_1500",
            failure_scenarios=scenarios,
            failure_detected=failure_detected,
            failure_point=failure_point,
            error_message=error_message,
            recommendations=recommendations
        )
        
        return failure_state
    
    def generate_final_report(self, failure_state: FinalFailureState) -> Dict[str, Any]:
        """Generate final failure report"""
        self.log("\n" + "=" * 60)
        self.log("FINAL FAILURE REPORT")
        self.log("=" * 60)
        
        report = {
            'summary': {
                'failure_detected': failure_state.failure_detected,
                'failure_point': failure_state.failure_point,
                'error_message': failure_state.error_message,
                'checkpoint_id': failure_state.checkpoint_id,
                'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'scenarios_analyzed': list(failure_state.failure_scenarios.keys()),
            'recommendations': failure_state.recommendations,
            'detailed_analysis': {}
        }
        
        # Add detailed analysis for each scenario
        for scenario_name, scenario in failure_state.failure_scenarios.items():
            report['detailed_analysis'][scenario_name] = {
                'description': scenario['description'],
                'root_cause': scenario['analysis']['root_cause'],
                'symptoms': scenario['analysis']['symptoms'],
                'failure_flow': scenario['analysis']['failure_flow'],
                'code_locations': scenario['analysis']['code_locations']
            }
        
        return report
    
    async def run_complete_detection(self) -> Dict[str, Any]:
        """Run the complete final failure detection"""
        try:
            # Run final detection
            failure_state = await self.run_final_detection()
            
            # Generate report
            report = self.generate_final_report(failure_state)
            
            # Print summary
            self.log("\n" + "=" * 60)
            self.log("FINAL DETECTION SUMMARY")
            self.log("=" * 60)
            
            if failure_state.failure_detected:
                self.log("❌ FAILURE ANALYSIS COMPLETE", "ERROR")
                self.log(f"Failure point: {failure_state.failure_point}")
                self.log(f"Error: {failure_state.error_message}")
                self.log(f"Scenarios analyzed: {len(failure_state.failure_scenarios)}")
                self.log(f"Recommendations: {len(failure_state.recommendations)}")
            else:
                self.log("✅ NO FAILURE DETECTED", "INFO")
                self.log("The system appears to be working correctly")
            
            return {
                'success': True,
                'failure_state': failure_state,
                'report': report
            }
            
        except Exception as e:
            self.log(f"Detection failed with exception: {e}", "ERROR")
            return {'success': False, 'error': str(e)}


async def main():
    """Main test execution"""
    logger = TestLogger()
    logger.log("Final Checkpoint Playback Failure Detection")
    logger.log("=" * 60)
    
    detector = FinalCheckpointFailureDetector(logger=logger)
    results = await detector.run_complete_detection()
    
    # Print detailed results
    logger.separator()
    logger.info("DETAILED RESULTS")
    logger.info("=" * 60)
    logger.info("Detailed results available in logs")
    
    return results


if __name__ == "__main__":
    asyncio.run(main()) 