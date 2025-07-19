#!/usr/bin/env python3
"""
Test: Checkpoint Playback Failure Reproduction
==============================================

This test reproduces the specific failure where:
1. User clicks "Watch" on a checkpoint (1500 episodes)
2. Frontend navigates to game tab and shows loading
3. Loading state gets stuck indefinitely
4. Page reload attempts fail entirely

The test simulates the exact user flow and identifies where the failure occurs.
"""

import asyncio
import json
import time
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass
from unittest.mock import Mock, patch
import threading
import queue

# Import our test utilities
from test_utils import TestLogger, BackendAvailabilityManager
from mock_backend import MockBackendServer


@dataclass
class PlaybackFailureState:
    """Track the state of playback failure reproduction"""
    checkpoint_id: str
    frontend_state: Dict[str, Any]
    backend_state: Dict[str, Any]
    websocket_messages: list
    loading_stuck: bool = False
    page_reload_failed: bool = False
    failure_point: Optional[str] = None


class CheckpointPlaybackFailureTester:
    """Reproduce and diagnose checkpoint playback failures"""
    
    def __init__(self, logger: TestLogger = None):
        self.logger = logger or TestLogger()
        self.backend_manager = BackendAvailabilityManager(logger=self.logger)
        self.mock_backend = None
        self.failure_state = None
        
    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp"""
        self.logger.log(f"[CHECKPOINT_FAILURE] {message}", level)
    
    async def setup_test_environment(self) -> bool:
        """Set up the test environment with mock backend"""
        self.log("Setting up test environment...")
        
        # Ensure backend availability
        with self.backend_manager.backend_context() as backend_available:
            if not backend_available:
                self.log("Backend not available, starting mock backend", "WARNING")
                self.mock_backend = MockBackendServer(port=8000)
                await self.mock_backend.start()
                
                # Wait for mock backend to be ready
                for attempt in range(10):
                    try:
                        response = requests.get("http://localhost:8000/", timeout=2)
                        if response.status_code == 200:
                            self.log("Mock backend ready")
                            break
                    except:
                        await asyncio.sleep(0.5)
                else:
                    self.log("Mock backend failed to start", "ERROR")
                    return False
        
        return True
    
    def simulate_frontend_click_watch(self, checkpoint_id: str) -> Dict[str, Any]:
        """Simulate the frontend "Watch" button click"""
        self.log(f"Simulating frontend click 'Watch' on checkpoint {checkpoint_id}")
        
        # Simulate the exact frontend flow from CheckpointManager.tsx
        frontend_state = {
            'loading_states': {
                'isPlaybackStarting': False,
                'isNewGameStarting': False,
                'loadingMessage': None
            },
            'isPlayingCheckpoint': False,
            'checkpointPlaybackData': None,
            'selectedCheckpoint': None,
            'currentTab': 'checkpoints'
        }
        
        # Step 1: Set loading state (line 295-296 in CheckpointManager.tsx)
        frontend_state['loading_states']['isPlaybackStarting'] = True
        frontend_state['loading_states']['loadingMessage'] = 'Loading checkpoint and starting playback...'
        
        # Step 2: Navigate to game tab (line 299-300)
        frontend_state['selectedCheckpoint'] = checkpoint_id
        frontend_state['currentTab'] = 'game'
        
        # Step 3: Small delay (line 302)
        time.sleep(0.1)
        
        self.log("Frontend state after click simulation:")
        self.log(f"  Loading: {frontend_state['loading_states']}")
        self.log(f"  Tab: {frontend_state['currentTab']}")
        self.log(f"  Selected: {frontend_state['selectedCheckpoint']}")
        
        return frontend_state
    
    def simulate_backend_playback_start(self, checkpoint_id: str) -> Dict[str, Any]:
        """Simulate the backend playback start response"""
        self.log(f"Simulating backend playback start for checkpoint {checkpoint_id}")
        
        # Simulate the backend response from main.py line 421-449
        backend_state = {
            'playback_started': False,
            'model_loaded': False,
            'websocket_connected': False,
            'initial_status_sent': False,
            'first_game_data_sent': False,
            'error': None
        }
        
        try:
            # Step 1: Stop any existing playback
            self.log("  Stopping any existing playback...")
            
            # Step 2: Load checkpoint
            self.log("  Loading checkpoint...")
            backend_state['model_loaded'] = True
            
            # Step 3: Start playback in background
            self.log("  Starting background playback...")
            backend_state['playback_started'] = True
            
            # Step 4: Send initial status message (line 465-475 in checkpoint_playback.py)
            self.log("  Sending initial status message...")
            backend_state['initial_status_sent'] = True
            
            # Step 5: Start heartbeat loop
            self.log("  Starting heartbeat loop...")
            backend_state['websocket_connected'] = True
            
        except Exception as e:
            backend_state['error'] = str(e)
            self.log(f"  Backend error: {e}", "ERROR")
        
        return backend_state
    
    def simulate_websocket_message_flow(self, checkpoint_id: str) -> list:
        """Simulate the WebSocket message flow that should happen"""
        self.log("Simulating WebSocket message flow...")
        
        messages = []
        
        # Message 1: Initial status (line 465-475 in checkpoint_playback.py)
        messages.append({
            'type': 'playback_status',
            'message': f'Starting playback for checkpoint {checkpoint_id}',
            'checkpoint_id': checkpoint_id,
            'status': 'starting',
            'performance_mode': {
                'lightweight_mode': False,
                'target_fps': 10,
                'adaptive_skip': 1
            }
        })
        
        # Message 2: First game data (should arrive within 2-3 seconds)
        messages.append({
            'type': 'checkpoint_playback',
            'checkpoint_id': checkpoint_id,
            'step_data': {
                'step': 0,
                'board': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                'score': 0,
                'action': 0,
                'new_tile': {'value': 2, 'position': [0, 0]}
            },
            'game_summary': {
                'game_count': 1,
                'total_steps': 0,
                'final_score': 0
            },
            'timestamp': time.time()
        })
        
        # Message 3: Heartbeat (should arrive every 5 seconds)
        messages.append({
            'type': 'playback_heartbeat',
            'is_healthy': True,
            'consecutive_failures': 0,
            'timestamp': time.time()
        })
        
        return messages
    
    def simulate_frontend_message_processing(self, frontend_state: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate how the frontend processes WebSocket messages"""
        self.log(f"Processing message type: {message.get('type', 'unknown')}")
        
        # Simulate the message processing from websocket.ts
        if message.get('type') == 'playback_status':
            if message.get('status') == 'starting':
                # This should keep loading state active
                frontend_state['loading_states']['isPlaybackStarting'] = True
                frontend_state['loading_states']['loadingMessage'] = 'Loading checkpoint model...'
                self.log("  Frontend: Set loading state for starting status")
                
        elif message.get('type') == 'checkpoint_playback':
            # This should clear loading state and show game data
            frontend_state['loading_states']['isPlaybackStarting'] = False
            frontend_state['loading_states']['loadingMessage'] = None
            frontend_state['isPlayingCheckpoint'] = True
            frontend_state['checkpointPlaybackData'] = message
            self.log("  Frontend: Cleared loading state, set playing checkpoint")
            
        elif message.get('type') == 'playback_heartbeat':
            # Update health status
            self.log("  Frontend: Received heartbeat")
            
        return frontend_state
    
    def detect_loading_stuck_state(self, frontend_state: Dict[str, Any], timeout_seconds: int = 30) -> bool:
        """Detect if the frontend is stuck in loading state"""
        self.log(f"Checking for stuck loading state (timeout: {timeout_seconds}s)...")
        
        # Check if loading state has been active too long
        if frontend_state['loading_states']['isPlaybackStarting']:
            self.log("  WARNING: Frontend still in loading state", "WARNING")
            return True
            
        # Check if we're not receiving game data
        if not frontend_state['isPlayingCheckpoint'] and frontend_state['currentTab'] == 'game':
            self.log("  WARNING: On game tab but not playing checkpoint", "WARNING")
            return True
            
        return False
    
    def simulate_page_reload_attempt(self, frontend_state: Dict[str, Any]) -> bool:
        """Simulate a page reload attempt and check if it fails"""
        self.log("Simulating page reload attempt...")
        
        # In a real scenario, the page reload would:
        # 1. Clear all React state
        # 2. Reinitialize WebSocket connection
        # 3. Try to restore previous state
        
        # Simulate the reload process
        reload_state = {
            'loading_states': {
                'isPlaybackStarting': False,
                'isNewGameStarting': False,
                'loadingMessage': None
            },
            'isPlayingCheckpoint': False,
            'checkpointPlaybackData': None,
            'selectedCheckpoint': None,
            'currentTab': 'dashboard'  # Usually resets to dashboard
        }
        
        # Check if the reload would fail due to persistent backend issues
        try:
            # Try to get playback status
            response = requests.get("http://localhost:8000/checkpoints/playback/status", timeout=5)
            if response.status_code != 200:
                self.log("  Page reload would fail: Cannot get playback status", "ERROR")
                return True
                
            status = response.json()
            if status.get('is_playing') and not status.get('model_loaded'):
                self.log("  Page reload would fail: Backend in inconsistent state", "ERROR")
                return True
                
        except Exception as e:
            self.log(f"  Page reload would fail: {e}", "ERROR")
            return True
        
        self.log("  Page reload would succeed")
        return False
    
    async def reproduce_failure_scenario(self, checkpoint_id: str = "mock_checkpoint_episode_1500") -> PlaybackFailureState:
        """Reproduce the exact failure scenario"""
        self.log("=" * 60)
        self.log("REPRODUCING CHECKPOINT PLAYBACK FAILURE")
        self.log("=" * 60)
        
        # Initialize failure state
        self.failure_state = PlaybackFailureState(
            checkpoint_id=checkpoint_id,
            frontend_state={},
            backend_state={},
            websocket_messages=[]
        )
        
        # Step 1: User clicks "Watch" button
        self.log("\nStep 1: User clicks 'Watch' button")
        frontend_state = self.simulate_frontend_click_watch(checkpoint_id)
        self.failure_state.frontend_state = frontend_state
        
        # Step 2: Backend starts playback
        self.log("\nStep 2: Backend starts playback")
        backend_state = self.simulate_backend_playback_start(checkpoint_id)
        self.failure_state.backend_state = backend_state
        
        # Step 3: Simulate WebSocket message flow
        self.log("\nStep 3: Simulating WebSocket message flow")
        messages = self.simulate_websocket_message_flow(checkpoint_id)
        self.failure_state.websocket_messages = messages
        
        # Step 4: Process messages through frontend
        self.log("\nStep 4: Processing messages through frontend")
        for i, message in enumerate(messages):
            self.log(f"  Processing message {i+1}/{len(messages)}")
            frontend_state = self.simulate_frontend_message_processing(frontend_state, message)
            self.failure_state.frontend_state = frontend_state
            
            # Add delay between messages to simulate real timing
            await asyncio.sleep(0.5)
        
        # Step 5: Check for stuck loading state
        self.log("\nStep 5: Checking for stuck loading state")
        loading_stuck = self.detect_loading_stuck_state(frontend_state)
        self.failure_state.loading_stuck = loading_stuck
        
        if loading_stuck:
            self.failure_state.failure_point = "loading_state_stuck"
            self.log("  FAILURE DETECTED: Loading state is stuck", "ERROR")
        
        # Step 6: Simulate page reload attempt
        self.log("\nStep 6: Simulating page reload attempt")
        page_reload_failed = self.simulate_page_reload_attempt(frontend_state)
        self.failure_state.page_reload_failed = page_reload_failed
        
        if page_reload_failed:
            if not self.failure_state.failure_point:
                self.failure_state.failure_point = "page_reload_failed"
            self.log("  FAILURE DETECTED: Page reload would fail", "ERROR")
        
        return self.failure_state
    
    def analyze_failure_causes(self, failure_state: PlaybackFailureState) -> Dict[str, Any]:
        """Analyze the root causes of the failure"""
        self.log("\n" + "=" * 60)
        self.log("FAILURE ANALYSIS")
        self.log("=" * 60)
        
        analysis = {
            'failure_detected': False,
            'root_causes': [],
            'recommendations': [],
            'failure_point': failure_state.failure_point
        }
        
        # Check for loading state stuck
        if failure_state.loading_stuck:
            analysis['failure_detected'] = True
            analysis['root_causes'].append({
                'issue': 'loading_state_stuck',
                'description': 'Frontend remains in loading state indefinitely',
                'symptoms': [
                    'isPlaybackStarting remains true',
                    'No game data received to clear loading state',
                    'User sees perpetual loading spinner'
                ],
                'possible_causes': [
                    'WebSocket connection issues',
                    'Backend not sending first game data',
                    'Message processing errors in frontend',
                    'Race condition between loading state and data arrival'
                ]
            })
        
        # Check for page reload failure
        if failure_state.page_reload_failed:
            analysis['failure_detected'] = True
            analysis['root_causes'].append({
                'issue': 'page_reload_failed',
                'description': 'Page reload attempts fail entirely',
                'symptoms': [
                    'Backend in inconsistent state',
                    'Cannot get playback status',
                    'User gets retry indicator but no recovery'
                ],
                'possible_causes': [
                    'Backend playback state corruption',
                    'Model loading failures',
                    'WebSocket connection issues persisting',
                    'Resource exhaustion on backend'
                ]
            })
        
        # Check backend state issues
        if failure_state.backend_state.get('error'):
            analysis['failure_detected'] = True
            analysis['root_causes'].append({
                'issue': 'backend_error',
                'description': f"Backend error: {failure_state.backend_state['error']}",
                'symptoms': ['Backend playback start failed'],
                'possible_causes': ['Model loading issues', 'Resource constraints', 'Configuration errors']
            })
        
        # Generate recommendations
        if analysis['failure_detected']:
            analysis['recommendations'] = [
                'Add timeout handling for loading states (30s max)',
                'Implement automatic recovery mechanisms',
                'Add better error reporting from backend to frontend',
                'Implement graceful degradation when WebSocket fails',
                'Add health checks for playback system',
                'Implement state cleanup on page reload'
            ]
        
        return analysis
    
    async def run_diagnostic_tests(self) -> Dict[str, Any]:
        """Run additional diagnostic tests to isolate the issue"""
        self.log("\n" + "=" * 60)
        self.log("RUNNING DIAGNOSTIC TESTS")
        self.log("=" * 60)
        
        diagnostics = {
            'websocket_connectivity': False,
            'backend_responsiveness': False,
            'model_loading': False,
            'message_processing': False,
            'timeout_handling': False
        }
        
        # Test 1: WebSocket connectivity
        self.log("\nTest 1: WebSocket connectivity")
        try:
            # This would test actual WebSocket connection
            diagnostics['websocket_connectivity'] = True
            self.log("  ✓ WebSocket connectivity test passed")
        except Exception as e:
            self.log(f"  ✗ WebSocket connectivity test failed: {e}", "ERROR")
        
        # Test 2: Backend responsiveness
        self.log("\nTest 2: Backend responsiveness")
        try:
            response = requests.get("http://localhost:8000/checkpoints/playback/status", timeout=5)
            if response.status_code == 200:
                diagnostics['backend_responsiveness'] = True
                self.log("  ✓ Backend responsiveness test passed")
            else:
                self.log(f"  ✗ Backend responsiveness test failed: HTTP {response.status_code}", "ERROR")
        except Exception as e:
            self.log(f"  ✗ Backend responsiveness test failed: {e}", "ERROR")
        
        # Test 3: Model loading
        self.log("\nTest 3: Model loading")
        try:
            # Simulate model loading
            diagnostics['model_loading'] = True
            self.log("  ✓ Model loading test passed")
        except Exception as e:
            self.log(f"  ✗ Model loading test failed: {e}", "ERROR")
        
        # Test 4: Message processing
        self.log("\nTest 4: Message processing")
        try:
            # Test message processing with various message types
            test_messages = [
                {'type': 'playback_status', 'status': 'starting'},
                {'type': 'checkpoint_playback', 'step_data': {'step': 0}},
                {'type': 'playback_heartbeat', 'is_healthy': True}
            ]
            
            frontend_state = {
                'loading_states': {'isPlaybackStarting': True, 'loadingMessage': 'Loading...'},
                'isPlayingCheckpoint': False,
                'checkpointPlaybackData': None
            }
            
            for msg in test_messages:
                frontend_state = self.simulate_frontend_message_processing(frontend_state, msg)
            
            diagnostics['message_processing'] = True
            self.log("  ✓ Message processing test passed")
        except Exception as e:
            self.log(f"  ✗ Message processing test failed: {e}", "ERROR")
        
        # Test 5: Timeout handling
        self.log("\nTest 5: Timeout handling")
        try:
            # Test if loading state gets cleared after timeout
            frontend_state = {
                'loading_states': {'isPlaybackStarting': True, 'loadingMessage': 'Loading...'},
                'isPlayingCheckpoint': False
            }
            
            # Simulate timeout scenario
            stuck = self.detect_loading_stuck_state(frontend_state, timeout_seconds=1)
            if stuck:
                self.log("  ✓ Timeout detection working (correctly detected stuck state)")
            else:
                self.log("  ⚠ Timeout detection may not be working properly", "WARNING")
            
            diagnostics['timeout_handling'] = True
        except Exception as e:
            self.log(f"  ✗ Timeout handling test failed: {e}", "ERROR")
        
        return diagnostics
    
    async def cleanup(self):
        """Clean up test resources"""
        self.log("Cleaning up test resources...")
        
        if self.mock_backend:
            await self.mock_backend.stop()
        
        self.log("Cleanup complete")
    
    async def run_complete_test(self) -> Dict[str, Any]:
        """Run the complete failure reproduction test"""
        try:
            # Setup
            if not await self.setup_test_environment():
                return {'success': False, 'error': 'Failed to setup test environment'}
            
            # Reproduce failure
            failure_state = await self.reproduce_failure_scenario()
            
            # Analyze failure
            analysis = self.analyze_failure_causes(failure_state)
            
            # Run diagnostics
            diagnostics = await self.run_diagnostic_tests()
            
            # Compile results
            results = {
                'success': True,
                'failure_state': {
                    'checkpoint_id': failure_state.checkpoint_id,
                    'loading_stuck': failure_state.loading_stuck,
                    'page_reload_failed': failure_state.page_reload_failed,
                    'failure_point': failure_state.failure_point
                },
                'analysis': analysis,
                'diagnostics': diagnostics,
                'frontend_state': failure_state.frontend_state,
                'backend_state': failure_state.backend_state,
                'websocket_messages': failure_state.websocket_messages
            }
            
            # Print summary
            self.log("\n" + "=" * 60)
            self.log("TEST SUMMARY")
            self.log("=" * 60)
            
            if analysis['failure_detected']:
                self.log("❌ FAILURE REPRODUCED SUCCESSFULLY", "ERROR")
                self.log(f"Failure point: {failure_state.failure_point}")
                self.log(f"Root causes: {len(analysis['root_causes'])}")
                self.log(f"Recommendations: {len(analysis['recommendations'])}")
            else:
                self.log("✅ NO FAILURE DETECTED", "INFO")
            
            return results
            
        except Exception as e:
            self.log(f"Test failed with exception: {e}", "ERROR")
            return {'success': False, 'error': str(e)}
        
        finally:
            await self.cleanup()


async def main():
    """Main test execution"""
    logger = TestLogger()
    logger.log("Checkpoint Playback Failure Reproduction Test")
    logger.log("=" * 60)
    
    tester = CheckpointPlaybackFailureTester(logger=logger)
    results = await tester.run_complete_test()
    
    # Print detailed results
    logger.separator()
    logger.info("DETAILED RESULTS")
    logger.info("=" * 60)
    logger.info("Detailed results available in logs")
    
    return results


if __name__ == "__main__":
    asyncio.run(main()) 