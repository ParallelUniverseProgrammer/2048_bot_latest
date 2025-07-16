#!/usr/bin/env python3
"""
Test: Real Checkpoint Playback Failure Detection
===============================================

This test attempts to reproduce the actual failure by:
1. Starting the real backend
2. Simulating the exact frontend behavior
3. Monitoring the actual WebSocket communication
4. Detecting where the failure occurs in the real system

This is more targeted than the mock test and should catch real issues.
"""

import asyncio
import json
import time
import requests
import websocket
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

# Import our test utilities
from test_utils import TestLogger, BackendAvailabilityManager


@dataclass
class RealFailureState:
    """Track the state of real failure detection"""
    checkpoint_id: str
    backend_url: str
    websocket_url: str
    frontend_simulation: Dict[str, Any]
    backend_status: Dict[str, Any]
    websocket_messages: List[Dict[str, Any]]
    failure_detected: bool = False
    failure_point: Optional[str] = None
    failure_time: Optional[float] = None
    error_message: Optional[str] = None


class RealCheckpointFailureDetector:
    """Detect real checkpoint playback failures"""
    
    def __init__(self, logger: TestLogger = None):
        self.logger = logger or TestLogger()
        self.backend_manager = BackendAvailabilityManager(logger=self.logger)
        self.websocket_connection = None
        self.websocket_messages = []
        self.websocket_connected = False
        self.failure_state = None
        
    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp"""
        self.logger.log(f"[REAL_FAILURE] {message}", level)
    
    async def ensure_backend_available(self) -> bool:
        """Ensure backend is available, start if needed"""
        self.log("Ensuring backend availability...")
        
        with self.backend_manager.backend_context() as backend_available:
            if backend_available:
                self.log("Real backend is available")
                return True
            else:
                self.log("Real backend not available, cannot test real failure", "ERROR")
                return False
    
    def get_checkpoint_with_1500_episodes(self) -> Optional[str]:
        """Find a checkpoint with 1500 episodes (the one mentioned in the failure)"""
        self.log("Looking for checkpoint with 1500 episodes...")
        
        try:
            response = requests.get("http://localhost:8000/checkpoints", timeout=10)
            if response.status_code == 200:
                checkpoints = response.json()
                
                for checkpoint in checkpoints:
                    if checkpoint.get('episode') == 1500:
                        self.log(f"Found checkpoint: {checkpoint['id']} with {checkpoint['episode']} episodes")
                        return checkpoint['id']
                
                self.log("No checkpoint with 1500 episodes found, using first available", "WARNING")
                if checkpoints:
                    return checkpoints[0]['id']
            
        except Exception as e:
            self.log(f"Error getting checkpoints: {e}", "ERROR")
        
        return None
    
    def simulate_frontend_click_watch(self, checkpoint_id: str) -> Dict[str, Any]:
        """Simulate the exact frontend "Watch" button click behavior"""
        self.log(f"Simulating frontend click 'Watch' on checkpoint {checkpoint_id}")
        
        # Simulate the exact sequence from CheckpointManager.tsx
        frontend_state = {
            'timestamp': time.time(),
            'actions': [],
            'loading_states': {
                'isPlaybackStarting': False,
                'isNewGameStarting': False,
                'loadingMessage': None
            },
            'navigation': {
                'from_tab': 'checkpoints',
                'to_tab': 'game',
                'selected_checkpoint': None
            },
            'websocket_ready': False,
            'data_received': False
        }
        
        # Action 1: Set loading state (CheckpointManager.tsx line 295-296)
        frontend_state['loading_states']['isPlaybackStarting'] = True
        frontend_state['loading_states']['loadingMessage'] = 'Loading checkpoint and starting playback...'
        frontend_state['actions'].append({
            'action': 'set_loading_state',
            'timestamp': time.time(),
            'details': 'Set isPlaybackStarting=true, loadingMessage="Loading checkpoint and starting playback..."'
        })
        
        # Action 2: Navigate to game tab (CheckpointManager.tsx line 299-300)
        frontend_state['navigation']['selected_checkpoint'] = checkpoint_id
        frontend_state['navigation']['to_tab'] = 'game'
        frontend_state['actions'].append({
            'action': 'navigate_to_game_tab',
            'timestamp': time.time(),
            'details': f'Selected checkpoint {checkpoint_id}, navigated to game tab'
        })
        
        # Action 3: Small delay (CheckpointManager.tsx line 302)
        time.sleep(0.1)
        frontend_state['actions'].append({
            'action': 'delay_100ms',
            'timestamp': time.time(),
            'details': '100ms delay to ensure loading state is visible'
        })
        
        # Action 4: Make API call to start playback (CheckpointManager.tsx line 304-310)
        try:
            response = requests.post(
                f"http://localhost:8000/checkpoints/{checkpoint_id}/playback/start",
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                frontend_state['actions'].append({
                    'action': 'playback_start_api_success',
                    'timestamp': time.time(),
                    'details': f'API call successful: {response.json()}'
                })
            else:
                frontend_state['actions'].append({
                    'action': 'playback_start_api_failed',
                    'timestamp': time.time(),
                    'details': f'API call failed: HTTP {response.status_code} - {response.text}'
                })
                # Clear loading state on error (CheckpointManager.tsx line 312-314)
                frontend_state['loading_states']['isPlaybackStarting'] = False
                frontend_state['loading_states']['loadingMessage'] = None
                
        except Exception as e:
            frontend_state['actions'].append({
                'action': 'playback_start_api_exception',
                'timestamp': time.time(),
                'details': f'API call exception: {str(e)}'
            })
            # Clear loading state on error
            frontend_state['loading_states']['isPlaybackStarting'] = False
            frontend_state['loading_states']['loadingMessage'] = None
        
        self.log(f"Frontend simulation completed with {len(frontend_state['actions'])} actions")
        return frontend_state
    
    def setup_websocket_monitoring(self, websocket_url: str):
        """Set up WebSocket monitoring to capture all messages"""
        self.log(f"Setting up WebSocket monitoring on {websocket_url}")
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self.websocket_messages.append({
                    'timestamp': time.time(),
                    'data': data,
                    'message_type': data.get('type', 'unknown')
                })
                self.log(f"WebSocket message received: {data.get('type', 'unknown')}")
            except json.JSONDecodeError:
                self.log(f"WebSocket message decode error: {message}", "ERROR")
        
        def on_error(ws, error):
            self.log(f"WebSocket error: {error}", "ERROR")
        
        def on_close(ws, close_status_code, close_msg):
            self.log(f"WebSocket closed: {close_status_code} - {close_msg}")
            self.websocket_connected = False
        
        def on_open(ws):
            self.log("WebSocket connected for monitoring")
            self.websocket_connected = True
        
        # Create WebSocket connection for monitoring
        self.websocket_connection = websocket.WebSocketApp(
            websocket_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Start WebSocket in a separate thread
        websocket_thread = threading.Thread(target=self.websocket_connection.run_forever)
        websocket_thread.daemon = True
        websocket_thread.start()
        
        # Wait for connection
        for attempt in range(10):
            if self.websocket_connected:
                break
            time.sleep(0.5)
        
        if not self.websocket_connected:
            self.log("WebSocket connection failed", "ERROR")
            return False
        
        return True
    
    def monitor_playback_status(self, timeout_seconds: int = 60) -> Dict[str, Any]:
        """Monitor playback status to detect failures"""
        self.log(f"Monitoring playback status for {timeout_seconds} seconds...")
        
        start_time = time.time()
        status_history = []
        failure_detected = False
        failure_point = None
        error_message = None
        
        while time.time() - start_time < timeout_seconds:
            try:
                # Check playback status
                response = requests.get("http://localhost:8000/checkpoints/playback/status", timeout=5)
                if response.status_code == 200:
                    status = response.json()
                    status_history.append({
                        'timestamp': time.time(),
                        'status': status
                    })
                    
                    # Check for failure conditions
                    if status.get('is_playing') and not status.get('model_loaded'):
                        failure_detected = True
                        failure_point = "backend_inconsistent_state"
                        error_message = "Backend says playing but model not loaded"
                        break
                    
                    if status.get('error_count', 0) > 0:
                        failure_detected = True
                        failure_point = "backend_errors"
                        error_message = f"Backend has {status.get('error_count')} errors"
                        break
                    
                    # Check if we're getting stuck in loading
                    if status.get('is_playing') and not self.websocket_messages:
                        # Backend says playing but no WebSocket messages
                        time_since_start = time.time() - start_time
                        if time_since_start > 10:  # 10 seconds without data
                            failure_detected = True
                            failure_point = "no_websocket_data"
                            error_message = f"No WebSocket data for {time_since_start:.1f} seconds"
                            break
                
                else:
                    failure_detected = True
                    failure_point = "status_endpoint_failed"
                    error_message = f"Status endpoint failed: HTTP {response.status_code}"
                    break
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                failure_detected = True
                failure_point = "status_monitoring_exception"
                error_message = f"Status monitoring exception: {str(e)}"
                break
        
        return {
            'failure_detected': failure_detected,
            'failure_point': failure_point,
            'error_message': error_message,
            'status_history': status_history,
            'websocket_messages': self.websocket_messages,
            'monitoring_duration': time.time() - start_time
        }
    
    def check_loading_state_stuck(self, frontend_state: Dict[str, Any]) -> bool:
        """Check if the frontend loading state is stuck"""
        self.log("Checking for stuck loading state...")
        
        # Check if loading state is still active
        if frontend_state['loading_states']['isPlaybackStarting']:
            # Check how long it's been active
            start_time = frontend_state['timestamp']
            current_time = time.time()
            duration = current_time - start_time
            
            if duration > 30:  # 30 seconds is too long
                self.log(f"Loading state stuck for {duration:.1f} seconds", "ERROR")
                return True
        
        # Check if we're on game tab but not receiving data
        if (frontend_state['navigation']['to_tab'] == 'game' and 
            not frontend_state['data_received'] and 
            not self.websocket_messages):
            
            start_time = frontend_state['timestamp']
            current_time = time.time()
            duration = current_time - start_time
            
            if duration > 15:  # 15 seconds without data is suspicious
                self.log(f"No data received for {duration:.1f} seconds on game tab", "ERROR")
                return True
        
        return False
    
    def simulate_page_reload_scenario(self) -> Dict[str, Any]:
        """Simulate what would happen during a page reload"""
        self.log("Simulating page reload scenario...")
        
        reload_result = {
            'reload_successful': False,
            'issues_detected': [],
            'backend_state_after_reload': None
        }
        
        try:
            # Simulate page reload by checking backend state
            response = requests.get("http://localhost:8000/checkpoints/playback/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                reload_result['backend_state_after_reload'] = status
                
                # Check for issues that would cause reload to fail
                if status.get('is_playing') and not status.get('model_loaded'):
                    reload_result['issues_detected'].append("Backend in inconsistent state")
                
                if status.get('error_count', 0) > 0:
                    reload_result['issues_detected'].append(f"Backend has {status.get('error_count')} errors")
                
                if not status.get('current_checkpoint'):
                    reload_result['issues_detected'].append("No current checkpoint")
                
                # Check if WebSocket would reconnect
                if not self.websocket_connected:
                    reload_result['issues_detected'].append("WebSocket not connected")
                
                reload_result['reload_successful'] = len(reload_result['issues_detected']) == 0
                
            else:
                reload_result['issues_detected'].append(f"Status endpoint failed: HTTP {response.status_code}")
        
        except Exception as e:
            reload_result['issues_detected'].append(f"Reload simulation exception: {str(e)}")
        
        return reload_result
    
    async def detect_real_failure(self, checkpoint_id: str = None) -> RealFailureState:
        """Detect real failure in the checkpoint playback system"""
        self.log("=" * 60)
        self.log("REAL CHECKPOINT FAILURE DETECTION")
        self.log("=" * 60)
        
        # Step 1: Ensure backend is available
        if not await self.ensure_backend_available():
            return RealFailureState(
                checkpoint_id="",
                backend_url="http://localhost:8000",
                websocket_url="ws://localhost:8000/ws",
                frontend_simulation={},
                backend_status={},
                websocket_messages=[],
                failure_detected=True,
                failure_point="backend_unavailable",
                error_message="Backend is not available"
            )
        
        # Step 2: Find checkpoint with 1500 episodes
        if not checkpoint_id:
            checkpoint_id = self.get_checkpoint_with_1500_episodes()
            if not checkpoint_id:
                return RealFailureState(
                    checkpoint_id="",
                    backend_url="http://localhost:8000",
                    websocket_url="ws://localhost:8000/ws",
                    frontend_simulation={},
                    backend_status={},
                    websocket_messages=[],
                    failure_detected=True,
                    failure_point="no_checkpoint_found",
                    error_message="No checkpoint with 1500 episodes found"
                )
        
        # Step 3: Set up WebSocket monitoring
        websocket_url = "ws://localhost:8000/ws"
        if not self.setup_websocket_monitoring(websocket_url):
            return RealFailureState(
                checkpoint_id=checkpoint_id,
                backend_url="http://localhost:8000",
                websocket_url=websocket_url,
                frontend_simulation={},
                backend_status={},
                websocket_messages=[],
                failure_detected=True,
                failure_point="websocket_monitoring_failed",
                error_message="Failed to set up WebSocket monitoring"
            )
        
        # Step 4: Simulate frontend click
        self.log(f"\nStep 4: Simulating frontend click on checkpoint {checkpoint_id}")
        frontend_state = self.simulate_frontend_click_watch(checkpoint_id)
        
        # Step 5: Monitor for failures
        self.log("\nStep 5: Monitoring for failures...")
        monitoring_result = self.monitor_playback_status(timeout_seconds=60)
        
        # Step 6: Check for stuck loading state
        loading_stuck = self.check_loading_state_stuck(frontend_state)
        
        # Step 7: Simulate page reload
        reload_result = self.simulate_page_reload_scenario()
        
        # Step 8: Determine if failure occurred
        failure_detected = (
            monitoring_result['failure_detected'] or 
            loading_stuck or 
            not reload_result['reload_successful']
        )
        
        failure_point = None
        error_message = None
        
        if monitoring_result['failure_detected']:
            failure_point = monitoring_result['failure_point']
            error_message = monitoring_result['error_message']
        elif loading_stuck:
            failure_point = "loading_state_stuck"
            error_message = "Frontend loading state stuck indefinitely"
        elif not reload_result['reload_successful']:
            failure_point = "page_reload_failed"
            error_message = f"Page reload would fail: {', '.join(reload_result['issues_detected'])}"
        
        # Create failure state
        failure_state = RealFailureState(
            checkpoint_id=checkpoint_id,
            backend_url="http://localhost:8000",
            websocket_url=websocket_url,
            frontend_simulation=frontend_state,
            backend_status=monitoring_result.get('status_history', [{}])[-1].get('status', {}),
            websocket_messages=self.websocket_messages,
            failure_detected=failure_detected,
            failure_point=failure_point,
            failure_time=time.time() if failure_detected else None,
            error_message=error_message
        )
        
        self.failure_state = failure_state
        return failure_state
    
    def generate_failure_report(self, failure_state: RealFailureState) -> Dict[str, Any]:
        """Generate a comprehensive failure report"""
        self.log("\n" + "=" * 60)
        self.log("FAILURE REPORT")
        self.log("=" * 60)
        
        report = {
            'summary': {
                'failure_detected': failure_state.failure_detected,
                'failure_point': failure_state.failure_point,
                'error_message': failure_state.error_message,
                'checkpoint_id': failure_state.checkpoint_id,
                'test_timestamp': datetime.now().isoformat()
            },
            'frontend_analysis': {
                'actions_performed': len(failure_state.frontend_simulation.get('actions', [])),
                'loading_state_stuck': failure_state.frontend_simulation.get('loading_states', {}).get('isPlaybackStarting', False),
                'navigation_successful': failure_state.frontend_simulation.get('navigation', {}).get('to_tab') == 'game',
                'data_received': failure_state.frontend_simulation.get('data_received', False)
            },
            'backend_analysis': {
                'is_playing': failure_state.backend_status.get('is_playing', False),
                'model_loaded': failure_state.backend_status.get('model_loaded', False),
                'current_checkpoint': failure_state.backend_status.get('current_checkpoint'),
                'error_count': failure_state.backend_status.get('error_count', 0),
                'connected_clients': failure_state.backend_status.get('connected_clients', 0)
            },
            'websocket_analysis': {
                'messages_received': len(failure_state.websocket_messages),
                'connection_active': self.websocket_connected,
                'message_types': [msg.get('message_type', 'unknown') for msg in failure_state.websocket_messages],
                'first_message_time': failure_state.websocket_messages[0]['timestamp'] if failure_state.websocket_messages else None,
                'last_message_time': failure_state.websocket_messages[-1]['timestamp'] if failure_state.websocket_messages else None
            },
            'recommendations': []
        }
        
        # Generate recommendations based on failure type
        if failure_state.failure_point == "loading_state_stuck":
            report['recommendations'] = [
                "Add timeout handling for loading states (30s max)",
                "Implement automatic recovery when no data received",
                "Add better error reporting from backend to frontend",
                "Implement graceful degradation when WebSocket fails"
            ]
        elif failure_state.failure_point == "backend_inconsistent_state":
            report['recommendations'] = [
                "Fix backend state management",
                "Add state validation before starting playback",
                "Implement automatic state recovery",
                "Add health checks for playback system"
            ]
        elif failure_state.failure_point == "no_websocket_data":
            report['recommendations'] = [
                "Fix WebSocket message broadcasting",
                "Add fallback to polling when WebSocket fails",
                "Implement message queuing and retry",
                "Add connection health monitoring"
            ]
        elif failure_state.failure_point == "page_reload_failed":
            report['recommendations'] = [
                "Implement state cleanup on page reload",
                "Add graceful shutdown of playback",
                "Implement session recovery",
                "Add better error handling for reload scenarios"
            ]
        
        return report
    
    def cleanup(self):
        """Clean up resources"""
        self.log("Cleaning up resources...")
        
        if self.websocket_connection:
            self.websocket_connection.close()
        
        self.log("Cleanup complete")
    
    async def run_complete_detection(self) -> Dict[str, Any]:
        """Run the complete real failure detection"""
        try:
            # Detect failure
            failure_state = await self.detect_real_failure()
            
            # Generate report
            report = self.generate_failure_report(failure_state)
            
            # Print summary
            self.log("\n" + "=" * 60)
            self.log("DETECTION SUMMARY")
            self.log("=" * 60)
            
            if failure_state.failure_detected:
                self.log("❌ REAL FAILURE DETECTED", "ERROR")
                self.log(f"Failure point: {failure_state.failure_point}")
                self.log(f"Error: {failure_state.error_message}")
                self.log(f"Recommendations: {len(report['recommendations'])}")
            else:
                self.log("✅ NO REAL FAILURE DETECTED", "INFO")
                self.log("The system appears to be working correctly")
            
            return {
                'success': True,
                'failure_state': failure_state,
                'report': report
            }
            
        except Exception as e:
            self.log(f"Detection failed with exception: {e}", "ERROR")
            return {'success': False, 'error': str(e)}
        
        finally:
            self.cleanup()


async def main():
    """Main test execution"""
    logger = TestLogger()
    logger.log("Real Checkpoint Playback Failure Detection")
    logger.log("=" * 60)
    
    detector = RealCheckpointFailureDetector(logger=logger)
    results = await detector.run_complete_detection()
    
    # Print detailed results
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2, default=str))
    
    return results


if __name__ == "__main__":
    asyncio.run(main()) 