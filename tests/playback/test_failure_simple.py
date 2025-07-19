#!/usr/bin/env python3
"""
Test: Simple Checkpoint Playback Failure Detection
=================================================

This test reproduces the specific failure where:
1. User clicks "Watch" on a checkpoint (1500 episodes)
2. Frontend navigates to game tab and shows loading
3. Loading state gets stuck indefinitely
4. Page reload attempts fail entirely

Simplified version that works with existing infrastructure.
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
class SimpleFailureState:
    """Track the state of simple failure detection"""
    checkpoint_id: str
    frontend_simulation: Dict[str, Any]
    backend_responses: Dict[str, Any]
    failure_detected: bool = False
    failure_point: Optional[str] = None
    error_message: Optional[str] = None


class SimpleCheckpointFailureDetector:
    """Detect checkpoint playback failures using simple HTTP requests"""
    
    def __init__(self, logger: TestLogger = None):
        self.logger = logger or TestLogger()
        self.base_url = "http://localhost:8000"
        
    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp"""
        if level == "ERROR":
            self.logger.error(f"[SIMPLE_FAILURE] {message}")
        elif level == "WARNING":
            self.logger.warning(f"[SIMPLE_FAILURE] {message}")
        else:
            self.logger.log(f"[SIMPLE_FAILURE] {message}")
    
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
    
    def get_checkpoints(self) -> list:
        """Get list of available checkpoints"""
        self.log("Getting available checkpoints...")
        
        try:
            response = requests.get(f"{self.base_url}/checkpoints", timeout=10)
            if response.status_code == 200:
                checkpoints = response.json()
                self.log(f"Found {len(checkpoints)} checkpoints")
                
                # Log checkpoint details
                for checkpoint in checkpoints:
                    self.log(f"  - {checkpoint.get('id', 'unknown')}: Episode {checkpoint.get('episode', 'unknown')}")
                
                return checkpoints
            else:
                self.log(f"Failed to get checkpoints: HTTP {response.status_code}", "ERROR")
        except Exception as e:
            self.log(f"Error getting checkpoints: {e}", "ERROR")
        
        return []
    
    def find_checkpoint_with_1500_episodes(self) -> Optional[str]:
        """Find a checkpoint with 1500 episodes"""
        checkpoints = self.get_checkpoints()
        
        for checkpoint in checkpoints:
            if checkpoint.get('episode') == 1500:
                checkpoint_id = checkpoint.get('id')
                self.log(f"Found checkpoint with 1500 episodes: {checkpoint_id}")
                return checkpoint_id
        
        # If no 1500 episode checkpoint, use the first one
        if checkpoints:
            checkpoint_id = checkpoints[0].get('id')
            self.log(f"Using first available checkpoint: {checkpoint_id}")
            return checkpoint_id
        
        return None
    
    def simulate_frontend_click_watch(self, checkpoint_id: str) -> Dict[str, Any]:
        """Simulate the frontend "Watch" button click"""
        self.log(f"Simulating frontend click 'Watch' on checkpoint {checkpoint_id}")
        
        # Simulate the exact frontend flow from CheckpointManager.tsx
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
            'api_calls': []
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
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/checkpoints/{checkpoint_id}/playback/start",
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            end_time = time.time()
            
            api_call = {
                'url': f"/checkpoints/{checkpoint_id}/playback/start",
                'method': 'POST',
                'status_code': response.status_code,
                'response_time': end_time - start_time,
                'response_text': response.text[:200] + "..." if len(response.text) > 200 else response.text
            }
            
            if response.status_code == 200:
                api_call['success'] = True
                api_call['response_data'] = response.json()
                frontend_state['actions'].append({
                    'action': 'playback_start_api_success',
                    'timestamp': time.time(),
                    'details': f'API call successful in {api_call["response_time"]:.2f}s'
                })
            else:
                api_call['success'] = False
                frontend_state['actions'].append({
                    'action': 'playback_start_api_failed',
                    'timestamp': time.time(),
                    'details': f'API call failed: HTTP {response.status_code}'
                })
                # Clear loading state on error (CheckpointManager.tsx line 312-314)
                frontend_state['loading_states']['isPlaybackStarting'] = False
                frontend_state['loading_states']['loadingMessage'] = None
                
        except Exception as e:
            api_call = {
                'url': f"/checkpoints/{checkpoint_id}/playback/start",
                'method': 'POST',
                'success': False,
                'error': str(e)
            }
            frontend_state['actions'].append({
                'action': 'playback_start_api_exception',
                'timestamp': time.time(),
                'details': f'API call exception: {str(e)}'
            })
            # Clear loading state on error
            frontend_state['loading_states']['isPlaybackStarting'] = False
            frontend_state['loading_states']['loadingMessage'] = None
        
        frontend_state['api_calls'].append(api_call)
        
        self.log(f"Frontend simulation completed with {len(frontend_state['actions'])} actions")
        return frontend_state
    
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
                response = requests.get(f"{self.base_url}/checkpoints/playback/status", timeout=5)
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
                    if status.get('is_playing') and not status.get('current_checkpoint'):
                        failure_detected = True
                        failure_point = "no_checkpoint_loaded"
                        error_message = "Backend says playing but no checkpoint loaded"
                        break
                    
                    # Check if we're getting data
                    if status.get('is_playing'):
                        # Try to get current playback data
                        try:
                            data_response = requests.get(f"{self.base_url}/checkpoints/playback/current", timeout=5)
                            if data_response.status_code == 200:
                                data = data_response.json()
                                if not data.get('has_data'):
                                    time_since_start = time.time() - start_time
                                    if time_since_start > 10:  # 10 seconds without data
                                        failure_detected = True
                                        failure_point = "no_playback_data"
                                        error_message = f"No playback data for {time_since_start:.1f} seconds"
                                        break
                        except:
                            pass
                
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
            response = requests.get(f"{self.base_url}/checkpoints/playback/status", timeout=5)
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
                
                reload_result['reload_successful'] = len(reload_result['issues_detected']) == 0
                
            else:
                reload_result['issues_detected'].append(f"Status endpoint failed: HTTP {response.status_code}")
        
        except Exception as e:
            reload_result['issues_detected'].append(f"Reload simulation exception: {str(e)}")
        
        return reload_result
    
    async def detect_failure(self, checkpoint_id: str = None) -> SimpleFailureState:
        """Detect failure in the checkpoint playback system"""
        self.log("=" * 60)
        self.log("SIMPLE CHECKPOINT FAILURE DETECTION")
        self.log("=" * 60)
        
        # Step 1: Check backend availability
        if not self.check_backend_availability():
            return SimpleFailureState(
                checkpoint_id="",
                frontend_simulation={},
                backend_responses={},
                failure_detected=True,
                failure_point="backend_unavailable",
                error_message="Backend is not available"
            )
        
        # Step 2: Find checkpoint with 1500 episodes
        if not checkpoint_id:
            checkpoint_id = self.find_checkpoint_with_1500_episodes()
            if not checkpoint_id:
                return SimpleFailureState(
                    checkpoint_id="",
                    frontend_simulation={},
                    backend_responses={},
                    failure_detected=True,
                    failure_point="no_checkpoint_found",
                    error_message="No checkpoint found"
                )
        
        # Step 3: Simulate frontend click
        self.log(f"\nStep 3: Simulating frontend click on checkpoint {checkpoint_id}")
        frontend_state = self.simulate_frontend_click_watch(checkpoint_id)
        
        # Step 4: Monitor for failures
        self.log("\nStep 4: Monitoring for failures...")
        monitoring_result = self.monitor_playback_status(timeout_seconds=60)
        
        # Step 5: Check for stuck loading state
        loading_stuck = self.check_loading_state_stuck(frontend_state)
        
        # Step 6: Simulate page reload
        reload_result = self.simulate_page_reload_scenario()
        
        # Step 7: Determine if failure occurred
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
        failure_state = SimpleFailureState(
            checkpoint_id=checkpoint_id,
            frontend_simulation=frontend_state,
            backend_responses={
                'monitoring_result': monitoring_result,
                'reload_result': reload_result
            },
            failure_detected=failure_detected,
            failure_point=failure_point,
            error_message=error_message
        )
        
        return failure_state
    
    def generate_failure_report(self, failure_state: SimpleFailureState) -> Dict[str, Any]:
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
                'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'frontend_analysis': {
                'actions_performed': len(failure_state.frontend_simulation.get('actions', [])),
                'loading_state_stuck': failure_state.frontend_simulation.get('loading_states', {}).get('isPlaybackStarting', False),
                'navigation_successful': failure_state.frontend_simulation.get('navigation', {}).get('to_tab') == 'game',
                'api_calls': failure_state.frontend_simulation.get('api_calls', [])
            },
            'backend_analysis': {
                'monitoring_result': failure_state.backend_responses.get('monitoring_result', {}),
                'reload_result': failure_state.backend_responses.get('reload_result', {})
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
        elif failure_state.failure_point == "no_playback_data":
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
    
    async def run_complete_detection(self) -> Dict[str, Any]:
        """Run the complete failure detection"""
        try:
            # Detect failure
            failure_state = await self.detect_failure()
            
            # Generate report
            report = self.generate_failure_report(failure_state)
            
            # Print summary
            self.log("\n" + "=" * 60)
            self.log("DETECTION SUMMARY")
            self.log("=" * 60)
            
            if failure_state.failure_detected:
                self.log("❌ FAILURE DETECTED", "ERROR")
                self.log(f"Failure point: {failure_state.failure_point}")
                self.log(f"Error: {failure_state.error_message}")
                self.log(f"Recommendations: {len(report['recommendations'])}")
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
    logger.log("Simple Checkpoint Playback Failure Detection")
    logger.log("=" * 60)
    
    detector = SimpleCheckpointFailureDetector(logger=logger)
    results = await detector.run_complete_detection()
    
    # Print detailed results
    logger.separator()
    logger.info("DETAILED RESULTS")
    logger.info("=" * 60)
    logger.info("Detailed results available in logs")
    
    return results


if __name__ == "__main__":
    asyncio.run(main()) 