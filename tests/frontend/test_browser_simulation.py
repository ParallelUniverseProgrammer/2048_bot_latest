#!/usr/bin/env python3
"""
Browser simulation test to reproduce frontend-backend interaction freezing issues.
This test simulates real browser behavior including resource constraints and timing issues.
"""

import sys
import os
import asyncio
import time
import json
import threading
from pathlib import Path
from typing import Dict, Any, List
import weakref
import gc

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backend.app.models.checkpoint_playback import CheckpointPlayback
from backend.app.models.checkpoint_metadata import CheckpointManager
from backend.app.api.websocket_manager import WebSocketManager

class BrowserSimulatedWebSocket:
    """WebSocket that simulates real browser behavior"""
    
    def __init__(self, slow_processing=False, memory_pressure=False, tab_switching=False):
        self.slow_processing = slow_processing
        self.memory_pressure = memory_pressure
        self.tab_switching = tab_switching
        self.messages = []
        self.closed = False
        self.message_queue = asyncio.Queue()
        self.processing_task = None
        self.is_visible = True
        self.memory_usage = 0
        self.max_memory = 100 * 1024 * 1024  # 100MB simulated limit
        
    async def accept(self):
        """Mock accept method"""
        pass
        
    async def send_text(self, message):
        """Simulate browser WebSocket send with potential issues"""
        if self.closed:
            raise Exception("WebSocket closed")
            
        # Simulate memory pressure
        if self.memory_pressure:
            self.memory_usage += len(message)
            if self.memory_usage > self.max_memory:
                print(f"WARNING:  Memory pressure detected: {self.memory_usage / 1024 / 1024:.1f}MB")
                # Simulate garbage collection pause
                await asyncio.sleep(0.1)
                self.memory_usage = self.memory_usage // 2  # Simulate GC
                
        # Simulate tab switching (background processing)
        if self.tab_switching and not self.is_visible:
            # Background tabs get throttled
            await asyncio.sleep(0.5)
            
        # Add to message queue for processing
        await self.message_queue.put(message)
        
        # Start processing task if not already running
        if not self.processing_task or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._process_messages())
            
    async def _process_messages(self):
        """Simulate frontend message processing"""
        while not self.message_queue.empty():
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                
                # Simulate slow processing (React state updates, DOM rendering)
                if self.slow_processing:
                    await asyncio.sleep(0.1)
                    
                # Parse and "process" the message like a real frontend would
                try:
                    data = json.loads(message)
                    await self._simulate_frontend_processing(data)
                except json.JSONDecodeError:
                    pass
                    
                self.messages.append(message)
                
            except asyncio.TimeoutError:
                break
                
    async def _simulate_frontend_processing(self, data: Dict[str, Any]):
        """Simulate frontend React state updates and DOM rendering"""
        message_type = data.get('type', 'unknown')
        
        if message_type == 'checkpoint_playback':
            # Simulate React state update storm
            await self._simulate_react_state_update()
            # Simulate DOM rendering
            await self._simulate_dom_rendering()
            
        elif message_type == 'playback_status':
            # Simulate loading state updates
            await self._simulate_loading_state_update()
            
        elif message_type == 'playback_heartbeat':
            # Simulate health monitoring
            await self._simulate_health_monitoring()
            
    async def _simulate_react_state_update(self):
        """Simulate React state updates that could cause performance issues"""
        # Simulate multiple state updates in quick succession
        for i in range(3):
            await asyncio.sleep(0.01)  # Simulate setState calls
            
        # Simulate useEffect chains
        await asyncio.sleep(0.02)
        
    async def _simulate_dom_rendering(self):
        """Simulate DOM rendering and animation"""
        # Simulate board tile updates
        await asyncio.sleep(0.05)
        
        # Simulate framer-motion animations
        if self.slow_processing:
            await asyncio.sleep(0.1)
            
    async def _simulate_loading_state_update(self):
        """Simulate loading state management"""
        await asyncio.sleep(0.01)
        
    async def _simulate_health_monitoring(self):
        """Simulate playback health monitoring"""
        await asyncio.sleep(0.005)
        
    def simulate_tab_switch(self, visible: bool):
        """Simulate browser tab switching"""
        self.is_visible = visible
        print(f"REFRESH: Tab switched: {'visible' if visible else 'background'}")

class BrowserSimulationTest:
    """Test class that simulates real browser conditions"""
    
    def __init__(self):
        checkpoint_dir = os.getenv('CHECKPOINTS_DIR', os.path.join(os.path.dirname(__file__), '..', 'backend', 'checkpoints'))
        print(f"[test_browser_simulation] Using checkpoint_dir: {checkpoint_dir}")
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.websocket_manager = WebSocketManager()
        self.playback = None
        self.test_timeout = 60
        self.simulated_clients = []
        
    async def create_simulated_browser_clients(self, scenarios: List[str]):
        """Create simulated browser clients with different scenarios"""
        clients = []
        
        for scenario in scenarios:
            if scenario == "normal":
                client = BrowserSimulatedWebSocket()
            elif scenario == "slow_processing":
                client = BrowserSimulatedWebSocket(slow_processing=True)
            elif scenario == "memory_pressure":
                client = BrowserSimulatedWebSocket(memory_pressure=True)
            elif scenario == "tab_switching":
                client = BrowserSimulatedWebSocket(tab_switching=True)
            elif scenario == "combined_stress":
                client = BrowserSimulatedWebSocket(slow_processing=True, memory_pressure=True, tab_switching=True)
            else:
                client = BrowserSimulatedWebSocket()
                
            await self.websocket_manager.connect(client, f"browser-sim-{scenario}")
            clients.append(client)
            
        self.simulated_clients = clients
        return clients
        
    async def simulate_frontend_state_storm(self):
        """Simulate frontend state update storms that could cause freezing"""
        print("STATUS:  Simulating frontend state update storm...")
        
        # Simulate rapid state updates like in a real React app
        tasks = []
        for i in range(10):
            task = asyncio.create_task(self._simulate_rapid_state_updates())
            tasks.append(task)
            
        await asyncio.gather(*tasks)
        
    async def _simulate_rapid_state_updates(self):
        """Simulate rapid React state updates"""
        for i in range(20):
            # Simulate setState calls
            await asyncio.sleep(0.001)
            
    async def simulate_resource_constraints(self):
        """Simulate system resource constraints"""
        print("STATUS: Simulating resource constraints...")
        
        # Simulate memory pressure
        memory_hog = []
        for i in range(1000):
            memory_hog.append([0] * 1000)  # Allocate memory
            await asyncio.sleep(0.001)
            
        # Simulate CPU pressure
        cpu_tasks = []
        for i in range(5):
            task = asyncio.create_task(self._cpu_intensive_task())
            cpu_tasks.append(task)
            
        await asyncio.gather(*cpu_tasks)
        
        # Clean up memory
        del memory_hog
        gc.collect()
        
    async def _cpu_intensive_task(self):
        """Simulate CPU-intensive task"""
        start_time = time.time()
        while time.time() - start_time < 1.0:
            # Simulate CPU work
            sum(range(1000))
            await asyncio.sleep(0.001)
            
    async def simulate_network_issues(self):
        """Simulate network connectivity issues"""
        print("STATUS: Simulating network issues...")
        
        # Simulate intermittent connectivity
        for client in self.simulated_clients:
            if hasattr(client, 'simulate_network_delay'):
                await client.simulate_network_delay(2.0)
                
    async def simulate_tab_switching_behavior(self):
        """Simulate browser tab switching behavior"""
        print("REFRESH: Simulating tab switching behavior...")
        
        # Switch tabs to background
        for client in self.simulated_clients:
            if hasattr(client, 'simulate_tab_switch'):
                client.simulate_tab_switch(False)
                
        await asyncio.sleep(5.0)  # Stay in background for 5 seconds
        
        # Switch back to foreground
        for client in self.simulated_clients:
            if hasattr(client, 'simulate_tab_switch'):
                client.simulate_tab_switch(True)
                
    async def test_checkpoint_playback_under_stress(self, checkpoint_id: str):
        """Test checkpoint playback under various stress conditions"""
        print(f"\nTESTING: Testing checkpoint playback under stress: {checkpoint_id}")
        
        try:
            # Create playback instance
            self.playback = CheckpointPlayback(self.checkpoint_manager)
            
            # Load checkpoint
            print("Loading checkpoint...")
            success = self.playback.load_checkpoint(checkpoint_id)
            if not success:
                return False, "Failed to load checkpoint"
                
            # Create simulated browser clients
            scenarios = ["normal", "slow_processing", "memory_pressure", "tab_switching", "combined_stress"]
            clients = await self.create_simulated_browser_clients(scenarios)
            print(f"Created {len(clients)} simulated browser clients")
            
            # Start playback
            print("Starting live playback...")
            playback_task = asyncio.create_task(
                self.playback.start_live_playback(self.websocket_manager)
            )
            
            # Run stress tests concurrently with playback
            stress_tasks = [
                asyncio.create_task(self.simulate_frontend_state_storm()),
                asyncio.create_task(self.simulate_resource_constraints()),
                asyncio.create_task(self.simulate_tab_switching_behavior()),
            ]
            
            # Monitor for 30 seconds
            try:
                await asyncio.wait_for(
                    asyncio.gather(playback_task, *stress_tasks),
                    timeout=30.0
                )
                return True, "Completed successfully"
                
            except asyncio.TimeoutError:
                print("ALARM: Test timed out - potential freeze detected!")
                
                # Try to stop playback gracefully
                self.playback.stop_playback()
                try:
                    await asyncio.wait_for(playback_task, timeout=5.0)
                    return False, "Timeout but stopped gracefully"
                except asyncio.TimeoutError:
                    return False, "Timeout and failed to stop - confirmed freeze!"
                    
        except Exception as e:
            return False, f"Exception: {str(e)}"
            
    async def test_memory_leak_detection(self):
        """Test for memory leaks in the playback system"""
        print("\nFIND: Testing for memory leaks...")
        
        checkpoints = self.checkpoint_manager.list_checkpoints()
        if not checkpoints:
            print("No checkpoints available")
            return
            
        checkpoint_id = checkpoints[0].id
        
        # Run multiple playback cycles to detect memory leaks
        for cycle in range(3):
            print(f"Memory leak test cycle {cycle + 1}/3")
            
            # Force garbage collection
            gc.collect()
            
            # Create fresh instances
            self.playback = CheckpointPlayback(self.checkpoint_manager)
            success = self.playback.load_checkpoint(checkpoint_id)
            
            if success:
                # Create client
                client = BrowserSimulatedWebSocket(memory_pressure=True)
                await self.websocket_manager.connect(client, f"memory-test-{cycle}")
                
                # Run short playback
                playback_task = asyncio.create_task(
                    self.playback.start_live_playback(self.websocket_manager)
                )
                
                try:
                    await asyncio.wait_for(playback_task, timeout=10.0)
                except asyncio.TimeoutError:
                    self.playback.stop_playback()
                    try:
                        await asyncio.wait_for(playback_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        pass
                        
                # Check simulated memory usage
                if hasattr(client, 'memory_usage'):
                    print(f"Cycle {cycle + 1} memory usage: {client.memory_usage / 1024 / 1024:.1f}MB")
                    
                # Disconnect client
                self.websocket_manager.disconnect(client)
                
            await asyncio.sleep(1.0)
            
    async def run_all_browser_simulation_tests(self):
        """Run all browser simulation tests"""
        print("STATUS: Starting Browser Simulation Tests")
        print("=" * 60)
        
        checkpoints = self.checkpoint_manager.list_checkpoints()
        if not checkpoints:
            print("No checkpoints available for testing")
            return
            
        results = []
        
        # Test first checkpoint under stress
        checkpoint_id = checkpoints[0].id
        print(f"Testing checkpoint: {checkpoint_id}")
        
        start_time = time.time()
        success, message = await self.test_checkpoint_playback_under_stress(checkpoint_id)
        duration = time.time() - start_time
        
        results.append({
            'test': 'Stress Test',
            'checkpoint': checkpoint_id,
            'success': success,
            'duration': duration,
            'message': message
        })
        
        # Test for memory leaks
        await self.test_memory_leak_detection()
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY: Browser Simulation Test Summary:")
        print("=" * 60)
        
        for result in results:
            status = "OK: PASS" if result['success'] else "ERROR: FAIL"
            print(f"{status} {result['test']}: {result['duration']:.2f}s")
            print(f"    Message: {result['message']}")
            
        print("\nTARGET: Browser Simulation Tests Complete")
        print("These tests simulate real browser conditions that could cause freezing.")

async def main():
    """Main test runner"""
    test = BrowserSimulationTest()
    await test.run_all_browser_simulation_tests()

if __name__ == "__main__":
    asyncio.run(main()) 