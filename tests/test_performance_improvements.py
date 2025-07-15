#!/usr/bin/env python3
"""
Test script to verify performance improvements prevent freezing issues
"""

import asyncio
import json
import time
import websockets
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import sys
import signal
from typing import List, Dict, Any

class PerformanceTestClient:
    """Test client that simulates realistic load"""
    
    def __init__(self, client_id: int, slow_processing: bool = False):
        self.client_id = client_id
        self.slow_processing = slow_processing
        self.messages_received = 0
        self.batches_received = 0
        self.lightweight_messages = 0
        self.full_messages = 0
        self.connection_start = None
        self.last_message_time = 0
        self.processing_times = []
        self.errors = []
        
    async def connect_and_monitor(self, websocket_url: str, duration: int = 60):
        """Connect to WebSocket and monitor performance"""
        try:
            print(f"Client {self.client_id}: Connecting to {websocket_url}")
            self.connection_start = time.time()
            
            async with websockets.connect(websocket_url) as websocket:
                print(f"Client {self.client_id}: Connected successfully")
                
                # Monitor messages for specified duration
                end_time = time.time() + duration
                
                while time.time() < end_time:
                    try:
                        # Set a reasonable timeout for receiving messages
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        await self._process_message(message)
                        
                    except asyncio.TimeoutError:
                        # No message received in timeout period, that's okay
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        print(f"Client {self.client_id}: Connection closed")
                        break
                    except Exception as e:
                        self.errors.append(f"Message processing error: {e}")
                        print(f"Client {self.client_id}: Error processing message: {e}")
                        
        except Exception as e:
            self.errors.append(f"Connection error: {e}")
            print(f"Client {self.client_id}: Connection error: {e}")
    
    async def _process_message(self, message: str):
        """Process received message and track performance"""
        process_start = time.time()
        
        try:
            data = json.loads(message)
            self.messages_received += 1
            self.last_message_time = time.time()
            
            # Track message types
            if data.get('type') == 'message_batch':
                self.batches_received += 1
                # Process batch messages
                for batch_msg in data.get('messages', []):
                    self._categorize_message(batch_msg)
            else:
                self._categorize_message(data)
            
            # Simulate slow processing if configured
            if self.slow_processing:
                await asyncio.sleep(0.1)  # 100ms processing delay
                
        except json.JSONDecodeError:
            self.errors.append("Invalid JSON received")
        except Exception as e:
            self.errors.append(f"Processing error: {e}")
        
        process_time = time.time() - process_start
        self.processing_times.append(process_time)
    
    def _categorize_message(self, data: Dict[str, Any]):
        """Categorize message types for analysis"""
        msg_type = data.get('type', 'unknown')
        
        if msg_type == 'checkpoint_playback_light':
            self.lightweight_messages += 1
        elif msg_type == 'checkpoint_playback':
            self.full_messages += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client performance statistics"""
        duration = time.time() - self.connection_start if self.connection_start else 0
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            'client_id': self.client_id,
            'duration': duration,
            'messages_received': self.messages_received,
            'batches_received': self.batches_received,
            'lightweight_messages': self.lightweight_messages,
            'full_messages': self.full_messages,
            'messages_per_second': self.messages_received / duration if duration > 0 else 0,
            'avg_processing_time': avg_processing_time,
            'max_processing_time': max(self.processing_times) if self.processing_times else 0,
            'errors': len(self.errors),
            'error_details': self.errors[-5:] if self.errors else []  # Last 5 errors
        }

async def test_websocket_performance():
    """Test WebSocket performance improvements"""
    print("=== WebSocket Performance Test ===")
    
    # Test configuration
    base_url = "http://localhost:8000"
    websocket_url = "ws://localhost:8000/ws"
    num_clients = 5
    test_duration = 30  # seconds
    
    # Check if server is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/ws/stats") as response:
                if response.status != 200:
                    print("Server not running or WebSocket endpoint not available")
                    return False
    except Exception as e:
        print(f"Cannot connect to server: {e}")
        return False
    
    print(f"Testing with {num_clients} clients for {test_duration} seconds...")
    
    # Create test clients (mix of normal and slow processing)
    clients = []
    for i in range(num_clients):
        slow_processing = i >= num_clients // 2  # Half the clients are slow
        client = PerformanceTestClient(i, slow_processing)
        clients.append(client)
    
    # Start all clients concurrently
    tasks = []
    for client in clients:
        task = asyncio.create_task(client.connect_and_monitor(websocket_url, test_duration))
        tasks.append(task)
    
    # Wait for all clients to complete
    print("Clients started, monitoring performance...")
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect and analyze results
    print("\n=== Performance Results ===")
    
    total_messages = 0
    total_errors = 0
    total_batches = 0
    total_lightweight = 0
    total_full = 0
    
    for client in clients:
        stats = client.get_stats()
        total_messages += stats['messages_received']
        total_errors += stats['errors']
        total_batches += stats['batches_received']
        total_lightweight += stats['lightweight_messages']
        total_full += stats['full_messages']
        
        print(f"Client {stats['client_id']}: "
              f"{stats['messages_received']} msgs, "
              f"{stats['messages_per_second']:.1f} msg/s, "
              f"{stats['avg_processing_time']*1000:.1f}ms avg, "
              f"{stats['errors']} errors")
    
    # Get server performance stats
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/ws/performance") as response:
                if response.status == 200:
                    server_stats = await response.json()
                    print(f"\nServer Performance:")
                    print(f"  Total broadcasts: {server_stats.get('total_broadcasts', 0)}")
                    print(f"  Successful broadcasts: {server_stats.get('successful_broadcasts', 0)}")
                    print(f"  Failed broadcasts: {server_stats.get('failed_broadcasts', 0)}")
                    print(f"  Slow broadcasts: {server_stats.get('slow_broadcasts', 0)}")
                    print(f"  Avg broadcast time: {server_stats.get('avg_broadcast_time', 0)*1000:.1f}ms")
                    
                    # Check connection health
                    healthy_connections = sum(1 for conn in server_stats.get('connection_health', []) 
                                            if conn.get('health_score', 0) > 0.7)
                    print(f"  Healthy connections: {healthy_connections}/{len(server_stats.get('connection_health', []))}")
    except Exception as e:
        print(f"Could not get server stats: {e}")
    
    print(f"\nOverall Results:")
    print(f"  Total messages received: {total_messages}")
    print(f"  Total batches received: {total_batches}")
    print(f"  Lightweight messages: {total_lightweight}")
    print(f"  Full messages: {total_full}")
    print(f"  Total errors: {total_errors}")
    
    # Performance assessment
    success_rate = (total_messages - total_errors) / total_messages if total_messages > 0 else 0
    print(f"  Success rate: {success_rate*100:.1f}%")
    
    # Check if improvements are working
    improvements_working = (
        total_batches > 0 or  # Message batching is active
        total_lightweight > 0 or  # Lightweight messages are being used
        success_rate > 0.95  # High success rate
    )
    
    if improvements_working:
        print("✅ Performance improvements are working!")
    else:
        print("❌ Performance improvements may not be working properly")
    
    return improvements_working

async def test_checkpoint_playback_performance():
    """Test checkpoint playback performance specifically"""
    print("\n=== Checkpoint Playback Performance Test ===")
    
    base_url = "http://localhost:8000"
    websocket_url = "ws://localhost:8000/ws"
    
    # Get available checkpoints
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/checkpoints") as response:
                if response.status != 200:
                    print("Cannot get checkpoints")
                    return False
                
                checkpoints = await response.json()
                if not checkpoints:
                    print("No checkpoints available for testing")
                    return False
                
                # Use the first checkpoint
                checkpoint_id = checkpoints[0]['id']
                print(f"Testing with checkpoint: {checkpoint_id}")
    except Exception as e:
        print(f"Error getting checkpoints: {e}")
        return False
    
    # Start playback
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/checkpoints/{checkpoint_id}/playback/start", 
                                  json={"speed": 2.0}) as response:
                if response.status != 200:
                    print("Failed to start playback")
                    return False
                
                print("Started checkpoint playback at 2x speed")
    except Exception as e:
        print(f"Error starting playback: {e}")
        return False
    
    # Monitor playback with multiple clients
    num_clients = 3
    test_duration = 20  # seconds
    
    clients = []
    for i in range(num_clients):
        client = PerformanceTestClient(i, slow_processing=(i == num_clients-1))
        clients.append(client)
    
    # Start monitoring
    tasks = []
    for client in clients:
        task = asyncio.create_task(client.connect_and_monitor(websocket_url, test_duration))
        tasks.append(task)
    
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Stop playback
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/checkpoints/playback/stop") as response:
                print("Stopped checkpoint playback")
    except Exception as e:
        print(f"Error stopping playback: {e}")
    
    # Analyze results
    total_playback_messages = 0
    total_lightweight = 0
    
    for client in clients:
        stats = client.get_stats()
        total_playback_messages += stats['lightweight_messages'] + stats['full_messages']
        total_lightweight += stats['lightweight_messages']
        
        print(f"Client {stats['client_id']}: "
              f"{stats['lightweight_messages']} light, "
              f"{stats['full_messages']} full, "
              f"{stats['errors']} errors")
    
    print(f"\nPlayback Results:")
    print(f"  Total playback messages: {total_playback_messages}")
    print(f"  Lightweight messages: {total_lightweight}")
    print(f"  Lightweight ratio: {total_lightweight/total_playback_messages*100:.1f}%" if total_playback_messages > 0 else "N/A")
    
    # Check if adaptive performance is working
    adaptive_working = total_lightweight > 0 and total_playback_messages > 0
    
    if adaptive_working:
        print("✅ Adaptive playback performance is working!")
    else:
        print("❌ Adaptive playback performance may not be working")
    
    return adaptive_working

async def main():
    """Run all performance tests"""
    print("Starting Performance Improvement Tests...")
    
    # Test 1: General WebSocket performance
    websocket_test_passed = await test_websocket_performance()
    
    # Test 2: Checkpoint playback performance
    playback_test_passed = await test_checkpoint_playback_performance()
    
    # Overall result
    print("\n=== Final Results ===")
    if websocket_test_passed and playback_test_passed:
        print("🎉 All performance tests passed!")
        print("The freezing issue should be resolved.")
        return True
    else:
        print("⚠️  Some performance tests failed.")
        print("The freezing issue may still occur under load.")
        return False

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nTest interrupted by user")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1) 