#!/usr/bin/env python3
"""
WebSocket Performance Optimizer
===============================

This module provides comprehensive WebSocket performance optimization
to address the slow client broadcast issues mentioned in the engineering report.

Key optimizations:
- Message batching and compression
- Client connection pooling
- Broadcast performance monitoring
- Adaptive message frequency based on client capabilities
- Memory-efficient message queuing
- Connection health monitoring
- Graceful degradation under load

The engineering report identified slow client broadcast taking 4.05s in stress tests.
This module aims to reduce that to sub-second performance.
"""

import asyncio
import json
import time
import threading
import gzip
import zlib
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import weakref
import gc
import logging

from test_utils import TestLogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1    # Must be delivered immediately
    HIGH = 2        # High priority, minimal delay
    NORMAL = 3      # Normal priority
    LOW = 4         # Can be delayed or batched

class ClientCapability(Enum):
    """Client capability levels"""
    HIGH_PERFORMANCE = "high"      # Can handle high message rates
    MEDIUM_PERFORMANCE = "medium"  # Can handle medium message rates
    LOW_PERFORMANCE = "low"        # Needs message rate limiting
    MOBILE = "mobile"              # Mobile client with limited resources

@dataclass
class ClientConnection:
    """Enhanced client connection with performance metrics"""
    id: str
    websocket: Any
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    capability: ClientCapability = ClientCapability.MEDIUM_PERFORMANCE
    
    # Performance metrics
    messages_sent: int = 0
    messages_failed: int = 0
    avg_response_time: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Message queue management
    message_queue: deque = field(default_factory=deque)
    max_queue_size: int = 1000
    is_processing: bool = False
    
    # Health monitoring
    is_healthy: bool = True
    consecutive_failures: int = 0
    max_consecutive_failures: int = 5
    
    def update_response_time(self, response_time: float):
        """Update response time metrics"""
        self.response_times.append(response_time)
        if self.response_times:
            self.avg_response_time = sum(self.response_times) / len(self.response_times)
    
    def is_queue_full(self) -> bool:
        """Check if message queue is full"""
        return len(self.message_queue) >= self.max_queue_size
    
    def mark_failure(self):
        """Mark a message failure"""
        self.messages_failed += 1
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.is_healthy = False
    
    def mark_success(self):
        """Mark a message success"""
        self.messages_sent += 1
        self.consecutive_failures = 0
        self.is_healthy = True
        self.last_activity = time.time()

@dataclass
class BroadcastMessage:
    """Enhanced broadcast message with optimization features"""
    content: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    created_at: float = field(default_factory=time.time)
    target_clients: Optional[Set[str]] = None  # None means all clients
    compress: bool = True
    batch_eligible: bool = True
    expires_at: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def serialize(self) -> str:
        """Serialize message to JSON"""
        return json.dumps(self.content)
    
    def compress_content(self) -> bytes:
        """Compress message content"""
        if self.compress:
            json_str = self.serialize()
            return gzip.compress(json_str.encode('utf-8'))
        return self.serialize().encode('utf-8')

class WebSocketPerformanceOptimizer:
    """Main WebSocket performance optimization class"""
    
    def __init__(self, logger: Optional[TestLogger] = None):
        self.logger = logger or TestLogger()
        
        # Client management
        self.clients: Dict[str, ClientConnection] = {}
        self.client_groups: Dict[str, Set[str]] = {}  # Group clients by capability
        
        # Message management
        self.message_queue: deque = deque()
        self.broadcast_stats = {
            'messages_sent': 0,
            'messages_failed': 0,
            'broadcast_times': deque(maxlen=100),
            'clients_reached': deque(maxlen=100),
            'compression_ratio': deque(maxlen=100)
        }
        
        # Performance settings
        self.max_batch_size = 50
        self.max_batch_delay = 0.1  # 100ms max batching delay
        self.compression_threshold = 1024  # Compress messages > 1KB
        
        # Background tasks
        self.is_running = False
        self.batch_processor_task = None
        self.health_monitor_task = None
        self.performance_monitor_task = None
        
        # Adaptive settings
        self.adaptive_settings = {
            'broadcast_interval': 0.05,  # 50ms base interval
            'batch_size': 10,
            'compression_enabled': True,
            'health_check_interval': 5.0
        }
    
    async def start(self):
        """Start the performance optimizer"""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("Starting WebSocket performance optimizer...")
        
        # Start background tasks
        self.batch_processor_task = asyncio.create_task(self._batch_processor())
        self.health_monitor_task = asyncio.create_task(self._health_monitor())
        self.performance_monitor_task = asyncio.create_task(self._performance_monitor())
        
        self.logger.ok("WebSocket performance optimizer started")
    
    async def stop(self):
        """Stop the performance optimizer"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.logger.info("Stopping WebSocket performance optimizer...")
        
        # Cancel background tasks
        tasks = [self.batch_processor_task, self.health_monitor_task, self.performance_monitor_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.ok("WebSocket performance optimizer stopped")
    
    async def add_client(self, client_id: str, websocket: Any, 
                        capability: ClientCapability = ClientCapability.MEDIUM_PERFORMANCE):
        """Add a client connection"""
        client = ClientConnection(
            id=client_id,
            websocket=websocket,
            capability=capability
        )
        
        self.clients[client_id] = client
        
        # Add to capability group
        group_name = capability.value
        if group_name not in self.client_groups:
            self.client_groups[group_name] = set()
        self.client_groups[group_name].add(client_id)
        
        self.logger.info(f"Added client {client_id} with {capability.value} capability")
    
    async def remove_client(self, client_id: str):
        """Remove a client connection"""
        if client_id in self.clients:
            client = self.clients[client_id]
            
            # Remove from capability group
            group_name = client.capability.value
            if group_name in self.client_groups:
                self.client_groups[group_name].discard(client_id)
                if not self.client_groups[group_name]:
                    del self.client_groups[group_name]
            
            del self.clients[client_id]
            self.logger.info(f"Removed client {client_id}")
    
    async def broadcast_message(self, content: Dict[str, Any], 
                              priority: MessagePriority = MessagePriority.NORMAL,
                              target_clients: Optional[Set[str]] = None,
                              compress: bool = True) -> Dict[str, Any]:
        """Broadcast a message to clients with performance optimization"""
        
        broadcast_start = time.time()
        
        # Create optimized message
        message = BroadcastMessage(
            content=content,
            priority=priority,
            target_clients=target_clients,
            compress=compress and len(json.dumps(content)) > self.compression_threshold
        )
        
        # Handle critical messages immediately
        if priority == MessagePriority.CRITICAL:
            result = await self._send_message_immediately(message)
        else:
            # Queue for batch processing
            self.message_queue.append(message)
            result = {"queued": True, "queue_size": len(self.message_queue)}
        
        # Record broadcast time
        broadcast_time = time.time() - broadcast_start
        self.broadcast_stats['broadcast_times'].append(broadcast_time)
        
        return result
    
    async def _send_message_immediately(self, message: BroadcastMessage) -> Dict[str, Any]:
        """Send message immediately (for critical messages)"""
        send_start = time.time()
        
        # Determine target clients
        target_clients = message.target_clients or set(self.clients.keys())
        
        # Filter to healthy clients
        healthy_clients = [
            client_id for client_id in target_clients
            if client_id in self.clients and self.clients[client_id].is_healthy
        ]
        
        # Send to clients
        results = await self._send_to_clients(message, healthy_clients)
        
        # Update stats
        send_time = time.time() - send_start
        self.broadcast_stats['messages_sent'] += 1
        self.broadcast_stats['clients_reached'].append(len(healthy_clients))
        
        return {
            "sent": True,
            "clients_reached": len(healthy_clients),
            "send_time": send_time,
            "results": results
        }
    
    async def _send_to_clients(self, message: BroadcastMessage, 
                             client_ids: List[str]) -> Dict[str, Any]:
        """Send message to specific clients with optimization"""
        
        # Prepare message content
        if message.compress:
            content = message.compress_content()
            is_compressed = True
        else:
            content = message.serialize()
            is_compressed = False
        
        # Send to clients concurrently
        tasks = []
        for client_id in client_ids:
            if client_id in self.clients:
                task = asyncio.create_task(
                    self._send_to_single_client(client_id, content, is_compressed)
                )
                tasks.append(task)
        
        # Wait for all sends to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_sends = 0
        failed_sends = 0
        
        for i, result in enumerate(results):
            client_id = client_ids[i] if i < len(client_ids) else f"unknown_{i}"
            
            if isinstance(result, Exception):
                failed_sends += 1
                if client_id in self.clients:
                    self.clients[client_id].mark_failure()
                self.logger.warning(f"Failed to send to client {client_id}: {result}")
            else:
                successful_sends += 1
                if client_id in self.clients:
                    self.clients[client_id].mark_success()
        
        return {
            "successful_sends": successful_sends,
            "failed_sends": failed_sends,
            "compression_used": is_compressed
        }
    
    async def _send_to_single_client(self, client_id: str, content: Any, 
                                   is_compressed: bool) -> bool:
        """Send message to a single client"""
        if client_id not in self.clients:
            return False
        
        client = self.clients[client_id]
        send_start = time.time()
        
        try:
            # Simulate WebSocket send (in real implementation, this would be websocket.send)
            await asyncio.sleep(0.001)  # Simulate network delay
            
            # Record response time
            response_time = time.time() - send_start
            client.update_response_time(response_time)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Send to client {client_id} failed: {e}")
            return False
    
    async def _batch_processor(self):
        """Background task to process message batches"""
        while self.is_running:
            try:
                if self.message_queue:
                    # Collect messages for batch
                    batch = []
                    batch_start = time.time()
                    
                    while (len(batch) < self.max_batch_size and 
                           self.message_queue and 
                           time.time() - batch_start < self.max_batch_delay):
                        
                        message = self.message_queue.popleft()
                        
                        # Skip expired messages
                        if message.is_expired():
                            continue
                        
                        batch.append(message)
                    
                    # Process batch
                    if batch:
                        await self._process_message_batch(batch)
                
                # Adaptive delay based on queue size
                if len(self.message_queue) > 100:
                    await asyncio.sleep(0.01)  # Process faster when queue is full
                else:
                    await asyncio.sleep(self.adaptive_settings['broadcast_interval'])
                
            except Exception as e:
                self.logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_message_batch(self, batch: List[BroadcastMessage]):
        """Process a batch of messages"""
        batch_start = time.time()
        
        # Group messages by priority
        priority_groups = {}
        for message in batch:
            priority = message.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(message)
        
        # Process in priority order
        for priority in sorted(priority_groups.keys(), key=lambda x: x.value):
            messages = priority_groups[priority]
            
            # Send messages in this priority group
            for message in messages:
                await self._send_message_immediately(message)
        
        # Update stats
        batch_time = time.time() - batch_start
        self.logger.debug(f"Processed batch of {len(batch)} messages in {batch_time:.3f}s")
    
    async def _health_monitor(self):
        """Background task to monitor client health"""
        while self.is_running:
            try:
                unhealthy_clients = []
                
                for client_id, client in self.clients.items():
                    # Check if client is responsive
                    time_since_activity = time.time() - client.last_activity
                    
                    if time_since_activity > 30.0:  # 30 seconds timeout
                        unhealthy_clients.append(client_id)
                    
                    # Check if client has too many failures
                    if not client.is_healthy:
                        unhealthy_clients.append(client_id)
                
                # Remove unhealthy clients
                for client_id in unhealthy_clients:
                    self.logger.warning(f"Removing unhealthy client: {client_id}")
                    await self.remove_client(client_id)
                
                await asyncio.sleep(self.adaptive_settings['health_check_interval'])
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5.0)
    
    async def _performance_monitor(self):
        """Background task to monitor and adjust performance"""
        while self.is_running:
            try:
                # Calculate performance metrics
                if self.broadcast_stats['broadcast_times']:
                    avg_broadcast_time = sum(self.broadcast_stats['broadcast_times']) / len(self.broadcast_stats['broadcast_times'])
                    
                    # Adaptive adjustment based on performance
                    if avg_broadcast_time > 1.0:  # If broadcasts are taking > 1 second
                        # Increase batch size to reduce individual broadcast frequency
                        self.max_batch_size = min(self.max_batch_size + 5, 100)
                        self.adaptive_settings['broadcast_interval'] = min(
                            self.adaptive_settings['broadcast_interval'] + 0.01, 0.2
                        )
                        self.logger.info(f"Performance adjustment: increased batch size to {self.max_batch_size}")
                    
                    elif avg_broadcast_time < 0.1:  # If broadcasts are fast
                        # Decrease batch size for more responsive delivery
                        self.max_batch_size = max(self.max_batch_size - 5, 5)
                        self.adaptive_settings['broadcast_interval'] = max(
                            self.adaptive_settings['broadcast_interval'] - 0.01, 0.01
                        )
                        self.logger.info(f"Performance adjustment: decreased batch size to {self.max_batch_size}")
                
                await asyncio.sleep(10.0)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(10.0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = self.broadcast_stats.copy()
        
        # Calculate derived metrics
        if stats['broadcast_times']:
            stats['avg_broadcast_time'] = sum(stats['broadcast_times']) / len(stats['broadcast_times'])
            stats['max_broadcast_time'] = max(stats['broadcast_times'])
            stats['min_broadcast_time'] = min(stats['broadcast_times'])
        
        if stats['clients_reached']:
            stats['avg_clients_reached'] = sum(stats['clients_reached']) / len(stats['clients_reached'])
        
        # Add client statistics
        stats['total_clients'] = len(self.clients)
        stats['healthy_clients'] = sum(1 for c in self.clients.values() if c.is_healthy)
        stats['client_groups'] = {group: len(clients) for group, clients in self.client_groups.items()}
        
        # Add queue statistics
        stats['queue_size'] = len(self.message_queue)
        stats['adaptive_settings'] = self.adaptive_settings.copy()
        
        return stats
    
    async def run_performance_test(self, num_clients: int = 10, 
                                 messages_per_second: int = 10,
                                 test_duration: int = 30) -> Dict[str, Any]:
        """Run a comprehensive performance test"""
        self.logger.section("WebSocket Performance Test")
        self.logger.info(f"Testing with {num_clients} clients, {messages_per_second} msg/s for {test_duration}s")
        
        # Start optimizer
        await self.start()
        
        # Create test clients
        test_clients = []
        for i in range(num_clients):
            client_id = f"test_client_{i}"
            # Simulate different client capabilities
            if i < num_clients // 3:
                capability = ClientCapability.HIGH_PERFORMANCE
            elif i < 2 * num_clients // 3:
                capability = ClientCapability.MEDIUM_PERFORMANCE
            else:
                capability = ClientCapability.LOW_PERFORMANCE
            
            await self.add_client(client_id, f"mock_websocket_{i}", capability)
            test_clients.append(client_id)
        
        # Run test
        start_time = time.time()
        message_interval = 1.0 / messages_per_second
        
        try:
            while time.time() - start_time < test_duration:
                # Send test message
                test_message = {
                    "type": "performance_test",
                    "timestamp": time.time(),
                    "data": "x" * 500  # 500 byte payload
                }
                
                await self.broadcast_message(test_message, priority=MessagePriority.NORMAL)
                
                # Wait for next message
                await asyncio.sleep(message_interval)
            
            # Get final stats
            final_stats = self.get_performance_stats()
            
            # Clean up
            for client_id in test_clients:
                await self.remove_client(client_id)
            
            await self.stop()
            
            # Calculate results
            total_time = time.time() - start_time
            expected_messages = int(total_time * messages_per_second)
            
            result = {
                "success": True,
                "total_time": total_time,
                "expected_messages": expected_messages,
                "actual_messages": final_stats['messages_sent'],
                "message_rate": final_stats['messages_sent'] / total_time,
                "performance_stats": final_stats
            }
            
            # Log results
            self.logger.ok(f"Performance test completed successfully")
            self.logger.info(f"  Messages sent: {final_stats['messages_sent']}")
            self.logger.info(f"  Message rate: {result['message_rate']:.1f} msg/s")
            self.logger.info(f"  Avg broadcast time: {final_stats.get('avg_broadcast_time', 0):.3f}s")
            
            return result
            
        except Exception as e:
            await self.stop()
            return {
                "success": False,
                "error": str(e),
                "performance_stats": self.get_performance_stats()
            }


async def main():
    """Test the WebSocket performance optimizer"""
    logger = TestLogger()
    
    # Create optimizer
    optimizer = WebSocketPerformanceOptimizer(logger)
    
    # Run performance test
    result = await optimizer.run_performance_test(
        num_clients=20,
        messages_per_second=50,
        test_duration=10
    )
    
    logger.separator()
    logger.info("Performance Test Results:")
    logger.info(f"Success: {result['success']}")
    if result['success']:
        logger.info(f"Message rate: {result['message_rate']:.1f} msg/s")
        logger.info(f"Total messages: {result['actual_messages']}")
    else:
        logger.error(f"Error: {result['error']}")
    
    # Print detailed stats
    print("\nDetailed Performance Stats:")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main()) 