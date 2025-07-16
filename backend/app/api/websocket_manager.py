"""
WebSocket manager for handling real-time connections with mobile optimizations and performance fixes
"""
import json
import asyncio
import re
import time
from typing import List, Dict, Any, Optional, Deque
from collections import deque
from dataclasses import dataclass, field
from fastapi import WebSocket
from rich.console import Console

console = Console()

@dataclass
class MessageBatch:
    """Batch of messages to send together"""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    total_size: int = 0
    
    def add_message(self, message: Dict[str, Any]) -> bool:
        """Add message to batch, return True if batch should be sent"""
        message_size = len(json.dumps(message))
        
        # Send batch if it would exceed size limit or time limit
        if (self.total_size + message_size > 50000 or  # 50KB limit
            len(self.messages) >= 10 or  # 10 message limit
            time.time() - self.created_at > 0.5):  # 500ms time limit
            return True
        
        self.messages.append(message)
        self.total_size += message_size
        return False

@dataclass
class ConnectionHealth:
    """Track connection health and performance"""
    consecutive_failures: int = 0
    last_success: float = field(default_factory=time.time)
    last_failure: float = 0
    total_messages_sent: int = 0
    total_failures: int = 0
    avg_send_time: float = 0
    is_circuit_open: bool = False
    circuit_open_until: float = 0
    slow_send_count: int = 0
    is_mobile: bool = False
    
    def record_success(self, send_time: float):
        """Record successful send"""
        self.consecutive_failures = 0
        self.last_success = time.time()
        self.total_messages_sent += 1
        
        # Update average send time (exponential moving average)
        if self.avg_send_time == 0:
            self.avg_send_time = send_time
        else:
            self.avg_send_time = 0.8 * self.avg_send_time + 0.2 * send_time
        
        # Close circuit if it was open and we're successful
        if self.is_circuit_open and time.time() > self.circuit_open_until:
            self.is_circuit_open = False
            console.print(f"[green]Circuit breaker closed for connection {id(self)}")
    
    def record_failure(self, send_time: float):
        """Record failed send"""
        self.consecutive_failures += 1
        self.last_failure = time.time()
        self.total_failures += 1
        
        # Open circuit breaker if too many consecutive failures
        # Be more lenient with mobile devices
        failure_threshold = 10 if self.is_mobile else 5
        if self.consecutive_failures >= failure_threshold:
            self.is_circuit_open = True
            # Shorter circuit breaker for mobile devices
            circuit_duration = 15.0 if self.is_mobile else 30.0
            self.circuit_open_until = time.time() + circuit_duration
            console.print(f"[red]Circuit breaker opened for connection {id(self)} - too many failures ({self.consecutive_failures})")
    
    def record_slow_send(self):
        """Record slow send"""
        self.slow_send_count += 1
        
        # Open circuit if too many slow sends - more lenient for mobile
        slow_send_threshold = 20 if self.is_mobile else 10
        if self.slow_send_count >= slow_send_threshold:
            self.is_circuit_open = True
            # Shorter circuit breaker for mobile devices
            circuit_duration = 10.0 if self.is_mobile else 15.0
            self.circuit_open_until = time.time() + circuit_duration
            console.print(f"[yellow]Circuit breaker opened for connection {id(self)} - too many slow sends ({self.slow_send_count})")
    
    def should_send(self) -> bool:
        """Check if we should send to this connection"""
        if self.is_circuit_open:
            return time.time() > self.circuit_open_until
        return True
    
    def get_health_score(self) -> float:
        """Get health score (0-1, higher is better)"""
        if self.total_messages_sent == 0:
            return 1.0
        
        failure_rate = self.total_failures / self.total_messages_sent
        recent_failure_penalty = min(self.consecutive_failures * 0.1, 0.5)
        slow_send_penalty = min(self.slow_send_count * 0.05, 0.3)
        
        return max(0, 1.0 - failure_rate - recent_failure_penalty - slow_send_penalty)

class ConnectionInfo:
    def __init__(self, websocket: WebSocket, user_agent: str = ""):
        self.websocket = websocket
        self.user_agent = user_agent
        self.is_mobile = self._detect_mobile(user_agent)
        self.is_mobile_safari = self._detect_mobile_safari(user_agent)
        self.connected_at = time.time()
        self.last_heartbeat = time.time()
        self.connection_quality = "unknown"
        self.retry_count = 0
        self.health = ConnectionHealth(is_mobile=self.is_mobile)
        self.message_batch = MessageBatch()
        self.last_batch_send = int(time.time())
        
    def _detect_mobile(self, user_agent: str) -> bool:
        """Detect if the connection is from a mobile device"""
        mobile_patterns = [
            r'Mobile', r'Android', r'iPhone', r'iPad', r'iPod',
            r'BlackBerry', r'IEMobile', r'Opera Mini', r'webOS'
        ]
        return any(re.search(pattern, user_agent, re.IGNORECASE) for pattern in mobile_patterns)
    
    def _detect_mobile_safari(self, user_agent: str) -> bool:
        """Detect if the connection is from Mobile Safari"""
        return (self.is_mobile and 
                re.search(r'Safari', user_agent, re.IGNORECASE) and 
                not re.search(r'Chrome', user_agent, re.IGNORECASE))
    
    def get_adaptive_timeout(self) -> float:
        """Get adaptive timeout based on device type and health"""
        base_timeout = 5.0 if self.is_mobile_safari else 3.0 if self.is_mobile else 1.0
        
        # Increase timeout for unhealthy connections
        health_score = self.health.get_health_score()
        return base_timeout * (2.0 - health_score)  # Scale from 1x to 2x based on health
    
    def get_heartbeat_interval(self) -> float:
        """Get heartbeat interval based on device type"""
        if self.is_mobile_safari:
            return 10.0  # Less frequent heartbeats for Mobile Safari
        elif self.is_mobile:
            return 5.0   # Medium frequency for mobile devices
        else:
            return 2.0   # Frequent heartbeats for desktop
    
    def should_use_connection_pooling(self) -> bool:
        """Determine if connection pooling should be used"""
        return bool(self.is_mobile)  # Always return a bool

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[ConnectionInfo] = []
        self.connection_count = 0
        self.mobile_connections = 0
        self.desktop_connections = 0
        self.connection_pool: Dict[str, List[ConnectionInfo]] = {
            "mobile": [],
            "desktop": []
        }
        self.max_connections_per_pool = 10
        self.heartbeat_tasks: Dict[WebSocket, asyncio.Task] = {}
        
        # Performance settings
        self.write_timeout = 2.0  # seconds
        self.max_heartbeat_lag = 30.0  # seconds
        self.batch_send_interval = 0.1  # seconds
        self.max_concurrent_sends = 5  # limit concurrent sends
        
        # Rate limiting
        self.broadcast_rate_limiter = {
            'last_broadcast': int(0),
            'min_interval': 0.05,  # 50ms minimum between broadcasts
            'message_queue': deque(maxlen=100),
            'queue_process_task': None
        }
        
        # Performance monitoring
        self.performance_stats = {
            'total_broadcasts': 0,
            'successful_broadcasts': 0,
            'failed_broadcasts': 0,
            'avg_broadcast_time': 0.0,  # Changed to float
            'slow_broadcasts': 0,
            'circuit_breaker_activations': 0
        }
        
        # Initialize batch processor task as None - will be started later
        self.batch_processor_task = None
    
    def start_batch_processor(self):
        """Start the batch processor task - call this after the event loop is running"""
        if self.batch_processor_task is None:
            self._start_batch_processor()
    
    def _start_batch_processor(self):
        """Start background task to process message batches"""
        async def batch_processor():
            while True:
                try:
                    await asyncio.sleep(self.batch_send_interval)
                    await self._process_message_batches()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    console.print(f"[red]Error in batch processor: {e}")
        
        self.batch_processor_task = asyncio.create_task(batch_processor())
    
    async def _process_message_batches(self):
        """Process pending message batches"""
        current_time = time.time()
        
        for conn in list(self.active_connections):
            # Send batch if it has messages and enough time has passed
            if (conn.message_batch.messages and 
                current_time - conn.last_batch_send > self.batch_send_interval):
                await self._send_batch(conn)

    async def connect(self, websocket: WebSocket, user_agent: str = ""):
        """Accept a new WebSocket connection with mobile optimization"""
        await websocket.accept()
        
        conn_info = ConnectionInfo(websocket, user_agent)
        self.active_connections.append(conn_info)
        self.connection_count += 1
        
        # Update connection counts by type
        if conn_info.is_mobile:
            self.mobile_connections += 1
        else:
            self.desktop_connections += 1
        
        # Add to appropriate connection pool
        pool_key = "mobile" if conn_info.is_mobile else "desktop"
        if len(self.connection_pool[pool_key]) < self.max_connections_per_pool:
            self.connection_pool[pool_key].append(conn_info)
        
        console.print(f"[green]WebSocket connected ({pool_key}). "
                     f"Total: {len(self.active_connections)} "
                     f"(Mobile: {self.mobile_connections}, Desktop: {self.desktop_connections})")
        
        # Send initial connection message with mobile-specific info
        await self.send_personal_message({
            "type": "connection_established",
            "message": "Connected to training server",
            "timestamp": asyncio.get_event_loop().time(),
            "mobile_optimized": conn_info.is_mobile,
            "connection_type": "mobile_safari" if conn_info.is_mobile_safari else 
                              "mobile" if conn_info.is_mobile else "desktop"
        }, websocket)
        
        # Start adaptive heartbeat for this connection
        self._start_adaptive_heartbeat(conn_info)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        conn_info = None
        for conn in self.active_connections:
            if conn.websocket == websocket:
                conn_info = conn
                break
        
        if conn_info:
            # Send any pending batch before disconnecting
            if conn_info.message_batch.messages:
                try:
                    asyncio.create_task(self._send_batch(conn_info))
                except Exception:
                    pass  # Ignore errors during disconnect
            
            self.active_connections.remove(conn_info)
            
            # Update connection counts
            if conn_info.is_mobile:
                self.mobile_connections -= 1
            else:
                self.desktop_connections -= 1
            
            # Remove from connection pool
            pool_key = "mobile" if conn_info.is_mobile else "desktop"
            if conn_info in self.connection_pool[pool_key]:
                self.connection_pool[pool_key].remove(conn_info)
            
            # Stop heartbeat task
            if websocket in self.heartbeat_tasks:
                self.heartbeat_tasks[websocket].cancel()
                del self.heartbeat_tasks[websocket]
            
            console.print(f"[yellow]WebSocket disconnected. "
                         f"Total: {len(self.active_connections)} "
                         f"(Mobile: {self.mobile_connections}, Desktop: {self.desktop_connections})")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            console.print(f"[red]Error sending personal message: {e}")
            self.disconnect(websocket)
    
    # ----------------------------- internal helpers -------------------------
    async def _safe_send(self, conn: ConnectionInfo, message_str: str):
        """Send with timeout and circuit breaker; disconnect on failure. Added timing diagnostics."""
        if not conn.health.should_send():
            return  # Circuit breaker is open
        
        start_time = time.perf_counter()
        try:
            # CRITICAL FIX: Use shorter timeout during training to prevent blocking
            timeout = min(conn.get_adaptive_timeout(), 0.5)  # Cap at 500ms during training
            await asyncio.wait_for(conn.websocket.send_text(message_str), timeout=timeout)
            
            duration = time.perf_counter() - start_time
            conn.health.record_success(duration)
            
            if duration > 0.5:
                conn.health.record_slow_send()
                console.print(f"[yellow]Slow send: {duration:.3f}s – client {id(conn.websocket)}")
                
        except asyncio.TimeoutError:
            duration = time.perf_counter() - start_time
            conn.health.record_failure(duration)
            console.print(f"[red]Send timeout after {duration:.3f}s – client {id(conn.websocket)}")
            
            # CRITICAL FIX: Don't disconnect on timeout, let circuit breaker handle it
            # This prevents cascading failures during training
        except Exception as e:
            duration = time.perf_counter() - start_time
            conn.health.record_failure(duration)
            console.print(f"[red]Error sending to connection after {duration:.3f}s: {e}")
            
            # Only disconnect on actual errors, not timeouts
            self.disconnect(conn.websocket)
    
    async def _send_batch(self, conn: ConnectionInfo):
        """Send batched messages to a connection"""
        if not conn.message_batch.messages:
            return
        
        # Create batch message
        batch_message = {
            "type": "message_batch",
            "messages": conn.message_batch.messages,
            "batch_size": len(conn.message_batch.messages),
            "timestamp": time.time()
        }
        
        message_str = json.dumps(batch_message)
        await self._safe_send(conn, message_str)
        
        # Reset batch
        conn.message_batch = MessageBatch()
        conn.last_batch_send = int(time.time())

    def _purge_stale_connections(self):
        """Remove connections that have not sent a heartbeat in too long."""
        now = time.time()
        for conn in list(self.active_connections):
            if now - conn.last_heartbeat > self.max_heartbeat_lag:
                console.print(f"[yellow]Dropping stale connection – no heartbeat for {int(now - conn.last_heartbeat)}s")
                self.disconnect(conn.websocket)

    async def broadcast(self, message: Dict[str, Any], priority: str = "normal"):
        """Send a message to all active WebSocket connections with performance optimizations"""
        if not self.active_connections:
            return

        # Rate limiting
        current_time = time.time()
        if current_time - self.broadcast_rate_limiter['last_broadcast'] < self.broadcast_rate_limiter['min_interval']:
            # Queue message for later processing
            self.broadcast_rate_limiter['message_queue'].append((message, priority))
            if not self.broadcast_rate_limiter['queue_process_task']:
                self.broadcast_rate_limiter['queue_process_task'] = asyncio.create_task(self._process_queued_messages())
            return

        self.broadcast_rate_limiter['last_broadcast'] = int(current_time)
        self.performance_stats['total_broadcasts'] += 1

        # Purge dead connections before we start
        self._purge_stale_connections()

        broadcast_start = time.perf_counter()
        
        # Determine if we should use batching based on message type and priority
        use_batching = (priority == "normal" and 
                       message.get('type') in ['training_update', 'checkpoint_playback'] and
                       len(self.active_connections) > 2)
        
        if use_batching:
            # Add to batches for processing
            for conn in list(self.active_connections):
                if conn.health.should_send():
                    should_send_now = conn.message_batch.add_message(message)
                    if should_send_now:
                        await self._send_batch(conn)
        else:
            # Send immediately (high priority or small number of connections)
            message_str = json.dumps(message)
            
            # Sort connections by health score (send to healthy connections first)
            sorted_connections = sorted(
                self.active_connections, 
                key=lambda c: c.health.get_health_score(), 
                reverse=True
            )
            
            # Use semaphore to limit concurrent sends
            semaphore = asyncio.Semaphore(self.max_concurrent_sends)
            
            async def send_with_semaphore(conn):
                async with semaphore:
                    await self._safe_send(conn, message_str)
            
            # Send to all connections concurrently (with limit)
            await asyncio.gather(
                *[send_with_semaphore(conn) for conn in sorted_connections],
                return_exceptions=True
            )
        
        total_duration = time.perf_counter() - broadcast_start
        
        # Update performance stats
        if total_duration > 1.0:
            self.performance_stats['slow_broadcasts'] += 1
            console.print(
                f"[yellow]Broadcast of type {message.get('type', 'unknown')} to {len(self.active_connections)} clients took {total_duration:.3f}s"
            )
        
        # Update average broadcast time
        if self.performance_stats['avg_broadcast_time'] == 0:
            self.performance_stats['avg_broadcast_time'] = total_duration
        else:
            self.performance_stats['avg_broadcast_time'] = float(
                0.9 * self.performance_stats['avg_broadcast_time'] + 
                0.1 * total_duration
            )
        
        self.performance_stats['successful_broadcasts'] += 1
    
    async def _process_queued_messages(self):
        """Process queued messages from rate limiter"""
        try:
            while self.broadcast_rate_limiter['message_queue']:
                message, priority = self.broadcast_rate_limiter['message_queue'].popleft()
                await self.broadcast(message, priority)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(self.broadcast_rate_limiter['min_interval'])
        finally:
            self.broadcast_rate_limiter['queue_process_task'] = None
    
    def get_connection_count(self) -> int:
        """Get the current number of active connections"""
        return len(self.active_connections)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get detailed connection statistics"""
        mobile_safari_count = sum(1 for conn in self.active_connections if conn.is_mobile_safari)
        healthy_connections = sum(1 for conn in self.active_connections if conn.health.get_health_score() > 0.7)
        
        return {
            "total_connections": len(self.active_connections),
            "mobile_connections": self.mobile_connections,
            "desktop_connections": self.desktop_connections,
            "mobile_safari_connections": mobile_safari_count,
            "healthy_connections": healthy_connections,
            "connection_pools": {
                "mobile": len(self.connection_pool["mobile"]),
                "desktop": len(self.connection_pool["desktop"])
            },
            "performance_stats": self.performance_stats,
            "circuit_breaker_stats": {
                "open_circuits": sum(1 for conn in self.active_connections if conn.health.is_circuit_open),
                "total_activations": self.performance_stats['circuit_breaker_activations']
            }
        }
    
    def get_connection_info(self, websocket: WebSocket) -> Optional[ConnectionInfo]:
        """Get connection info for a specific WebSocket"""
        for conn in self.active_connections:
            if conn.websocket == websocket:
                return conn
        return None
    
    def _start_adaptive_heartbeat(self, conn_info: ConnectionInfo):
        """Start adaptive heartbeat for a connection"""
        async def heartbeat_loop():
            try:
                while True:
                    interval = conn_info.get_heartbeat_interval()
                    await asyncio.sleep(interval)
                    
                    # Check if connection is still active
                    if conn_info not in self.active_connections:
                        break
                    
                    try:
                        await conn_info.websocket.send_text(json.dumps({
                            "type": "heartbeat",
                            "timestamp": time.time(),
                            "interval": interval,
                            "mobile_optimized": conn_info.is_mobile,
                            "health_score": conn_info.health.get_health_score(),
                            "circuit_open": conn_info.health.is_circuit_open
                        }))
                        conn_info.last_heartbeat = time.time()
                    except Exception as e:
                        console.print(f"[red]Heartbeat failed for connection: {e}")
                        self.disconnect(conn_info.websocket)
                        break
            except asyncio.CancelledError:
                pass  # Task was cancelled (normal shutdown)
        
        task = asyncio.create_task(heartbeat_loop())
        self.heartbeat_tasks[conn_info.websocket] = task
    
    async def broadcast_high_priority(self, message: Dict[str, Any]):
        """Send high priority message immediately without batching or rate limiting"""
        await self.broadcast(message, priority="high")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        return {
            **self.performance_stats,
            "connection_health": [
                {
                    "id": id(conn.websocket),
                    "health_score": conn.health.get_health_score(),
                    "consecutive_failures": conn.health.consecutive_failures,
                    "total_failures": conn.health.total_failures,
                    "avg_send_time": conn.health.avg_send_time,
                    "is_circuit_open": conn.health.is_circuit_open,
                    "slow_send_count": conn.health.slow_send_count
                }
                for conn in self.active_connections
            ]
        }
    
    def __del__(self):
        """Cleanup when manager is destroyed"""
        if hasattr(self, 'batch_processor_task') and self.batch_processor_task is not None:
            self.batch_processor_task.cancel() 