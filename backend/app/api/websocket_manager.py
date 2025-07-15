"""
WebSocket manager for handling real-time connections with mobile optimizations
"""
import json
import asyncio
import re
import time
from typing import List, Dict, Any, Optional
from fastapi import WebSocket
from rich.console import Console

console = Console()

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
        """Get adaptive timeout based on device type"""
        if self.is_mobile_safari:
            return 5.0  # Longer timeout for Mobile Safari
        elif self.is_mobile:
            return 3.0  # Medium timeout for mobile devices
        else:
            return 1.0  # Short timeout for desktop
    
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
        return self.is_mobile  # Enable pooling for mobile devices

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
        # Write safety – drop a connection if a single send takes longer than X seconds
        self.write_timeout = 2.0  # seconds
        # Consider a connection dead if we haven't seen a heartbeat in this many seconds
        self.max_heartbeat_lag = 30.0  # seconds

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
        """Send with timeout; disconnect on failure."""
        try:
            await asyncio.wait_for(conn.websocket.send_text(message_str), timeout=self.write_timeout)
        except Exception as e:
            console.print(f"[red]Error sending to connection: {e}")
            # If send fails or times-out we drop the connection so it can't block the whole broadcast
            self.disconnect(conn.websocket)

    def _purge_stale_connections(self):
        """Remove connections that have not sent a heartbeat in too long."""
        now = time.time()
        for conn in list(self.active_connections):
            if now - conn.last_heartbeat > self.max_heartbeat_lag:
                console.print(f"[yellow]Dropping stale connection – no heartbeat for {int(now - conn.last_heartbeat)}s")
                self.disconnect(conn.websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Send a message to all active WebSocket connections with mobile priority"""
        if not self.active_connections:
            return

        # Purge dead connections before we start
        self._purge_stale_connections()

        message_str = json.dumps(message)

        # Prioritise mobile, but send sequentially so one bad client can't freeze others
        for conn in list(self.active_connections):
            await self._safe_send(conn, message_str)
    
    def get_connection_count(self) -> int:
        """Get the current number of active connections"""
        return len(self.active_connections)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get detailed connection statistics"""
        mobile_safari_count = sum(1 for conn in self.active_connections if conn.is_mobile_safari)
        return {
            "total_connections": len(self.active_connections),
            "mobile_connections": self.mobile_connections,
            "desktop_connections": self.desktop_connections,
            "mobile_safari_connections": mobile_safari_count,
            "connection_pools": {
                "mobile": len(self.connection_pool["mobile"]),
                "desktop": len(self.connection_pool["desktop"])
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
                            "mobile_optimized": conn_info.is_mobile
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