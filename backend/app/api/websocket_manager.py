"""
WebSocket manager for handling real-time connections
"""
import json
import asyncio
from typing import List, Dict, Any
from fastapi import WebSocket
from rich.console import Console

console = Console()

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_count = 0
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_count += 1
        console.print(f"[green]WebSocket connected. Total connections: {len(self.active_connections)}")
        
        # Send initial connection message
        await self.send_personal_message({
            "type": "connection_established",
            "message": "Connected to training server",
            "timestamp": asyncio.get_event_loop().time()
        }, websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            console.print(f"[yellow]WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            console.print(f"[red]Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Send a message to all active WebSocket connections"""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message)
        # Send to all connections concurrently to avoid head-of-line blocking
        coros = [conn.send_text(message_str) for conn in self.active_connections]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for conn, res in zip(self.active_connections.copy(), results):
            if isinstance(res, Exception):
                console.print(f"[red]Error broadcasting to connection: {res}")
                self.disconnect(conn)
    
    def get_connection_count(self) -> int:
        """Get the current number of active connections"""
        return len(self.active_connections) 