#!/usr/bin/env python3
"""
Tunnel Manager Utility
======================

This utility provides tunnel management functionality for testing:
- Start cloudflared tunnel and capture the URL
- Test tunnel endpoints
- Manage tunnel lifecycle
- Provide tunnel status information

This utility is part of the test utilities and supports integration testing.
"""

import subprocess
import re
import sys
import requests
import time
from typing import Optional, Tuple, Dict, Any
from tests.utilities.test_utils import TestLogger
from tests.utilities.backend_manager import requires_mock_backend

class TunnelManager:
    """Utility class for managing cloudflared tunnels"""
    
    def __init__(self, cloudflared_path: str = "./cloudflared.exe", 
                 local_url: str = "http://localhost:8000"):
        self.logger = TestLogger()
        self.cloudflared_path = cloudflared_path
        self.local_url = local_url
        self.process: Optional[subprocess.Popen] = None
        self.tunnel_url: Optional[str] = None
    
    def start_tunnel_and_get_url(self) -> Tuple[Optional[str], Optional[subprocess.Popen]]:
        """Start tunnel and capture the URL"""
        self.logger.info("Starting cloudflared tunnel...")
        
        # Start cloudflared process
        try:
            self.process = subprocess.Popen(
                [self.cloudflared_path, "tunnel", "--url", self.local_url],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
        except Exception as e:
            self.logger.error(f"Failed to start tunnel: {e}")
            return None, None
        
        self.tunnel_url = None
        
        try:
            if self.process.stdout:
                for line in self.process.stdout:
                    self.logger.debug(line.strip())
                    
                    # Look for the tunnel URL in the output
                    if "Visit it at" in line or "https://" in line:
                        # Extract URL using regex
                        url_match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                        if url_match:
                            self.tunnel_url = url_match.group(0)
                            self.logger.ok(f"Tunnel URL captured: {self.tunnel_url}")
                            break
        
        except KeyboardInterrupt:
            self.logger.warning("Tunnel interrupted by user")
            self.terminate_tunnel()
            return None, None
        
        return self.tunnel_url, self.process
    
    @requires_mock_backend("Tunnel URL Test")
    def test_tunnel_url(self, url: str) -> bool:
        """Test if the tunnel URL is working"""
        self.logger.info("Testing tunnel URL...")
        
        try:
            response = requests.get(f"{url}/healthz", timeout=10)
            if response.status_code == 200:
                self.logger.ok(f"Health check: {response.status_code}")
                return True
            else:
                self.logger.error(f"Health check failed: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return False
    
    def wait_for_tunnel_ready(self, max_wait_time: int = 60, check_interval: int = 5) -> bool:
        """Wait for tunnel to be ready"""
        self.logger.info(f"Waiting for tunnel to be ready (max {max_wait_time}s)...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            if self.tunnel_url and self.test_tunnel_url(self.tunnel_url):
                self.logger.ok("Tunnel is ready!")
                return True
            
            time.sleep(check_interval)
            self.logger.info(f"Still waiting... ({int(time.time() - start_time)}s elapsed)")
        
        self.logger.error("Tunnel did not become ready within timeout")
        return False
    
    def terminate_tunnel(self) -> None:
        """Terminate the tunnel process"""
        if self.process:
            self.logger.info("Stopping tunnel...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                self.logger.ok("Tunnel stopped successfully")
            except subprocess.TimeoutExpired:
                self.logger.warning("Tunnel did not stop gracefully, forcing termination")
                self.process.kill()
            self.process = None
            self.tunnel_url = None
    
    def get_tunnel_status(self) -> Dict[str, Any]:
        """Get current tunnel status"""
        return {
            "running": self.process is not None and self.process.poll() is None,
            "url": self.tunnel_url,
            "local_url": self.local_url
        }
    
    def run_tunnel_with_testing(self) -> bool:
        """Run tunnel with automatic testing"""
        self.logger.banner("Tunnel Manager - Start and Test", 60)
        
        # Start tunnel
        url, proc = self.start_tunnel_and_get_url()
        if not url or not proc:
            return False
        
        self.logger.ok(f"Tunnel active at: {url}")
        
        # Wait for tunnel to be ready
        if not self.wait_for_tunnel_ready():
            self.terminate_tunnel()
            return False
        
        # Test the tunnel
        self.logger.info("Testing tunnel endpoints...")
        if not self.test_tunnel_url(url):
            self.terminate_tunnel()
            return False
        
        self.logger.success("Tunnel is running and tested successfully!")
        self.logger.info("Press Ctrl+C to stop")
        
        try:
            proc.wait()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self.terminate_tunnel()
        
        return True

@requires_mock_backend("Tunnel Manager Demo")
def main():
    """Main entry point for tunnel manager"""
    logger = TestLogger()
    logger.banner("Tunnel Manager", 60)
    
    manager = TunnelManager()
    success = manager.run_tunnel_with_testing()
    
    if success:
        logger.success("Tunnel manager completed successfully!")
        return 0
    else:
        logger.error("Tunnel manager failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 