#!/usr/bin/env python3
"""
Simple script to start cloudflared tunnel and capture the URL
"""
import subprocess
import re
import sys
from typing import Optional, Tuple

def start_tunnel_and_get_url() -> Tuple[Optional[str], Optional[subprocess.Popen]]:
    """Start tunnel and capture the URL"""
    print("🌐 Starting cloudflared tunnel...")
    
    # Start cloudflared process
    try:
        process = subprocess.Popen(
            ["./cloudflared.exe", "tunnel", "--url", "http://localhost:8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
    except Exception as e:
        print(f"❌ Failed to start tunnel: {e}")
        return None, None
    
    tunnel_url = None
    
    try:
        if process.stdout:
            for line in process.stdout:
                print(line.strip())
                
                # Look for the tunnel URL in the output
                if "Visit it at" in line or "https://" in line:
                    # Extract URL using regex
                    url_match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                    if url_match:
                        tunnel_url = url_match.group(0)
                        print(f"\n✅ Tunnel URL captured: {tunnel_url}")
                        break
    
    except KeyboardInterrupt:
        print("\n⚠️  Tunnel interrupted by user")
        process.terminate()
        return None, None
    
    return tunnel_url, process

if __name__ == "__main__":
    result = start_tunnel_and_get_url()
    if result and result[0] and result[1]:
        url, proc = result
        print(f"\n🎉 Tunnel active at: {url}")
        print("📋 Testing endpoints...")
        
        # Test the tunnel URL
        import requests
        try:
            response = requests.get(f"{url}/healthz", timeout=10)
            if response.status_code == 200:
                print(f"✅ Health check: {response.status_code}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Health check error: {e}")
        
        print("\n🔄 Tunnel is running... Press Ctrl+C to stop")
        try:
            proc.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping tunnel...")
            proc.terminate()
    else:
        print("❌ Failed to capture tunnel URL") 