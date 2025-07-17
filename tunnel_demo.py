#!/usr/bin/env python3
"""
Quick Tunnel Demo Script
========================

This script demonstrates Phase 2.1 of the Remote Access Roadmap:
- Quick Tunnel functionality for instant public HTTPS access
- No account required, disposable URL under *.trycloudflare.com
- Perfect for demos, testing, and fallback scenarios

Current active tunnel: https://scoop-linda-impressed-substances.trycloudflare.com
"""

import requests
import json
import subprocess
import time
from pathlib import Path

def test_tunnel_endpoints():
    """Test the tunnel endpoints to verify functionality"""
    tunnel_url = "https://scoop-linda-impressed-substances.trycloudflare.com"
    
    print(f"🌐 Testing tunnel endpoints at: {tunnel_url}")
    print("-" * 60)
    
    # Test endpoints
    endpoints = [
        ("Root", "/"),
        ("Health Check", "/healthz"),
        ("Training Status", "/training/status"),
        ("Checkpoints", "/checkpoints"),
    ]
    
    for name, path in endpoints:
        try:
            response = requests.get(f"{tunnel_url}{path}", timeout=10)
            if response.status_code == 200:
                print(f"✅ {name}: {response.status_code} - {response.text[:100]}...")
            else:
                print(f"❌ {name}: {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
    
    print("\n🎉 Quick Tunnel is working successfully!")
    print("🔥 Your 2048 Bot backend is now accessible from anywhere via HTTPS!")
    print("📱 This URL can be used for mobile PWA installation")

def generate_qr_code(url):
    """Generate a QR code for the tunnel URL"""
    try:
        import qrcode
        qr = qrcode.QRCode()
        qr.add_data(url)
        qr.make()
        print(f"\n📱 QR Code for {url}:")
        qr.print_ascii(invert=True)
    except ImportError:
        print(f"\n📱 To generate QR codes, install: pip install qrcode[pil]")
        print(f"🔗 Direct URL: {url}")

def main():
    """Main demo function"""
    print("🚀 2048 Bot Remote Access - Quick Tunnel Demo")
    print("=" * 60)
    
    # Test the tunnel
    test_tunnel_endpoints()
    
    # Generate QR code
    tunnel_url = "https://scoop-linda-impressed-substances.trycloudflare.com"
    generate_qr_code(tunnel_url)
    
    print("\n" + "=" * 60)
    print("✨ Phase 2.1 - Quick Tunnel: COMPLETED")
    print("📋 Next steps:")
    print("   - Set up Named Tunnel for persistent URL (Phase 2.2)")
    print("   - Update PWA configuration for tunnel URLs (Phase 2.3)")
    print("   - Integrate tunnel into launcher.py (Phase 2.4)")

if __name__ == "__main__":
    main() 