#!/usr/bin/env python3
"""
Tunnel Demo Integration Test
============================

This test verifies the Quick Tunnel functionality for remote access:
- Tests tunnel endpoints to verify functionality
- Validates HTTPS access from external locations
- Checks PWA compatibility via tunnel
- Verifies API endpoint accessibility

This test is part of the integration test suite and requires a running backend.
"""

from tests.utilities.test_utils import TestLogger, BackendTester
import requests
import json
import time
from typing import Dict, List, Tuple, Optional

class TunnelDemoTester:
    """Test class for tunnel demo functionality"""
    
    def __init__(self, tunnel_url: str = "https://scoop-linda-impressed-substances.trycloudflare.com"):
        self.logger = TestLogger()
        self.tunnel_url = tunnel_url
        self.backend = BackendTester("http://localhost:8000", self.logger)
    
    def test_tunnel_endpoints(self) -> List[Tuple[str, bool, str]]:
        """Test the tunnel endpoints to verify functionality"""
        self.logger.section("Testing Tunnel Endpoints")
        
        endpoints = [
            ("Root", "/"),
            ("Health Check", "/healthz"),
            ("Training Status", "/training/status"),
            ("Checkpoints", "/checkpoints"),
        ]
        
        results = []
        
        for name, path in endpoints:
            try:
                response = requests.get(f"{self.tunnel_url}{path}", timeout=10)
                if response.status_code == 200:
                    self.logger.ok(f"{name}: {response.status_code} - {response.text[:100]}...")
                    results.append((name, True, f"{response.status_code} - OK"))
                else:
                    self.logger.error(f"{name}: {response.status_code}")
                    results.append((name, False, f"{response.status_code} - ERROR"))
            except Exception as e:
                self.logger.error(f"{name}: Error - {e}")
                results.append((name, False, str(e)))
        
        return results
    
    def test_pwa_compatibility(self) -> bool:
        """Test PWA compatibility via tunnel"""
        self.logger.section("Testing PWA Compatibility")
        
        try:
            # Test manifest via tunnel
            manifest_response = requests.get(f"{self.tunnel_url}/manifest.webmanifest", timeout=10)
            if manifest_response.status_code == 200:
                manifest = manifest_response.json()
                self.logger.ok("PWA Manifest accessible via tunnel")
                self.logger.info(f"Name: {manifest.get('name', 'Unknown')}")
                self.logger.info(f"Start URL: {manifest.get('start_url', 'Unknown')}")
            else:
                self.logger.error(f"PWA Manifest not accessible via tunnel ({manifest_response.status_code})")
                return False
            
            # Test service worker via tunnel
            sw_response = requests.get(f"{self.tunnel_url}/sw.js", timeout=10)
            if sw_response.status_code == 200:
                self.logger.ok("Service Worker accessible via tunnel")
                self.logger.info(f"Size: {len(sw_response.content)} bytes")
            else:
                self.logger.error(f"Service Worker not accessible via tunnel ({sw_response.status_code})")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"PWA Compatibility Error: {e}")
            return False
    
    def test_api_accessibility(self) -> bool:
        """Test API accessibility via tunnel"""
        self.logger.section("Testing API Accessibility")
        
        try:
            # Test key API endpoints
            api_endpoints = [
                ("Health Check", "/healthz"),
                ("Training Status", "/training/status"),
                ("Checkpoints", "/checkpoints"),
            ]
            
            for name, path in api_endpoints:
                response = requests.get(f"{self.tunnel_url}{path}", timeout=10)
                if response.status_code == 200:
                    self.logger.ok(f"{name} accessible via tunnel")
                else:
                    self.logger.error(f"{name} not accessible via tunnel ({response.status_code})")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"API Accessibility Error: {e}")
            return False
    
    def generate_qr_code_info(self) -> None:
        """Provide QR code generation information"""
        self.logger.section("QR Code Information")
        
        self.logger.info("To generate QR codes for mobile access:")
        self.logger.info("1. Install qrcode library: pip install qrcode[pil]")
        self.logger.info("2. Use the tunnel URL for mobile PWA installation")
        self.logger.info(f"3. Direct URL: {self.tunnel_url}")
        
        # Try to generate QR code if qrcode is available
        try:
            import qrcode
            qr = qrcode.QRCode()
            qr.add_data(self.tunnel_url)
            qr.make()
            self.logger.info(f"📱 QR Code for {self.tunnel_url}:")
            qr.print_ascii(invert=True)
        except ImportError:
            self.logger.info("📱 QR code library not available - install with: pip install qrcode[pil]")
    
    def run_full_test_suite(self) -> bool:
        """Run the complete tunnel demo test suite"""
        self.logger.banner("Tunnel Demo Integration Test Suite", 60)
        
        # Test backend connectivity first
        if not self.backend.test_connectivity():
            self.logger.error("Backend not available, skipping tunnel tests")
            return False
        
        # Step 1: Test tunnel endpoints
        self.logger.step(1, 4, "Testing tunnel endpoints")
        endpoint_results = self.test_tunnel_endpoints()
        
        # Check if all endpoints are working
        all_endpoints_working = all(success for _, success, _ in endpoint_results)
        if not all_endpoints_working:
            self.logger.error("Some tunnel endpoints are not working")
            return False
        
        # Step 2: Test PWA compatibility
        self.logger.step(2, 4, "Testing PWA compatibility")
        if not self.test_pwa_compatibility():
            return False
        
        # Step 3: Test API accessibility
        self.logger.step(3, 4, "Testing API accessibility")
        if not self.test_api_accessibility():
            return False
        
        # Step 4: Generate QR code info
        self.logger.step(4, 4, "Providing mobile access information")
        self.generate_qr_code_info()
        
        self.logger.success("Tunnel demo test suite completed successfully!")
        return True

def main():
    """Main entry point for tunnel demo integration test"""
    logger = TestLogger()
    logger.banner("Tunnel Demo Integration Test", 60)
    
    logger.info("This test demonstrates Phase 2.1 of the Remote Access Roadmap:")
    logger.info("• Quick Tunnel functionality for instant public HTTPS access")
    logger.info("• No account required, disposable URL under *.trycloudflare.com")
    logger.info("• Perfect for demos, testing, and fallback scenarios")
    
    tester = TunnelDemoTester()
    success = tester.run_full_test_suite()
    
    if success:
        logger.success("🎉 Quick Tunnel is working successfully!")
        logger.info("🔥 Your 2048 Bot backend is now accessible from anywhere via HTTPS!")
        logger.info("📱 This URL can be used for mobile PWA installation")
        logger.info("")
        logger.info("✨ Phase 2.1 - Quick Tunnel: COMPLETED")
        logger.info("📋 Next steps:")
        logger.info("   - Set up Named Tunnel for persistent URL (Phase 2.2)")
        logger.info("   - Update PWA configuration for tunnel URLs (Phase 2.3)")
        logger.info("   - Integrate tunnel into launcher.py (Phase 2.4)")
        return 0
    else:
        logger.error("❌ Tunnel demo test failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 