#!/usr/bin/env python3
"""
Test PWA Functionality
Tests the QR code generation and iOS tooltip functionality
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def test_qr_code_generation():
    """Test QR code generation with direct app URL"""
    print("🧪 Testing QR Code Generation...")
    
    # Import the launcher module
    sys.path.insert(0, '.')
    from launcher import QRCodeGenerator
    
    # Test URL generation
    test_url = "http://192.168.1.100:5173"
    
    # Create a temporary QR code to test
    try:
        QRCodeGenerator.generate_qr_code(test_url, "test_qr.png")
        
        # Check if the QR code file was created
        if os.path.exists("test_qr.png"):
            print("✅ QR code generated successfully")
            os.remove("test_qr.png")  # Clean up
            return True
        else:
            print("❌ QR code file not created")
            return False
    except Exception as e:
        print(f"❌ QR code generation failed: {e}")
        return False

def test_pwa_manifest():
    """Test that the PWA manifest is properly configured"""
    print("\n🧪 Testing PWA Manifest...")
    
    # Check if the main vite config has PWA configuration
    vite_config_path = Path("frontend/vite.config.ts")
    if not vite_config_path.exists():
        print("❌ Vite config not found")
        return False
    
    with open(vite_config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for PWA plugin
    if 'VitePWA' not in content:
        print("❌ VitePWA plugin not found in config")
        return False
    
    # Check for PWA icons
    pwa_icons = [
        "frontend/public/pwa-192x192.png",
        "frontend/public/pwa-512x512.png",
        "frontend/public/apple-touch-icon.png"
    ]
    
    for icon in pwa_icons:
        if not Path(icon).exists():
            print(f"❌ PWA icon not found: {icon}")
            return False
    
    print("✅ PWA manifest and icons configured correctly")
    return True

def test_ios_tooltip_functionality():
    """Test that the React app has iOS tooltip functionality"""
    print("\n🧪 Testing iOS Tooltip Functionality...")
    
    app_path = Path("frontend/src/App.tsx")
    if not app_path.exists():
        print("❌ App.tsx not found")
        return False
    
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for iOS tooltip-related code
    tooltip_elements = [
        'showIOSTooltip',
        'iPad|iPhone|iPod',
        'Safari',
        'Add to Home Screen',
        'share button'
    ]
    
    for element in tooltip_elements:
        if element not in content:
            print(f"❌ Missing tooltip element: {element}")
            return False
    
    print("✅ iOS tooltip functionality implemented")
    return True

def test_no_install_page():
    """Test that the PWA install page has been removed"""
    print("\n🧪 Testing PWA Install Page Removal...")
    
    # Check that the install page no longer exists
    install_page_path = Path("frontend/public/pwa-install.html")
    if install_page_path.exists():
        print("❌ PWA install page still exists - should be removed")
        return False
    
    print("✅ PWA install page has been removed")
    return True

def test_no_install_functionality():
    """Test that the old install functionality has been removed"""
    print("\n🧪 Testing Install Functionality Removal...")
    
    app_path = Path("frontend/src/App.tsx")
    if not app_path.exists():
        print("❌ App.tsx not found")
        return False
    
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check that old install-related code has been removed
    removed_elements = [
        'showInstallPrompt',
        'deferredPrompt',
        'beforeinstallprompt',
        'appinstalled',
        'handleInstallApp',
        'isPWAInstalled'
    ]
    
    for element in removed_elements:
        if element in content:
            print(f"❌ Old install element still present: {element}")
            return False
    
    print("✅ Old install functionality has been removed")
    return True

def main():
    """Run all PWA functionality tests"""
    print("🚀 Testing PWA Functionality\n")
    
    tests = [
        test_qr_code_generation,
        test_pwa_manifest,
        test_ios_tooltip_functionality,
        test_no_install_page,
        test_no_install_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All PWA functionality tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 