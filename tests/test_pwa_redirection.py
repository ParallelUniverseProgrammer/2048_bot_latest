#!/usr/bin/env python3
"""
Test PWA Functionality
Tests that the simplified PWA system works correctly without install page
"""

import os
import sys
from pathlib import Path

def test_no_install_page():
    """Test that the PWA install page has been removed"""
    print("🧪 Testing PWA Install Page Removal...")
    
    # Check that the install page no longer exists
    install_page_path = Path("frontend/public/pwa-install.html")
    if install_page_path.exists():
        print("❌ PWA install page still exists - should be removed")
        return False
    
    print("✅ PWA install page has been removed")
    return True

def test_direct_app_access():
    """Test that the app can be accessed directly"""
    print("\n🧪 Testing Direct App Access...")
    
    # Check that the main app files exist
    app_files = [
        "frontend/src/App.tsx",
        "frontend/index.html",
        "frontend/vite.config.ts"
    ]
    
    for file_path in app_files:
        if not Path(file_path).exists():
            print(f"❌ App file not found: {file_path}")
            return False
    
    print("✅ Main app files are accessible")
    return True

def test_ios_tooltip_implementation():
    """Test that iOS tooltip is properly implemented"""
    print("\n🧪 Testing iOS Tooltip Implementation...")
    
    app_path = Path("frontend/src/App.tsx")
    if not app_path.exists():
        print("❌ App.tsx not found")
        return False
    
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for iOS tooltip implementation
    tooltip_elements = [
        'showIOSTooltip',
        'iPad|iPhone|iPod',
        'Safari',
        'Add to Home Screen',
        'share button',
        'setShowIOSTooltip'
    ]
    
    for element in tooltip_elements:
        if element not in content:
            print(f"❌ Missing tooltip element: {element}")
            return False
    
    print("✅ iOS tooltip properly implemented")
    return True

def test_no_old_install_code():
    """Test that old install code has been removed"""
    print("\n🧪 Testing Old Install Code Removal...")
    
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
        'isPWAInstalled',
        'install=pwa',
        'source=qr'
    ]
    
    for element in removed_elements:
        if element in content:
            print(f"❌ Old install element still present: {element}")
            return False
    
    print("✅ Old install code has been removed")
    return True

def test_qr_code_direct_access():
    """Test that QR code points directly to app"""
    print("\n🧪 Testing QR Code Direct Access...")
    
    # Import the launcher module
    sys.path.insert(0, '.')
    from launcher import QRCodeGenerator
    
    # Test that QR code generation works with direct URL
    test_url = "http://192.168.1.100:5173"
    
    try:
        # This should not throw an error
        QRCodeGenerator.generate_qr_code(test_url, "test_qr_direct.png")
        
        if os.path.exists("test_qr_direct.png"):
            os.remove("test_qr_direct.png")  # Clean up
            print("✅ QR code generation works with direct URL")
            return True
        else:
            print("❌ QR code file not created")
            return False
    except Exception as e:
        print(f"❌ QR code generation failed: {e}")
        return False

def main():
    """Run all PWA functionality tests"""
    print("🚀 Testing PWA Functionality\n")
    
    tests = [
        test_no_install_page,
        test_direct_app_access,
        test_ios_tooltip_implementation,
        test_no_old_install_code,
        test_qr_code_direct_access
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