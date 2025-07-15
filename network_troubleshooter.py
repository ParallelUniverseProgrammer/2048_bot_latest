#!/usr/bin/env python3
"""
Network Troubleshooter for 2048 Bot - Spectrum Router Edition
Helps diagnose and fix mobile access issues with newer Spectrum routers.
"""

import os
import sys
import socket
import subprocess
import requests
import time
import json
import platform
from typing import List, Dict, Optional, Tuple
import netifaces

class Colors:
    """Terminal color constants"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class SpectrumRouterTroubleshooter:
    """Troubleshooter specifically for Spectrum router issues"""
    
    def __init__(self):
        self.host_ip = None
        self.backend_port = 8000
        self.frontend_port = 5173
        self.issues_found = []
        self.solutions = []
        
    def run_full_diagnostic(self):
        """Run complete diagnostic for Spectrum router issues"""
        print(f"{Colors.HEADER}🔍 Spectrum Router Network Diagnostic{Colors.ENDC}")
        print("=" * 60)
        
        # Step 1: Network Discovery
        self._discover_network()
        
        # Step 2: Check Windows Firewall
        self._check_windows_firewall()
        
        # Step 3: Check Spectrum Router Settings
        self._check_spectrum_router_issues()
        
        # Step 4: Test Local Connectivity
        self._test_local_connectivity()
        
        # Step 5: Generate Solutions
        self._generate_solutions()
        
        # Step 6: Provide Quick Fix Commands
        self._provide_quick_fixes()
        
    def _discover_network(self):
        """Discover current network configuration"""
        print(f"\n{Colors.OKCYAN}📡 Network Discovery{Colors.ENDC}")
        print("-" * 30)
        
        # Get all network interfaces
        interfaces = []
        try:
            for interface in netifaces.interfaces():
                try:
                    addresses = netifaces.ifaddresses(interface)
                    if netifaces.AF_INET in addresses:
                        for addr_info in addresses[netifaces.AF_INET]:
                            ip = addr_info.get('addr')
                            if ip and not ip.startswith('127.') and not ip.startswith('169.254.'):
                                interfaces.append((interface, ip))
                except Exception:
                    continue
        except Exception as e:
            print(f"{Colors.FAIL}✗ Failed to get network interfaces: {e}{Colors.ENDC}")
            return
        
        print(f"{Colors.OKGREEN}✓ Found {len(interfaces)} network interfaces:{Colors.ENDC}")
        for interface, ip in interfaces:
            print(f"  • {interface}: {ip}")
        
        # Find the best IP (same logic as launcher)
        self.host_ip = self._find_best_ip(interfaces)
        if self.host_ip:
            print(f"{Colors.OKGREEN}✓ Selected IP: {self.host_ip}{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}✗ Could not determine suitable IP address{Colors.ENDC}")
            self.issues_found.append("No suitable IP address found")
    
    def _find_best_ip(self, interfaces: List[Tuple[str, str]]) -> Optional[str]:
        """Find the best IP address for LAN access"""
        if not interfaces:
            return None
        
        ips = [ip for _, ip in interfaces]
        
        # Prefer 192.168.x.x for home networks
        for interface, ip in interfaces:
            if ip.startswith('192.168.'):
                return ip
        
        # Then prefer 10.x.x.x
        for interface, ip in interfaces:
            if ip.startswith('10.'):
                return ip
        
        # Finally accept 172.x.x.x
        for interface, ip in interfaces:
            if ip.startswith('172.'):
                return ip
        
        # Return first available
        return interfaces[0][1] if interfaces else None
    
    def _check_windows_firewall(self):
        """Check Windows Firewall settings"""
        print(f"\n{Colors.OKCYAN}🛡️ Windows Firewall Check{Colors.ENDC}")
        print("-" * 30)
        
        # Check if Windows Firewall is blocking the ports
        try:
            # Check if ports are in use
            for port in [self.backend_port, self.frontend_port]:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        sock.bind(('', port))
                        print(f"{Colors.OKGREEN}✓ Port {port} is available{Colors.ENDC}")
                except OSError:
                    print(f"{Colors.WARNING}⚠ Port {port} is already in use{Colors.ENDC}")
                    self.issues_found.append(f"Port {port} is already in use")
        except Exception as e:
            print(f"{Colors.FAIL}✗ Error checking ports: {e}{Colors.ENDC}")
        
        # Check Windows Firewall rules
        try:
            result = subprocess.run([
                'powershell', '-Command', 
                'Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*Python*" -or $_.DisplayName -like "*Node*"} | Select-Object DisplayName,Enabled,Profile | ConvertTo-Json'
            ], capture_output=True, text=True, shell=True)
            
            if result.returncode == 0 and result.stdout.strip():
                rules = json.loads(result.stdout)
                if not isinstance(rules, list):
                    rules = [rules]
                
                if rules:
                    print(f"{Colors.OKGREEN}✓ Found {len(rules)} relevant firewall rules:{Colors.ENDC}")
                    for rule in rules:
                        status = "Enabled" if rule.get('Enabled') else "Disabled"
                        print(f"  • {rule.get('DisplayName', 'Unknown')}: {status}")
                else:
                    print(f"{Colors.WARNING}⚠ No relevant firewall rules found{Colors.ENDC}")
                    self.issues_found.append("No firewall rules for Python/Node.js")
        except Exception as e:
            print(f"{Colors.WARNING}⚠ Could not check firewall rules: {e}{Colors.ENDC}")
    
    def _check_spectrum_router_issues(self):
        """Check for common Spectrum router issues"""
        print(f"\n{Colors.OKCYAN}📡 Spectrum Router Issues{Colors.ENDC}")
        print("-" * 30)
        
        # Check for AP Isolation (most common issue)
        print(f"{Colors.WARNING}⚠ Checking for AP Isolation (most common Spectrum issue)...{Colors.ENDC}")
        print("   This prevents devices from communicating with each other on the same network.")
        print("   You'll need to check your router settings manually.")
        
        # Check for client isolation
        print(f"{Colors.WARNING}⚠ Checking for Client Isolation...{Colors.ENDC}")
        print("   This is another common setting that blocks device-to-device communication.")
        
        # Check for firewall settings
        print(f"{Colors.WARNING}⚠ Checking for router firewall settings...{Colors.ENDC}")
        print("   Some Spectrum routers have aggressive firewall settings.")
        
        self.issues_found.extend([
            "AP Isolation may be enabled",
            "Client Isolation may be enabled", 
            "Router firewall may be blocking local traffic"
        ])
    
    def _test_local_connectivity(self):
        """Test local network connectivity"""
        print(f"\n{Colors.OKCYAN}🔗 Local Connectivity Test{Colors.ENDC}")
        print("-" * 30)
        
        if not self.host_ip:
            print(f"{Colors.FAIL}✗ Cannot test connectivity without valid IP{Colors.ENDC}")
            return
        
        # Test if we can reach the router
        router_ip = self._get_router_ip()
        if router_ip:
            print(f"{Colors.OKGREEN}✓ Router IP: {router_ip}{Colors.ENDC}")
            
            # Test router connectivity
            try:
                response = requests.get(f"http://{router_ip}", timeout=5)
                print(f"{Colors.OKGREEN}✓ Router is reachable{Colors.ENDC}")
            except Exception:
                print(f"{Colors.WARNING}⚠ Router is not reachable via HTTP{Colors.ENDC}")
        
        # Test if other devices can reach us
        print(f"{Colors.OKCYAN}Testing if other devices can reach this computer...{Colors.ENDC}")
        print(f"   Your computer should be accessible at: http://{self.host_ip}:{self.frontend_port}")
        print(f"   Backend API: http://{self.host_ip}:{self.backend_port}")
        
        # Check if servers are running
        self._check_server_status()
    
    def _get_router_ip(self) -> Optional[str]:
        """Get router IP address"""
        if not self.host_ip:
            return None
        
        # Common router IP patterns
        parts = self.host_ip.split('.')
        if len(parts) == 4:
            # Most common: 192.168.1.1 or 192.168.0.1
            if parts[0] == '192' and parts[1] == '168':
                if parts[2] == '1':
                    return '192.168.1.1'
                elif parts[2] == '0':
                    return '192.168.0.1'
                else:
                    return f"192.168.{parts[2]}.1"
            # 10.x.x.x networks
            elif parts[0] == '10':
                return f"10.{parts[1]}.{parts[2]}.1"
        
        return None
    
    def _check_server_status(self):
        """Check if the servers are currently running"""
        print(f"\n{Colors.OKCYAN}🖥️ Server Status Check{Colors.ENDC}")
        print("-" * 30)
        
        # Check backend
        try:
            response = requests.get(f"http://{self.host_ip}:{self.backend_port}/docs", timeout=5)
            print(f"{Colors.OKGREEN}✓ Backend server is running{Colors.ENDC}")
        except Exception:
            print(f"{Colors.WARNING}⚠ Backend server is not running{Colors.ENDC}")
            self.issues_found.append("Backend server is not running")
        
        # Check frontend
        try:
            response = requests.get(f"http://{self.host_ip}:{self.frontend_port}", timeout=5)
            print(f"{Colors.OKGREEN}✓ Frontend server is running{Colors.ENDC}")
        except Exception:
            print(f"{Colors.WARNING}⚠ Frontend server is not running{Colors.ENDC}")
            self.issues_found.append("Frontend server is not running")
    
    def _generate_solutions(self):
        """Generate solutions for found issues"""
        print(f"\n{Colors.OKCYAN}🔧 Solutions for Spectrum Router Issues{Colors.ENDC}")
        print("-" * 30)
        
        print(f"{Colors.BOLD}1. Router Settings to Check:{Colors.ENDC}")
        print("   • Access your router at: http://192.168.1.1 (or your router's IP)")
        print("   • Default Spectrum credentials are usually:")
        print("     - Username: admin")
        print("     - Password: password (or the password on your router)")
        print("   • Look for these settings and DISABLE them:")
        print("     - AP Isolation")
        print("     - Client Isolation") 
        print("     - Device Isolation")
        print("     - Network Isolation")
        print("     - Guest Network Isolation")
        
        print(f"\n{Colors.BOLD}2. Windows Firewall Fixes:{Colors.ENDC}")
        print("   • Open Windows Defender Firewall")
        print("   • Click 'Allow an app or feature through Windows Defender Firewall'")
        print("   • Add Python and Node.js to the allowed list")
        print("   • Or temporarily disable firewall for testing")
        
        print(f"\n{Colors.BOLD}3. Alternative Solutions:{Colors.ENDC}")
        print("   • Try using a different port (some routers block certain ports)")
        print("   • Use a mobile hotspot temporarily to test")
        print("   • Check if your phone is on the same WiFi network")
        print("   • Try accessing via IP address instead of hostname")
        
        print(f"\n{Colors.BOLD}4. Spectrum-Specific Issues:{Colors.ENDC}")
        print("   • Some newer Spectrum routers have 'Advanced Security' enabled")
        print("   • This can block local device communication")
        print("   • Check for 'Advanced Security' or 'Security Suite' settings")
        print("   • Disable any 'Network Protection' features temporarily")
    
    def _provide_quick_fixes(self):
        """Provide quick fix commands"""
        print(f"\n{Colors.OKCYAN}⚡ Quick Fix Commands{Colors.ENDC}")
        print("-" * 30)
        
        print(f"{Colors.BOLD}Windows Firewall - Allow Python:{Colors.ENDC}")
        print("netsh advfirewall firewall add rule name=\"Python Server\" dir=in action=allow protocol=TCP localport=8000,5173")
        
        print(f"\n{Colors.BOLD}Windows Firewall - Allow Node.js:{Colors.ENDC}")
        print("netsh advfirewall firewall add rule name=\"Node.js Server\" dir=in action=allow protocol=TCP localport=8000,5173")
        
        print(f"\n{Colors.BOLD}Test if ports are accessible:{Colors.ENDC}")
        print(f"curl http://{self.host_ip}:{self.backend_port}/docs")
        print(f"curl http://{self.host_ip}:{self.frontend_port}")
        
        print(f"\n{Colors.BOLD}Check what's using the ports:{Colors.ENDC}")
        print("netstat -ano | findstr :8000")
        print("netstat -ano | findstr :5173")
        
        print(f"\n{Colors.BOLD}Kill processes using the ports:{Colors.ENDC}")
        print("taskkill /PID <PID> /F")
        
        print(f"\n{Colors.BOLD}Test mobile access:{Colors.ENDC}")
        print(f"   Open your phone's browser and go to: http://{self.host_ip}:{self.frontend_port}")
        print(f"   If that doesn't work, try: http://{self.host_ip}:8000/docs")
        
        print(f"\n{Colors.BOLD}Router Access:{Colors.ENDC}")
        router_ip = self._get_router_ip()
        if router_ip:
            print(f"   Access your router at: http://{router_ip}")
        else:
            print("   Try: http://192.168.1.1 or http://192.168.0.1")

def main():
    """Main function"""
    print(f"{Colors.HEADER}🔍 2048 Bot Network Troubleshooter - Spectrum Edition{Colors.ENDC}")
    print("=" * 70)
    print("This tool will help diagnose why your phone can't access the app.")
    print("It's specifically designed for newer Spectrum router issues.\n")
    
    troubleshooter = SpectrumRouterTroubleshooter()
    troubleshooter.run_full_diagnostic()
    
    print(f"\n{Colors.HEADER}📋 Summary{Colors.ENDC}")
    print("=" * 30)
    if troubleshooter.issues_found:
        print(f"{Colors.WARNING}Found {len(troubleshooter.issues_found)} potential issues:{Colors.ENDC}")
        for i, issue in enumerate(troubleshooter.issues_found, 1):
            print(f"  {i}. {issue}")
    else:
        print(f"{Colors.OKGREEN}No obvious issues found. Try the router settings above.{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}🎯 Most Likely Solution:{Colors.ENDC}")
    print("1. Access your Spectrum router settings")
    print("2. Disable 'AP Isolation' or 'Client Isolation'")
    print("3. Disable any 'Advanced Security' features")
    print("4. Restart your router")
    print("5. Try accessing the app again")
    
    print(f"\n{Colors.OKCYAN}Need more help? Check the router manual or contact Spectrum support.{Colors.ENDC}")

if __name__ == "__main__":
    main() 