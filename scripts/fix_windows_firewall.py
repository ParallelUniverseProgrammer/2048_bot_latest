#!/usr/bin/env python3
"""
Quick Windows Firewall Fix for 2048 Bot
Automatically adds firewall rules to allow mobile access.
"""

import subprocess
import sys
import os

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

def run_command(command, description):
    """Run a command and show the result"""
    print(f"{Colors.OKCYAN}{description}...{Colors.ENDC}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{Colors.OKGREEN}✓ {description} completed{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.WARNING}⚠ {description} failed: {result.stderr}{Colors.ENDC}")
            return False
    except Exception as e:
        print(f"{Colors.FAIL}✗ {description} error: {e}{Colors.ENDC}")
        return False

def main():
    """Main function"""
    print(f"{Colors.HEADER}🛡️ Windows Firewall Fix for 2048 Bot{Colors.ENDC}")
    print("=" * 50)
    print("This script will add firewall rules to allow mobile access.")
    print("You may need to run this as Administrator.\n")
    
    # Check if running as administrator
    try:
        is_admin = os.getuid() == 0
    except AttributeError:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    
    if not is_admin:
        print(f"{Colors.WARNING}⚠ This script may need to be run as Administrator{Colors.ENDC}")
        print("   Right-click on PowerShell/Command Prompt and select 'Run as Administrator'")
        print("   Then run: python scripts/fix_windows_firewall.py\n")
    
    # Add firewall rules for Python
    print(f"{Colors.BOLD}Adding Python firewall rules...{Colors.ENDC}")
    
    # Rule for Python backend (port 8000)
    run_command(
        'netsh advfirewall firewall add rule name="2048 Bot Backend" dir=in action=allow protocol=TCP localport=8000',
        "Adding backend firewall rule (port 8000)"
    )
    
    # Rule for Node.js frontend (port 5173)
    run_command(
        'netsh advfirewall firewall add rule name="2048 Bot Frontend" dir=in action=allow protocol=TCP localport=5173',
        "Adding frontend firewall rule (port 5173)"
    )
    
    # Rule for Python executable
    python_exe = sys.executable
    if python_exe:
        run_command(
            f'netsh advfirewall firewall add rule name="Python {os.path.basename(python_exe)}" dir=in action=allow program="{python_exe}"',
            "Adding Python executable firewall rule"
        )
    
    # Rule for Node.js (if found)
    try:
        node_result = subprocess.run(['where', 'node'], capture_output=True, text=True, shell=True)
        if node_result.returncode == 0:
            node_path = node_result.stdout.strip().split('\n')[0]
            run_command(
                f'netsh advfirewall firewall add rule name="Node.js" dir=in action=allow program="{node_path}"',
                "Adding Node.js firewall rule"
            )
    except Exception:
        print(f"{Colors.WARNING}⚠ Could not find Node.js executable{Colors.ENDC}")
    
    print(f"\n{Colors.OKGREEN}✅ Firewall rules added successfully!{Colors.ENDC}")
    print(f"\n{Colors.BOLD}Next steps:{Colors.ENDC}")
    print("1. Start your 2048 bot: python launcher.py")
    print("2. Try accessing from your phone again")
    print("3. If it still doesn't work, run the network troubleshooter:")
    print("   python scripts/network_troubleshooter.py")
    
    print(f"\n{Colors.OKCYAN}If you still have issues, the problem is likely in your Spectrum router settings.{Colors.ENDC}")

if __name__ == "__main__":
    main() 