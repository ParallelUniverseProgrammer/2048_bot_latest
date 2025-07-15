#!/usr/bin/env python3
"""
2048 Transformer Training Launcher
Platform-agnostic launcher script that starts the backend and frontend servers
and generates a QR code for mobile access on the same LAN.
"""

import os
import sys
import time
import socket
import subprocess
import threading
import requests
import json
import platform
import asyncio
import signal
from typing import List, Optional, Tuple
from urllib.parse import urlparse
from pathlib import Path
import argparse

# Third-party imports (will be installed if missing)
try:
    import qrcode
    import qrcode.image.svg
    import netifaces
    import psutil
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "qrcode[pil]", "netifaces", "psutil"], check=True, shell=True)
    import qrcode
    import qrcode.image.svg
    import netifaces
    import psutil

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

class NetworkDiscovery:
    """Handles network discovery and IP address detection"""
    
    @staticmethod
    def get_local_ip_addresses() -> List[str]:
        """Get all possible local IP addresses"""
        ip_addresses = []
        
        # Method 1: Using netifaces to get all network interfaces
        try:
            for interface in netifaces.interfaces():
                try:
                    addresses = netifaces.ifaddresses(interface)
                    if netifaces.AF_INET in addresses:
                        for addr_info in addresses[netifaces.AF_INET]:
                            ip = addr_info.get('addr')
                            if ip and not ip.startswith('127.') and not ip.startswith('169.254.'):
                                ip_addresses.append(ip)
                except Exception:
                    continue
        except Exception:
            pass
        
        # Method 2: Socket-based discovery
        try:
            # Connect to Google DNS to find the preferred route
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                if ip not in ip_addresses:
                    ip_addresses.append(ip)
        except Exception:
            pass
        
        # Method 3: Get hostname IP
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            if ip and not ip.startswith('127.') and ip not in ip_addresses:
                ip_addresses.append(ip)
        except Exception:
            pass
        
        return ip_addresses
    
    @staticmethod
    def find_best_ip() -> Optional[str]:
        """Find the best IP address for LAN access"""
        ip_addresses = NetworkDiscovery.get_local_ip_addresses()
        
        if not ip_addresses:
            return None
        
        # Get network adapter info to avoid virtual adapters
        try:
            import subprocess
            result = subprocess.run([
                'powershell', '-Command', 
                'Get-NetAdapter | Where-Object {$_.Status -eq "Up"} | Get-NetIPAddress | Where-Object {$_.AddressFamily -eq "IPv4"} | Select-Object IPAddress,InterfaceAlias | ConvertTo-Json'
            ], capture_output=True, text=True, shell=True)
            
            if result.returncode == 0 and result.stdout.strip():
                import json
                adapters = json.loads(result.stdout)
                if not isinstance(adapters, list):
                    adapters = [adapters]
                
                # Prioritize real network adapters over virtual ones
                real_adapters = []
                virtual_adapters = []
                
                for adapter in adapters:
                    ip = adapter['IPAddress']
                    interface = adapter['InterfaceAlias'].lower()
                    
                    # Skip loopback and other non-LAN addresses
                    if ip.startswith('127.') or ip.startswith('169.254.'):
                        continue
                    
                    # Identify virtual adapters
                    if any(keyword in interface for keyword in ['wsl', 'hyper-v', 'virtual', 'tap', 'vpn', 'vethernet']):
                        virtual_adapters.append(ip)
                    else:
                        real_adapters.append(ip)
                
                # Prefer real adapters
                preferred_ips = real_adapters if real_adapters else virtual_adapters
                
                if preferred_ips:
                    # From real adapters, prefer 192.168.x.x for home networks
                    for ip in preferred_ips:
                        if ip.startswith('192.168.'):
                            return ip
                    
                    # Then prefer 10.x.x.x
                    for ip in preferred_ips:
                        if ip.startswith('10.'):
                            return ip
                    
                    # Finally accept 172.x.x.x
                    for ip in preferred_ips:
                        if ip.startswith('172.'):
                            return ip
                    
                    # Return first available
                    return preferred_ips[0]
        except Exception:
            pass
        
        # Fallback to original method if PowerShell fails
        private_ranges = [
            ('192.168.', 1),  # Most common home networks
            ('10.', 2),       # Corporate networks  
            ('172.', 3),      # Less common private range
        ]
        
        # Score and sort IP addresses
        scored_ips = []
        for ip in ip_addresses:
            score = 10  # Default score
            for prefix, bonus in private_ranges:
                if ip.startswith(prefix):
                    score -= bonus
                    break
            scored_ips.append((score, ip))
        
        # Return the best IP (lowest score)
        scored_ips.sort()
        return scored_ips[0][1] if scored_ips else None

class ProcessManager:
    """Manages background processes"""
    
    def __init__(self):
        self.processes = []
        self.threads = []
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        print(f"\n{Colors.WARNING}Received signal {signum}, shutting down...{Colors.ENDC}")
        self.cleanup()
        sys.exit(0)
    
    def add_process(self, process: subprocess.Popen):
        """Add a process to be managed"""
        self.processes.append(process)
    
    def add_thread(self, thread: threading.Thread):
        """Add a thread to be managed"""
        self.threads.append(thread)
    
    def cleanup(self):
        """Clean up all processes and threads"""
        self.running = False
        
        print(f"{Colors.OKCYAN}Terminating processes...{Colors.ENDC}")
        for process in self.processes:
            try:
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            except Exception as e:
                print(f"{Colors.WARNING}Error terminating process: {e}{Colors.ENDC}")
        
        print(f"{Colors.OKCYAN}Waiting for threads...{Colors.ENDC}")
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2)

class ServerHealth:
    """Handles server health checks"""
    
    @staticmethod
    def wait_for_backend(host: str, port: int, timeout: int = 30) -> bool:
        """Wait for backend to be ready"""
        url = f"http://{host}:{port}"
        start_time = time.time()
        
        print(f"{Colors.OKCYAN}Waiting for backend at {url}...{Colors.ENDC}")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{url}/docs", timeout=2)
                if response.status_code == 200:
                    print(f"{Colors.OKGREEN}✓ Backend is ready!{Colors.ENDC}")
                    return True
            except requests.RequestException:
                pass
            
            time.sleep(1)
            print(".", end="", flush=True)
        
        print(f"\n{Colors.FAIL}✗ Backend failed to start within {timeout} seconds{Colors.ENDC}")
        return False
    
    @staticmethod
    def wait_for_frontend(host: str, port: int, timeout: int = 30) -> bool:
        """Wait for frontend to be ready"""
        url = f"http://{host}:{port}"
        start_time = time.time()
        
        print(f"{Colors.OKCYAN}Waiting for frontend at {url}...{Colors.ENDC}")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    print(f"{Colors.OKGREEN}✓ Frontend is ready!{Colors.ENDC}")
                    return True
            except requests.RequestException:
                pass
            
            time.sleep(1)
            print(".", end="", flush=True)
        
        print(f"\n{Colors.FAIL}✗ Frontend failed to start within {timeout} seconds{Colors.ENDC}")
        return False

class QRCodeGenerator:
    """Generates QR codes for mobile access"""
    
    @staticmethod
    def generate_qr_code(url: str, output_path: Optional[str] = None) -> None:
        """Generate and display QR code"""
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        
        # Save image if path provided
        if output_path:
            img = qr.make_image(fill_color="black", back_color="white")
            img.save(output_path)
            print(f"{Colors.OKGREEN}QR code saved to: {output_path}{Colors.ENDC}")
        
        # Display in terminal
        print(f"\n{Colors.HEADER}{'='*50}{Colors.ENDC}")
        print(f"{Colors.HEADER}📱 MOBILE ACCESS QR CODE{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
        print(f"{Colors.OKBLUE}URL: {url}{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
        
        # Print QR code to terminal
        qr_terminal = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=1,
            border=2,
        )
        qr_terminal.add_data(url)
        qr_terminal.make(fit=True)
        
        # Print as ASCII
        matrix = qr_terminal.get_matrix()
        for row in matrix:
            print(''.join(['██' if cell else '  ' for cell in row]))
        
        print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")

class Launcher:
    """Main launcher class"""
    
    def __init__(self, dev_mode: bool = False):
        self.process_manager = ProcessManager()
        self.backend_port = 8000
        self.frontend_port = 5173
        self.host_ip = None
        self.dev_mode = dev_mode
        
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        print(f"{Colors.OKCYAN}Checking dependencies...{Colors.ENDC}")
        
        # Check if we're in the right directory
        if not os.path.exists("backend") or not os.path.exists("frontend"):
            print(f"{Colors.FAIL}Error: Please run this script from the project root directory{Colors.ENDC}")
            return False
        
        # Check Poetry
        try:
            subprocess.run(["poetry", "--version"], check=True, capture_output=True, shell=True)
            print(f"{Colors.OKGREEN}✓ Poetry found{Colors.ENDC}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{Colors.FAIL}✗ Poetry not found. Please install Poetry first.{Colors.ENDC}")
            return False
        
        # Check Node.js
        try:
            subprocess.run(["node", "--version"], check=True, capture_output=True, shell=True)
            print(f"{Colors.OKGREEN}✓ Node.js found{Colors.ENDC}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{Colors.FAIL}✗ Node.js not found. Please install Node.js first.{Colors.ENDC}")
            return False
        
        # Check npm
        try:
            subprocess.run(["npm", "--version"], check=True, capture_output=True, shell=True)
            print(f"{Colors.OKGREEN}✓ npm found{Colors.ENDC}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{Colors.FAIL}✗ npm not found. Please install npm first.{Colors.ENDC}")
            return False
        
        return True
    
    def setup_network(self) -> bool:
        """Setup network configuration"""
        print(f"{Colors.OKCYAN}Discovering network configuration...{Colors.ENDC}")
        
        # Find the best IP address
        self.host_ip = NetworkDiscovery.find_best_ip()
        if not self.host_ip:
            print(f"{Colors.FAIL}✗ Could not determine local IP address{Colors.ENDC}")
            return False
        
        print(f"{Colors.OKGREEN}✓ Using IP address: {self.host_ip}{Colors.ENDC}")
        
        # Check if ports are available
        if not self._check_port_available(self.backend_port):
            print(f"{Colors.FAIL}✗ Port {self.backend_port} is already in use{Colors.ENDC}")
            return False
        
        if not self._check_port_available(self.frontend_port):
            print(f"{Colors.FAIL}✗ Port {self.frontend_port} is already in use{Colors.ENDC}")
            return False
        
        print(f"{Colors.OKGREEN}✓ Ports {self.backend_port} and {self.frontend_port} are available{Colors.ENDC}")
        return True
    
    def _check_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('', port))
                return True
        except OSError:
            return False
    
    def install_dependencies(self) -> bool:
        """Install project dependencies"""
        print(f"{Colors.OKCYAN}Installing dependencies...{Colors.ENDC}")
        
        # Install backend dependencies
        print(f"{Colors.OKCYAN}Installing backend dependencies...{Colors.ENDC}")
        try:
            subprocess.run(["poetry", "install"], cwd="backend", check=True, shell=True)
            print(f"{Colors.OKGREEN}✓ Backend dependencies installed{Colors.ENDC}")
        except subprocess.CalledProcessError:
            print(f"{Colors.FAIL}✗ Failed to install backend dependencies{Colors.ENDC}")
            return False
        
        # Install frontend dependencies
        print(f"{Colors.OKCYAN}Installing frontend dependencies...{Colors.ENDC}")
        try:
            subprocess.run(["npm", "install"], cwd="frontend", check=True, shell=True)
            print(f"{Colors.OKGREEN}✓ Frontend dependencies installed{Colors.ENDC}")
        except subprocess.CalledProcessError:
            print(f"{Colors.FAIL}✗ Failed to install frontend dependencies{Colors.ENDC}")
            return False
        
        return True
    
    def start_backend(self) -> bool:
        """Start the backend server"""
        print(f"{Colors.OKCYAN}Starting backend server...{Colors.ENDC}")
        
        # Set up environment variables for CORS
        backend_env = os.environ.copy()
        cors_origins = [
            f"http://localhost:{self.frontend_port}",
            f"http://127.0.0.1:{self.frontend_port}",
            f"http://{self.host_ip}:{self.frontend_port}"
        ]
        backend_env["CORS_ORIGINS"] = ",".join(cors_origins)
        
        # Modify the backend to accept external connections
        backend_cmd = [
            "poetry", "run", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", str(self.backend_port),
            "--reload"
        ]
        
        try:
            backend_process = subprocess.Popen(
                backend_cmd,
                cwd="backend",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=backend_env,
                shell=True
            )
            self.process_manager.add_process(backend_process)
            
            # Wait for backend to be ready
            if not ServerHealth.wait_for_backend(self.host_ip, self.backend_port):
                return False
            
            return True
        except Exception as e:
            print(f"{Colors.FAIL}✗ Failed to start backend: {e}{Colors.ENDC}")
            return False
    
    def start_frontend(self) -> bool:
        """Start the frontend server (dev or preview)"""
        print(f"{Colors.OKCYAN}Starting frontend server...{Colors.ENDC}")

        # Create a temporary Vite config that injects the correct backend URL
        vite_config = f"""
import {{ defineConfig }} from 'vite'
import react from '@vitejs/plugin-react'
import {{ VitePWA }} from 'vite-plugin-pwa'

export default defineConfig({{
  plugins: [
    react({{
      // Disable fast refresh for mobile compatibility
      refresh: false
    }}),
    VitePWA({{
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'favicon-16x16.png'],
      manifest: {{
        name: '2048 Transformer Training',
        short_name: '2048 AI',
        description: 'Real-time visualization for 2048 transformer training',
        theme_color: '#3b82f6',
        background_color: '#0f172a',
        display: 'standalone',
        orientation: 'portrait',
        scope: '/',
        start_url: '/',
        icons: [
          {{
            src: 'pwa-192x192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'any maskable'
          }},
          {{
            src: 'pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable'
          }}
        ]
      }}
    }})
  ],
  server: {{
    host: '0.0.0.0',
    port: {self.frontend_port},
    strictPort: true,
    // Mobile-friendly server settings
    hmr: {{
      port: {self.frontend_port + 1},
      host: '0.0.0.0'
    }},
    // Longer timeout for mobile connections
    timeout: 30000,
    // CORS settings for mobile
    cors: {{
      origin: ['http://{self.host_ip}:{self.frontend_port}', 'http://localhost:{self.frontend_port}'],
      credentials: true
    }}
  }},
  define: {{
    __BACKEND_URL__: JSON.stringify('http://{self.host_ip}:{self.backend_port}')
  }},
  // Build optimizations for mobile
  build: {{
    target: 'es2015',
    minify: false,
    sourcemap: true
  }}
}})
"""
        
        # Write temporary config
        with open("frontend/vite.config.temp.ts", "w") as f:
            f.write(vite_config)
        
        if self.dev_mode:
            frontend_cmd = [
                "npm", "run", "dev", "--", 
                "--config", "vite.config.temp.ts",
                "--host", "0.0.0.0",
                "--port", str(self.frontend_port),
                "--force"
            ]
        else:
            # Build production bundle first
            print(f"{Colors.OKCYAN}Building production bundle...{Colors.ENDC}")
            try:
                subprocess.run([
                    "npm", "run", "build", "--", "--config", "vite.config.temp.ts"
                ], cwd="frontend", check=True, shell=True)
                print(f"{Colors.OKGREEN}✓ Production build completed{Colors.ENDC}")
            except subprocess.CalledProcessError:
                print(f"{Colors.FAIL}✗ Production build failed{Colors.ENDC}")
                return False

            frontend_cmd = [
                "npm", "run", "preview", "--", 
                "--config", "vite.config.temp.ts",
                "--host", "0.0.0.0",
                "--port", str(self.frontend_port)
            ]
        
        try:
            frontend_process = subprocess.Popen(
                frontend_cmd,
                cwd="frontend",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True
            )
            self.process_manager.add_process(frontend_process)
            
            # Wait for frontend to be ready
            if not ServerHealth.wait_for_frontend(self.host_ip, self.frontend_port):
                return False
            
            return True
        except Exception as e:
            print(f"{Colors.FAIL}✗ Failed to start frontend: {e}{Colors.ENDC}")
            return False
    
    def show_access_info(self):
        """Display access information and QR code"""
        frontend_url = f"http://{self.host_ip}:{self.frontend_port}"
        backend_url = f"http://{self.host_ip}:{self.backend_port}"
        
        print(f"\n{Colors.HEADER}🚀 2048 Transformer Training Server Started!{Colors.ENDC}")
        print(f"{Colors.OKGREEN}Frontend: {frontend_url}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}Backend API: {backend_url}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}Backend Docs: {backend_url}/docs{Colors.ENDC}")
        
        # Generate QR code
        QRCodeGenerator.generate_qr_code(frontend_url, "mobile_access_qr.png")
        
        print(f"\n{Colors.OKCYAN}📱 Scan the QR code above with your phone to access the app!{Colors.ENDC}")
        print(f"{Colors.WARNING}Press Ctrl+C to stop the servers{Colors.ENDC}")
    
    def run(self):
        """Main launcher routine"""
        print(f"{Colors.HEADER}🚀 2048 Transformer Training Launcher{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
        
        try:
            # Check dependencies
            if not self.check_dependencies():
                return False
            
            # Setup network
            if not self.setup_network():
                return False
            
            # Install dependencies
            if not self.install_dependencies():
                return False
            
            # Start backend
            if not self.start_backend():
                return False
            
            # Start frontend
            if not self.start_frontend():
                return False
            
            # Show access information
            self.show_access_info()
            
            # Keep running until interrupted
            try:
                while self.process_manager.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}Shutting down...{Colors.ENDC}")
            
            return True
            
        except Exception as e:
            print(f"{Colors.FAIL}✗ Unexpected error: {e}{Colors.ENDC}")
            return False
        
        finally:
            # Cleanup
            self.process_manager.cleanup()
            
            # Remove temporary files
            temp_config = "frontend/vite.config.temp.ts"
            if os.path.exists(temp_config):
                os.remove(temp_config)
            
            print(f"{Colors.OKGREEN}✓ Cleanup completed{Colors.ENDC}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Launch 2048 training stack")
    parser.add_argument("--dev", action="store_true", help="Run frontend in Vite dev mode (HMR)")
    args = parser.parse_args()

    launcher = Launcher(dev_mode=args.dev)
    success = launcher.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 