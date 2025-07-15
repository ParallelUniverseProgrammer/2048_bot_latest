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
import psutil
import atexit
from typing import List, Optional, Tuple, Dict, Any
from urllib.parse import urlparse
from pathlib import Path
import argparse
import queue
import logging
from datetime import datetime

# Third-party imports (will be installed if missing)
try:
    import qrcode
    import qrcode.image.svg
    import netifaces
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "qrcode[pil]", "netifaces"], check=True, shell=True)
    import qrcode
    import qrcode.image.svg
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

class Logger:
    """Enhanced logging with file output"""
    
    def __init__(self, log_file: str = "launcher.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("Launcher")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def warning(self, message: str):
        self.logger.warning(message)

class PortManager:
    """Manages port availability and cleanup"""
    
    @staticmethod
    def is_port_in_use(port: int) -> bool:
        """Check if a port is in use"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('', port))
                return False
        except OSError:
            return True
    
    @staticmethod
    def find_free_port(start_port: int = 8000, max_attempts: int = 100) -> Optional[int]:
        """Find a free port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            if not PortManager.is_port_in_use(port):
                return port
        return None
    
    @staticmethod
    def kill_process_on_port(port: int) -> bool:
        """Kill any process using the specified port"""
        try:
            # Find processes using the port
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    connections = proc.info['connections']
                    if connections:
                        for conn in connections:
                            if conn.laddr.port == port:
                                print(f"{Colors.WARNING}Killing process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}{Colors.ENDC}")
                                proc.terminate()
                                proc.wait(timeout=5)
                                return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    continue
        except Exception as e:
            print(f"{Colors.WARNING}Error killing process on port {port}: {e}{Colors.ENDC}")
        
        return False
    
    @staticmethod
    def ensure_port_free(port: int, force_kill: bool = False) -> bool:
        """Ensure a port is free, optionally killing processes using it"""
        if not PortManager.is_port_in_use(port):
            return True
        
        if force_kill:
            return PortManager.kill_process_on_port(port)
        
        return False

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
        
        # Get network adapter info to avoid virtual adapters (Windows)
        if platform.system() == "Windows":
            try:
                result = subprocess.run([
                    'powershell', '-Command', 
                    'Get-NetAdapter | Where-Object {$_.Status -eq "Up"} | Get-NetIPAddress | Where-Object {$_.AddressFamily -eq "IPv4"} | Select-Object IPAddress,InterfaceAlias | ConvertTo-Json'
                ], capture_output=True, text=True, shell=True)
                
                if result.returncode == 0 and result.stdout.strip():
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
        
        # Fallback to original method
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

class ProcessMonitor:
    """Monitors and reports on process output and health"""
    
    def __init__(self, name: str, process: subprocess.Popen, logger: Logger):
        self.name = name
        self.process = process
        self.logger = logger
        self.output_queue = queue.Queue()
        self.error_queue = queue.Queue()
        self.running = True
        
        # Start monitoring threads
        self.stdout_thread = threading.Thread(target=self._monitor_stdout, daemon=True)
        self.stderr_thread = threading.Thread(target=self._monitor_stderr, daemon=True)
        self.health_thread = threading.Thread(target=self._monitor_health, daemon=True)
        
        self.stdout_thread.start()
        self.stderr_thread.start()
        self.health_thread.start()
    
    def _monitor_stdout(self):
        """Monitor stdout for output"""
        try:
            if self.process.stdout:
                for line in iter(self.process.stdout.readline, ''):
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        self.output_queue.put(line)
                        self.logger.debug(f"[{self.name}] {line}")
        except Exception as e:
            self.logger.error(f"Error monitoring {self.name} stdout: {e}")
    
    def _monitor_stderr(self):
        """Monitor stderr for errors"""
        try:
            if self.process.stderr:
                for line in iter(self.process.stderr.readline, ''):
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        self.error_queue.put(line)
                        self.logger.error(f"[{self.name}] ERROR: {line}")
                        print(f"{Colors.FAIL}[{self.name}] ERROR: {line}{Colors.ENDC}")
        except Exception as e:
            self.logger.error(f"Error monitoring {self.name} stderr: {e}")
    
    def _monitor_health(self):
        """Monitor process health"""
        while self.running:
            try:
                if self.process.poll() is not None:
                    self.logger.error(f"[{self.name}] Process terminated unexpectedly")
                    print(f"{Colors.FAIL}[{self.name}] Process terminated unexpectedly{Colors.ENDC}")
                    break
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error monitoring {self.name} health: {e}")
                break
    
    def get_recent_errors(self, count: int = 5) -> List[str]:
        """Get recent error messages"""
        errors = []
        while not self.error_queue.empty() and len(errors) < count:
            try:
                errors.append(self.error_queue.get_nowait())
            except queue.Empty:
                break
        return errors
    
    def get_recent_output(self, count: int = 5) -> List[str]:
        """Get recent output messages"""
        output = []
        while not self.output_queue.empty() and len(output) < count:
            try:
                output.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return output
    
    def stop(self):
        """Stop monitoring"""
        self.running = False

class ProcessManager:
    """Enhanced process management with monitoring"""
    
    def __init__(self, logger: Logger):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.monitors: Dict[str, ProcessMonitor] = {}
        self.threads = []
        self.running = True
        self.logger = logger
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        print(f"\n{Colors.WARNING}Received signal {signum}, shutting down...{Colors.ENDC}")
        self.cleanup()
        sys.exit(0)
    
    def add_process(self, name: str, process: subprocess.Popen):
        """Add a process to be managed with monitoring"""
        self.processes[name] = process
        self.monitors[name] = ProcessMonitor(name, process, self.logger)
        self.logger.info(f"Added process: {name} (PID: {process.pid})")
    
    def add_thread(self, thread: threading.Thread):
        """Add a thread to be managed"""
        self.threads.append(thread)
    
    def get_process_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all processes"""
        status = {}
        for name, process in self.processes.items():
            monitor = self.monitors.get(name)
            status[name] = {
                'pid': process.pid,
                'running': process.poll() is None,
                'returncode': process.poll(),
                'recent_errors': monitor.get_recent_errors() if monitor else [],
                'recent_output': monitor.get_recent_output() if monitor else []
            }
        return status
    
    def cleanup(self):
        """Enhanced cleanup with better process termination"""
        self.running = False
        
        print(f"{Colors.OKCYAN}Terminating processes...{Colors.ENDC}")
        
        # Stop all monitors
        for monitor in self.monitors.values():
            monitor.stop()
        
        # Terminate processes gracefully
        for name, process in self.processes.items():
            try:
                if process.poll() is None:  # Process is still running
                    self.logger.info(f"Terminating {name} (PID: {process.pid})")
                    process.terminate()
                    
                    # Wait for graceful termination
                    try:
                        process.wait(timeout=10)
                        self.logger.info(f"{name} terminated gracefully")
                    except subprocess.TimeoutExpired:
                        self.logger.warning(f"{name} didn't terminate gracefully, killing...")
                        process.kill()
                        process.wait()
                        self.logger.info(f"{name} killed")
                else:
                    self.logger.info(f"{name} already terminated (return code: {process.returncode})")
            except Exception as e:
                self.logger.error(f"Error terminating {name}: {e}")
                print(f"{Colors.WARNING}Error terminating {name}: {e}{Colors.ENDC}")
        
        # Wait for threads
        print(f"{Colors.OKCYAN}Waiting for threads...{Colors.ENDC}")
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)
        
        # Force cleanup of any remaining processes
        self._force_cleanup_ports()
        
        self.logger.info("Process cleanup completed")
    
    def _force_cleanup_ports(self):
        """Force cleanup of ports that might still be in use"""
        ports_to_clean = [8000, 5173, 5174]  # Backend, Frontend, HMR
        for port in ports_to_clean:
            if PortManager.is_port_in_use(port):
                self.logger.warning(f"Port {port} still in use, attempting to free it")
                if PortManager.kill_process_on_port(port):
                    self.logger.info(f"Successfully freed port {port}")
                else:
                    self.logger.warning(f"Could not free port {port}")

class ServerHealth:
    """Enhanced server health checks with detailed error reporting"""
    
    @staticmethod
    def wait_for_backend(host: str, port: int, logger: Logger, timeout: int = 30) -> bool:
        """Wait for backend to be ready with detailed error reporting"""
        url = f"http://{host}:{port}"
        start_time = time.time()
        
        print(f"{Colors.OKCYAN}Waiting for backend at {url}...{Colors.ENDC}")
        logger.info(f"Waiting for backend at {url}")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{url}/docs", timeout=2)
                if response.status_code == 200:
                    print(f"{Colors.OKGREEN}✓ Backend is ready!{Colors.ENDC}")
                    logger.info("Backend is ready")
                    return True
            except requests.ConnectionError:
                pass
            except requests.Timeout:
                pass
            except Exception as e:
                logger.debug(f"Backend health check error: {e}")
            
            time.sleep(1)
            print(".", end="", flush=True)
        
        print(f"\n{Colors.FAIL}✗ Backend failed to start within {timeout} seconds{Colors.ENDC}")
        logger.error(f"Backend failed to start within {timeout} seconds")
        return False
    
    @staticmethod
    def wait_for_frontend(host: str, port: int, logger: Logger, timeout: int = 30) -> bool:
        """Wait for frontend to be ready with detailed error reporting"""
        url = f"http://{host}:{port}"
        start_time = time.time()
        
        print(f"{Colors.OKCYAN}Waiting for frontend at {url}...{Colors.ENDC}")
        logger.info(f"Waiting for frontend at {url}")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    print(f"{Colors.OKGREEN}✓ Frontend is ready!{Colors.ENDC}")
                    logger.info("Frontend is ready")
                    return True
            except requests.ConnectionError:
                pass
            except requests.Timeout:
                pass
            except Exception as e:
                logger.debug(f"Frontend health check error: {e}")
            
            time.sleep(1)
            print(".", end="", flush=True)
        
        print(f"\n{Colors.FAIL}✗ Frontend failed to start within {timeout} seconds{Colors.ENDC}")
        logger.error(f"Frontend failed to start within {timeout} seconds")
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
    """Enhanced launcher with robust error handling and monitoring"""
    
    def __init__(self, dev_mode: bool = False, force_ports: bool = False):
        self.logger = Logger()
        self.process_manager = ProcessManager(self.logger)
        self.backend_port = 8000
        self.frontend_port = 5173
        self.host_ip = None
        self.dev_mode = dev_mode
        self.force_ports = force_ports
        
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        print(f"{Colors.OKCYAN}Checking dependencies...{Colors.ENDC}")
        self.logger.info("Checking dependencies")
        
        # Check if we're in the right directory
        if not os.path.exists("backend") or not os.path.exists("frontend"):
            error_msg = "Error: Please run this script from the project root directory"
            print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
        
        # Check Poetry
        try:
            subprocess.run(["poetry", "--version"], check=True, capture_output=True, shell=True)
            print(f"{Colors.OKGREEN}✓ Poetry found{Colors.ENDC}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            error_msg = "Poetry not found. Please install Poetry first."
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
        
        # Check Node.js
        try:
            subprocess.run(["node", "--version"], check=True, capture_output=True, shell=True)
            print(f"{Colors.OKGREEN}✓ Node.js found{Colors.ENDC}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            error_msg = "Node.js not found. Please install Node.js first."
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
        
        # Check npm
        try:
            subprocess.run(["npm", "--version"], check=True, capture_output=True, shell=True)
            print(f"{Colors.OKGREEN}✓ npm found{Colors.ENDC}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            error_msg = "npm not found. Please install npm first."
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
        
        self.logger.info("All dependencies found")
        return True
    
    def setup_network(self) -> bool:
        """Setup network configuration with port management"""
        print(f"{Colors.OKCYAN}Discovering network configuration...{Colors.ENDC}")
        self.logger.info("Setting up network configuration")
        
        # Find the best IP address
        self.host_ip = NetworkDiscovery.find_best_ip()
        if not self.host_ip:
            error_msg = "Could not determine local IP address"
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
        
        print(f"{Colors.OKGREEN}✓ Using IP address: {self.host_ip}{Colors.ENDC}")
        self.logger.info(f"Using IP address: {self.host_ip}")
        
        # Check and manage ports
        if not self._setup_ports():
            return False
        
        return True
    
    def _setup_ports(self) -> bool:
        """Setup and verify port availability"""
        ports_to_check = [
            (self.backend_port, "Backend"),
            (self.frontend_port, "Frontend"),
            (self.frontend_port + 1, "HMR")  # Hot Module Replacement
        ]
        
        for port, name in ports_to_check:
            if PortManager.is_port_in_use(port):
                if self.force_ports:
                    print(f"{Colors.WARNING}Port {port} ({name}) is in use, attempting to free it...{Colors.ENDC}")
                    self.logger.warning(f"Port {port} ({name}) is in use, attempting to free it")
                    if PortManager.kill_process_on_port(port):
                        print(f"{Colors.OKGREEN}✓ Successfully freed port {port}{Colors.ENDC}")
                        self.logger.info(f"Successfully freed port {port}")
                    else:
                        error_msg = f"Could not free port {port} ({name})"
                        print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
                        self.logger.error(error_msg)
                        return False
                else:
                    error_msg = f"Port {port} ({name}) is already in use. Use --force-ports to attempt to free it."
                    print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
                    self.logger.error(error_msg)
                    return False
        
        print(f"{Colors.OKGREEN}✓ All ports are available{Colors.ENDC}")
        self.logger.info("All ports are available")
        return True
    
    def install_dependencies(self) -> bool:
        """Install project dependencies with error reporting"""
        print(f"{Colors.OKCYAN}Installing dependencies...{Colors.ENDC}")
        self.logger.info("Installing dependencies")
        
        # Install backend dependencies
        print(f"{Colors.OKCYAN}Installing backend dependencies...{Colors.ENDC}")
        try:
            result = subprocess.run(
                ["poetry", "install"], 
                cwd="backend", 
                check=True, 
                shell=True,
                capture_output=True,
                text=True
            )
            print(f"{Colors.OKGREEN}✓ Backend dependencies installed{Colors.ENDC}")
            self.logger.info("Backend dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to install backend dependencies: {e.stderr}"
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
        
        # Install frontend dependencies
        print(f"{Colors.OKCYAN}Installing frontend dependencies...{Colors.ENDC}")
        try:
            result = subprocess.run(
                ["npm", "install"], 
                cwd="frontend", 
                check=True, 
                shell=True,
                capture_output=True,
                text=True
            )
            print(f"{Colors.OKGREEN}✓ Frontend dependencies installed{Colors.ENDC}")
            self.logger.info("Frontend dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to install frontend dependencies: {e.stderr}"
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
        
        return True
    
    def start_backend(self) -> bool:
        """Start the backend server with enhanced error reporting"""
        print(f"{Colors.OKCYAN}Starting backend server...{Colors.ENDC}")
        self.logger.info("Starting backend server")
        
        # Set up environment variables for CORS
        backend_env = os.environ.copy()
        cors_origins = [
            f"http://localhost:{self.frontend_port}",
            f"http://127.0.0.1:{self.frontend_port}",
            f"http://{self.host_ip}:{self.frontend_port}"
        ]
        backend_env["CORS_ORIGINS"] = ",".join(cors_origins)
        
        # Backend command
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
            
            self.process_manager.add_process("Backend", backend_process)
            
            # Wait for backend to be ready
            if not ServerHealth.wait_for_backend(self.host_ip, self.backend_port, self.logger):
                # Get recent errors for debugging
                status = self.process_manager.get_process_status()
                backend_status = status.get("Backend", {})
                recent_errors = backend_status.get("recent_errors", [])
                
                if recent_errors:
                    print(f"{Colors.FAIL}Recent backend errors:{Colors.ENDC}")
                    for error in recent_errors[-3:]:  # Show last 3 errors
                        print(f"{Colors.FAIL}  {error}{Colors.ENDC}")
                
                return False
            
            return True
        except Exception as e:
            error_msg = f"Failed to start backend: {e}"
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
    
    def start_frontend(self) -> bool:
        """Start the frontend server with enhanced error reporting"""
        print(f"{Colors.OKCYAN}Starting frontend server...{Colors.ENDC}")
        self.logger.info("Starting frontend server")

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
        config_path = "frontend/vite.config.temp.ts"
        try:
            with open(config_path, "w") as f:
                f.write(vite_config)
            self.logger.info("Created temporary Vite config")
        except Exception as e:
            error_msg = f"Failed to create Vite config: {e}"
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
        
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
                result = subprocess.run([
                    "npm", "run", "build", "--", "--config", "vite.config.temp.ts"
                ], cwd="frontend", check=True, shell=True, capture_output=True, text=True)
                print(f"{Colors.OKGREEN}✓ Production build completed{Colors.ENDC}")
                self.logger.info("Production build completed successfully")
            except subprocess.CalledProcessError as e:
                error_msg = f"Production build failed: {e.stderr}"
                print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
                self.logger.error(error_msg)
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
            
            self.process_manager.add_process("Frontend", frontend_process)
            
            # Wait for frontend to be ready
            if not ServerHealth.wait_for_frontend(self.host_ip, self.frontend_port, self.logger):
                # Get recent errors for debugging
                status = self.process_manager.get_process_status()
                frontend_status = status.get("Frontend", {})
                recent_errors = frontend_status.get("recent_errors", [])
                
                if recent_errors:
                    print(f"{Colors.FAIL}Recent frontend errors:{Colors.ENDC}")
                    for error in recent_errors[-3:]:  # Show last 3 errors
                        print(f"{Colors.FAIL}  {error}{Colors.ENDC}")
                
                return False
            
            return True
        except Exception as e:
            error_msg = f"Failed to start frontend: {e}"
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
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
        
        self.logger.info(f"Servers started successfully - Frontend: {frontend_url}, Backend: {backend_url}")
    
    def show_status(self):
        """Show current status of all processes"""
        status = self.process_manager.get_process_status()
        
        print(f"\n{Colors.HEADER}📊 Process Status{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*30}{Colors.ENDC}")
        
        for name, info in status.items():
            status_icon = "🟢" if info['running'] else "🔴"
            print(f"{status_icon} {name}: {'Running' if info['running'] else 'Stopped'} (PID: {info['pid']})")
            
            if info['recent_errors']:
                print(f"   Recent errors:")
                for error in info['recent_errors'][-2:]:  # Show last 2 errors
                    print(f"   - {error}")
    
    def run(self):
        """Main launcher routine with enhanced error handling"""
        print(f"{Colors.HEADER}🚀 2048 Transformer Training Launcher{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
        
        self.logger.info("Launcher started")
        
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
                    
                    # Show status every 30 seconds
                    if int(time.time()) % 30 == 0:
                        self.show_status()
                        
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}Shutting down...{Colors.ENDC}")
                self.logger.info("Shutdown requested by user")
            
            return True
            
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
        
        finally:
            # Cleanup
            self.process_manager.cleanup()
            
            # Remove temporary files
            temp_config = "frontend/vite.config.temp.ts"
            if os.path.exists(temp_config):
                try:
                    os.remove(temp_config)
                    self.logger.info("Removed temporary Vite config")
                except Exception as e:
                    self.logger.warning(f"Could not remove temporary config: {e}")
            
            print(f"{Colors.OKGREEN}✓ Cleanup completed{Colors.ENDC}")
            self.logger.info("Launcher cleanup completed")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Launch 2048 training stack")
    parser.add_argument("--dev", action="store_true", help="Run frontend in Vite dev mode (HMR)")
    parser.add_argument("--force-ports", action="store_true", help="Force kill processes using required ports")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Set logging level")
    args = parser.parse_args()

    launcher = Launcher(dev_mode=args.dev, force_ports=args.force_ports)
    success = launcher.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 