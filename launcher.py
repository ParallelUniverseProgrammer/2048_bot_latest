#!/usr/bin/env python3
"""2048 Training Launcher
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
import shutil
import math

# Third-party imports (will be installed if missing)
try:
    import qrcode
    import qrcode.image.svg
    import netifaces
    from qrcode import constants
    # GUI imports for QR-only mode
    import tkinter as tk
    from tkinter import ttk, messagebox
    from PIL import Image, ImageTk
    import math
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "qrcode[pil]", "netifaces", "pillow"], check=True, shell=True)
    import qrcode
    import qrcode.image.svg
    import netifaces
    from qrcode import constants
    # GUI imports for QR-only mode
    import tkinter as tk
    from tkinter import ttk, messagebox
    from PIL import Image, ImageTk
    import math

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

class ConsoleUI:
    """Ultra-responsive console UI with high-frequency animations and smooth transitions"""
    
    def __init__(self, quiet: bool = False, no_color: bool = False):
        self.quiet = quiet
        self.no_color = no_color
        
        # Terminal size detection with fallback
        try:
            terminal_size = shutil.get_terminal_size((80, 24))
            self.terminal_width = terminal_size.columns
            self.terminal_height = terminal_size.lines
        except (OSError, AttributeError):
            # Fallback for terminals that don't support size detection
            self.terminal_width = 80
            self.terminal_height = 24
        
        # Animation state
        self.animation_running = False
        self.animation_thread = None
        self.animation_queue = queue.Queue()
        
        # Progress state
        self.current_step = ""
        self.target_progress = 0.0
        self.current_progress = 0.0
        self.status_message = ""
        self.error_message = ""
        self.micro_progress = 0.0
        self.micro_step = ""
        
        # Animation frames
        self.spinner_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.pulse_frames = ['●', '○', '●', '○', '●', '○']
        self.dots_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.bar_frames = ['▰', '▱', '▰', '▱', '▰', '▱']
        
        self.frame_index = 0
        self.last_frame_time = time.perf_counter()  # Use perf_counter for precise timing
        self.frame_rate = 60  # Fixed: Use proper 60fps instead of 10fps
        self.frame_interval = 1.0 / self.frame_rate
        
        # Animation state machine
        self.animation_state = "idle"  # idle, running, transitioning, error
        self.state_start_time = time.perf_counter()
        
        # Terminal compatibility flags
        self.terminal_supports_colors = self._check_color_support()
        self.terminal_supports_unicode = self._check_unicode_support()
        
        # Setup terminal for better experience
        self._setup_terminal()
        
        # Start animation thread
        self._start_animation_thread()
    
    def _check_color_support(self) -> bool:
        """Check if terminal supports colors"""
        if self.no_color:
            return False
        
        # Check for common color-supporting terminals
        term = os.environ.get('TERM', '').lower()
        color_terms = ['xterm', 'linux', 'screen', 'tmux', 'vt100', 'ansi']
        return any(color_term in term for color_term in color_terms)
    
    def _check_unicode_support(self) -> bool:
        """Check if terminal supports Unicode characters"""
        try:
            # Test Unicode output
            sys.stdout.write('█')
            sys.stdout.flush()
            return True
        except UnicodeEncodeError:
            return False
    
    def _setup_terminal(self):
        """Setup terminal for better display"""
        if not self.quiet:
            self.clear_screen()
            self.hide_cursor()
    
    def clear_screen(self):
        """Clear the entire terminal screen"""
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')
    
    def hide_cursor(self):
        """Hide the terminal cursor"""
        if not self.quiet:
            sys.stdout.write('\033[?25l')
            sys.stdout.flush()
    
    def show_cursor(self):
        """Show the terminal cursor"""
        if not self.quiet:
            sys.stdout.write('\033[?25h')
            sys.stdout.flush()
    
    def move_cursor(self, row: int, col: int):
        """Move cursor to specific position"""
        if not self.quiet:
            sys.stdout.write(f'\033[{row};{col}H')
            sys.stdout.flush()
    
    def clear_line(self):
        """Clear current line"""
        if not self.quiet:
            sys.stdout.write('\033[K')
            sys.stdout.flush()
    
    def _start_animation_thread(self):
        """Start the high-frequency animation thread"""
        if self.quiet:
            return
            
        self.animation_running = True
        self.animation_state = "running"
        self.state_start_time = time.perf_counter()
        self.animation_thread = threading.Thread(target=self._animation_loop, daemon=True)
        self.animation_thread.start()
    
    def _animation_loop(self):
        """Ultra-smooth animation loop running at 60fps with precise timing"""
        last_time = time.perf_counter()
        frame_count = 0
        
        while self.animation_running:
            try:
                current_time = time.perf_counter()
                delta_time = current_time - last_time
                
                # Ultra-precise frame rate limiting
                if delta_time >= self.frame_interval:
                    # Update frame index with proper timing
                    self.frame_index = (self.frame_index + 1) % len(self.spinner_frames)
                    last_time = current_time
                    frame_count += 1
                    
                    # Ultra-smooth progress interpolation with improved easing
                    if abs(self.current_progress - self.target_progress) > 0.0005:  # Higher precision threshold
                        # Use improved easing for smoother movement
                        progress_diff = self.target_progress - self.current_progress
                        easing_factor = 0.12  # Slightly more responsive easing
                        self.current_progress += progress_diff * easing_factor
                        
                        # Clamp to prevent overshooting
                        if abs(self.current_progress - self.target_progress) < 0.001:
                            self.current_progress = self.target_progress
                    
                    # Render the current frame
                    self._render_frame()
                    
                else:
                    # Ultra-precise sleep for remaining time
                    sleep_time = self.frame_interval - delta_time
                    if sleep_time > 0.001:  # Only sleep if significant time remains
                        time.sleep(sleep_time * 0.8)  # Sleep 80% of remaining time for better responsiveness
                
            except Exception as e:
                # Continue animation even if there's an error, but log it
                time.sleep(self.frame_interval)
    
    def _render_frame(self):
        """Render a single animation frame with true double-buffered rendering"""
        try:
            # Calculate positions
            center_x = self.terminal_width // 2
            center_y = self.terminal_height // 2
            
            # Build current content hash for efficient change detection
            current_content_hash = hash(f"{self.current_step}{self.current_progress:.3f}{self.status_message}{self.error_message}{self.micro_step}{self.micro_progress:.3f}")
            
            # Only re-render if content has actually changed
            if not hasattr(self, '_last_content_hash') or current_content_hash != self._last_content_hash:
                self._last_content_hash = current_content_hash
                
                # Use cursor positioning instead of screen clearing for smooth updates
                if not hasattr(self, '_first_render'):
                    self._first_render = True
                    self.clear_screen()  # Only clear on first render
                else:
                    # Move cursor to top for smooth updates
                    self.move_cursor(1, 1)
                
                # Header with animated title
                header = "🚀 2048 Bot Training Launcher"
                if not self.terminal_supports_unicode:
                    header = "2048 Bot Training Launcher"  # Fallback without emoji
                title_animation = self._get_title_animation(header)
                print(f"\n{Colors.HEADER}{title_animation:^{self.terminal_width}}{Colors.ENDC}")
                
                # Progress section with proper spacing
                print(f"\n{' ' * (center_x - 20)}")
                
                # Animated spinner with current step
                spinner = self.spinner_frames[self.frame_index]
                if not self.terminal_supports_unicode:
                    spinner = "|/-\\"[self.frame_index % 4]  # ASCII fallback
                step_display = self.current_step if self.current_step else "Initializing..."
                print(f"{' ' * (center_x - 15)}{Colors.OKBLUE}{spinner}{Colors.ENDC} {Colors.BOLD}{step_display}{Colors.ENDC}")
                
                # Smooth progress bar with proper width calculation
                bar_width = min(40, self.terminal_width - 20)  # Responsive width
                filled_width = int(bar_width * self.current_progress)
                
                # Use appropriate characters based on terminal support
                if self.terminal_supports_unicode:
                    filled_char = '█'
                    empty_char = '░'
                else:
                    filled_char = '#'
                    empty_char = '-'
                
                bar = filled_char * filled_width + empty_char * (bar_width - filled_width)
                
                # Add animated end cap
                if self.current_progress > 0 and self.current_progress < 1.0:
                    if self.terminal_supports_unicode:
                        end_cap = self.bar_frames[self.frame_index % len(self.bar_frames)]
                    else:
                        end_cap = ">"  # ASCII fallback
                    if filled_width < bar_width:
                        bar = bar[:-1] + end_cap
                
                print(f"{' ' * (center_x - bar_width//2)}{Colors.OKGREEN}{bar}{Colors.ENDC}")
                
                # Progress percentage with smooth animation
                percentage = int(self.current_progress * 100)
                print(f"{' ' * (center_x - 3)}{Colors.OKCYAN}{percentage:3d}%{Colors.ENDC}")
                
                # Micro-progress indicator
                if self.micro_step and self.micro_progress > 0:
                    micro_bar_width = min(20, self.terminal_width - 20)
                    micro_filled = int(micro_bar_width * self.micro_progress)
                    
                    # Use appropriate characters for micro-progress
                    if self.terminal_supports_unicode:
                        micro_filled_char = '▰'
                        micro_empty_char = '▱'
                    else:
                        micro_filled_char = '='
                        micro_empty_char = '-'
                    
                    micro_bar = micro_filled_char * micro_filled + micro_empty_char * (micro_bar_width - micro_filled)
                    print(f"{' ' * (center_x - micro_bar_width//2)}{Colors.OKBLUE}{micro_bar}{Colors.ENDC}")
                    print(f"{' ' * (center_x - len(self.micro_step)//2)}{Colors.OKBLUE}{self.micro_step}{Colors.ENDC}")
                
                # Status message with typing effect
                if self.status_message:
                    status_display = self._get_typing_effect(self.status_message)
                    print(f"\n{' ' * (center_x - len(status_display)//2)}{Colors.OKBLUE}{status_display}{Colors.ENDC}")
                
                # Error message with pulse effect
                if self.error_message:
                    error_display = self._get_pulse_effect(f"❌ {self.error_message}")
                    if not self.terminal_supports_unicode:
                        error_display = f"ERROR: {self.error_message}"  # Fallback without emoji
                    print(f"\n{' ' * (center_x - len(error_display)//2)}{Colors.FAIL}{error_display}{Colors.ENDC}")
                
                # Footer with animated dots
                footer = "Press Ctrl+C to stop"
                footer_animation = self._get_dots_animation(footer)
                print(f"\n{' ' * (center_x - len(footer_animation)//2)}{Colors.WARNING}{footer_animation}{Colors.ENDC}")
                
                sys.stdout.flush()
            
        except Exception as e:
            # Fallback to simple rendering if animation fails
            if not hasattr(self, '_fallback_rendered'):
                self._fallback_rendered = True
                print(f"Animation error: {e}")
                print("Falling back to simple display mode")
                self._render_simple_fallback()
    
    def _render_simple_fallback(self):
        """Simple fallback rendering when animation system fails"""
        try:
            self.clear_screen()
            print("🚀 2048 Bot Training Launcher")
            print(f"Step: {self.current_step}")
            print(f"Progress: {int(self.current_progress * 100)}%")
            if self.status_message:
                print(f"Status: {self.status_message}")
            if self.error_message:
                print(f"Error: {self.error_message}")
            print("Press Ctrl+C to stop")
        except Exception:
            # Ultimate fallback - just print basic info
            print(f"Step: {self.current_step}, Progress: {int(self.current_progress * 100)}%")
    
    def _get_title_animation(self, title: str) -> str:
        """Animate the title with subtle effects"""
        if self.frame_index % 20 < 10:
            return title
        else:
            return title.replace("🚀", "⚡")
    
    def _get_typing_effect(self, text: str) -> str:
        """Create a typing effect for status messages with proper state management"""
        if not text:
            return ""
        
        # Calculate typing progress based on frame index and state
        state_duration = 60  # frames per character
        total_chars = len(text)
        
        # Calculate how many characters should be shown
        chars_to_show = min(total_chars, int(self.frame_index / state_duration))
        
        # Ensure typing completes and stays complete
        if chars_to_show >= total_chars:
            return text
        
        # Show partial text with blinking cursor
        cursor_char = "▋" if (self.frame_index // 5) % 2 == 0 else " "
        return text[:chars_to_show] + cursor_char
    
    def _get_pulse_effect(self, text: str) -> str:
        """Create a smooth pulse effect for error messages"""
        # Slower, more subtle pulse effect
        pulse_cycle = (self.frame_index // 15) % 4  # Slower pulse
        if pulse_cycle < 2:
            return text
        else:
            return text.replace("❌", "⚠️")
    
    def _get_dots_animation(self, text: str) -> str:
        """Add smooth animated dots to footer"""
        # Smoother dots animation
        dots_count = (self.frame_index // 8) % 4  # Slower, smoother animation
        dots = "." * dots_count
        return text + dots
    
    def update_progress(self, step: str, progress: float, status: str = "", error: str = ""):
        """Update progress with smooth interpolation and state management"""
        # State transition handling
        if step != self.current_step:
            self.animation_state = "transitioning"
            self.state_start_time = time.perf_counter()
        
        self.current_step = step
        self.target_progress = max(0.0, min(1.0, progress))  # Clamp to valid range
        self.status_message = status
        self.error_message = error
        
        # Update animation state
        if self.animation_state == "transitioning":
            # Return to running state after brief transition
            if time.perf_counter() - self.state_start_time > 0.1:
                self.animation_state = "running"
    
    def update_micro_progress(self, step: str, progress: float):
        """Update micro-progress with ultra-smooth interpolation"""
        if self.quiet:
            return
        
        # Update micro step and target progress
        self.micro_step = step
        self.target_micro_progress = max(0.0, min(1.0, progress))  # Clamp to 0.0-1.0
        
        # Ultra-smooth interpolation for micro-progress
        if abs(self.micro_progress - self.target_micro_progress) > 0.0005:
            progress_diff = self.target_micro_progress - self.micro_progress
            easing_factor = 0.18  # Slightly faster easing for micro-progress
            self.micro_progress += progress_diff * easing_factor
            
            # Clamp to prevent overshooting
            if abs(self.micro_progress - self.target_micro_progress) < 0.001:
                self.micro_progress = self.target_micro_progress
    
    def render_progress_screen(self, step: str, progress: float, status: str = "", error: str = ""):
        """Render a beautiful progress screen (legacy method for compatibility)"""
        self.update_progress(step, progress, status, error)
    
    def render_qr_screen(self, frontend_url: str, backend_url: str):
        """Render the final QR code screen with enhanced animations and smooth transitions"""
        if self.quiet:
            return
        
        # Smooth transition to QR screen
        self.animation_state = "transitioning"
        self.state_start_time = time.perf_counter()
        
        # Stop animation thread for QR screen
        self.animation_running = False
        if self.animation_thread:
            self.animation_thread.join(timeout=1)
        
        # Use cursor positioning instead of screen clearing for smooth transition
        self.move_cursor(1, 1)
        
        # Calculate positions
        center_x = self.terminal_width // 2
        
        # Animated header with celebration effect
        header = "🎉 2048 Bot Training Ready!"
        if not self.terminal_supports_unicode:
            header = "2048 Bot Training Ready!"  # Fallback without emoji
        print(f"\n{Colors.HEADER}{header:^{self.terminal_width}}{Colors.ENDC}")
        
        # Animated separator
        separator = "=" * min(50, self.terminal_width - 10)
        print(f"\n{Colors.HEADER}{separator:^{self.terminal_width}}{Colors.ENDC}")
        
        # URLs with highlight effect
        print(f"\n{Colors.OKGREEN}{'Frontend:':<12} {frontend_url}{Colors.ENDC}".center(self.terminal_width))
        print(f"{Colors.OKGREEN}{'Backend:':<12} {backend_url}{Colors.ENDC}".center(self.terminal_width))
        print(f"{Colors.OKGREEN}{'Docs:':<12} {backend_url}/docs{Colors.ENDC}".center(self.terminal_width))
        
        # Separator
        print(f"\n{Colors.HEADER}{separator:^{self.terminal_width}}{Colors.ENDC}")
        
        # QR Code with enhanced display
        qr_title = "📱 QR Code for Mobile Access"
        if not self.terminal_supports_unicode:
            qr_title = "QR Code for Mobile Access"  # Fallback without emoji
        print(f"\n{Colors.OKCYAN}{qr_title:^{self.terminal_width}}{Colors.ENDC}")
        
        # Generate QR code with proper terminal compatibility
        QRCodeGenerator.generate_qr_code(frontend_url, "mobile_access_qr.png", center=True, term_width=self.terminal_width)
        
        # Enhanced instructions
        instruction_title = "📱 Scan this QR code with your phone!"
        if not self.terminal_supports_unicode:
            instruction_title = "Scan this QR code with your phone!"  # Fallback without emoji
        print(f"\n{Colors.OKCYAN}{instruction_title:^{self.terminal_width}}{Colors.ENDC}")
        
        ios_msg = "💡 iOS: Tap share (📤) then 'Add to Home Screen'"
        if not self.terminal_supports_unicode:
            ios_msg = "iOS: Tap share then 'Add to Home Screen'"  # Fallback without emoji
        print(f"{Colors.WARNING}{ios_msg:^{self.terminal_width}}{Colors.ENDC}")
        
        # Footer
        footer = "Press Ctrl+C to stop the servers"
        print(f"\n{Colors.HEADER}{separator:^{self.terminal_width}}{Colors.ENDC}")
        print(f"{Colors.WARNING}{footer:^{self.terminal_width}}{Colors.ENDC}")
        
        sys.stdout.flush()
    
    def render_error_screen(self, step: str, error: str):
        """Render error screen with enhanced styling and smooth transitions"""
        if self.quiet:
            return
        
        # Smooth transition to error screen
        self.animation_state = "error"
        self.state_start_time = time.perf_counter()
        
        # Stop animation thread for error screen
        self.animation_running = False
        if self.animation_thread:
            self.animation_thread.join(timeout=1)
        
        # Use cursor positioning instead of screen clearing for smooth transition
        self.move_cursor(1, 1)
        
        # Calculate positions
        center_x = self.terminal_width // 2
        
        # Header with error styling
        header = "❌ Setup Failed"
        if not self.terminal_supports_unicode:
            header = "ERROR: Setup Failed"  # Fallback without emoji
        print(f"\n{Colors.FAIL}{header:^{self.terminal_width}}{Colors.ENDC}")
        
        # Error details
        print(f"\n{Colors.FAIL}{'Failed at:':^{self.terminal_width}}{Colors.ENDC}")
        print(f"{Colors.BOLD}{step:^{self.terminal_width}}{Colors.ENDC}")
        
        # Error message with formatting
        print(f"\n{Colors.FAIL}{error:^{self.terminal_width}}{Colors.ENDC}")
        
        # Instructions
        instructions = "Check the logs for more details"
        print(f"\n{Colors.WARNING}{instructions:^{self.terminal_width}}{Colors.ENDC}")
        
        sys.stdout.flush()
    
    def cleanup(self):
        """Cleanup terminal state and animation thread with proper state management"""
        # Transition to cleanup state
        self.animation_state = "idle"
        
        # Stop animation thread
        self.animation_running = False
        if self.animation_thread and self.animation_thread.is_alive():
            self.animation_thread.join(timeout=2)
        
        # Reset animation state
        self.animation_state = "idle"
        
        if not self.quiet:
            self.show_cursor()

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
        """Check if a port is in use with retry logic to avoid race conditions"""
        for attempt in range(3):  # Retry up to 3 times
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)  # Add timeout to prevent hanging
                    sock.bind(('', port))
                    return False  # Port is available
            except OSError:
                if attempt < 2:  # Not the last attempt
                    time.sleep(0.1)  # Brief delay before retry
                    continue
                return True  # Port is in use after all attempts
        return True  # Default to assuming port is in use if all attempts fail
    
    @staticmethod
    def find_free_port(start_port: int = 8000, max_attempts: int = 100) -> Optional[int]:
        """Find a free port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            if not PortManager.is_port_in_use(port):
                return port
        return None
    
    @staticmethod
    def kill_process_on_port(port: int) -> bool:
        """Kill any process using the specified port with platform-specific improvements"""
        try:
            if platform.system() == "Windows":
                return PortManager._kill_process_on_port_windows(port)
            else:
                return PortManager._kill_process_on_port_unix(port)
        except Exception as e:
            print(f"{Colors.WARNING}Error killing process on port {port}: {e}{Colors.ENDC}")
            return False
    
    @staticmethod
    def _kill_process_on_port_windows(port: int) -> bool:
        """Windows-specific port killing using netstat and taskkill"""
        try:
            # Use netstat to find process using the port
            result = subprocess.run(
                ['netstat', '-ano'], 
                capture_output=True, 
                text=True, 
                shell=True
            )
            
            if result.returncode != 0:
                return False
                
            # Parse netstat output to find PID using the port
            lines = result.stdout.split('\n')
            for line in lines:
                if f':{port} ' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        if pid.isdigit() and pid != '0':  # Skip System Idle Process
                            # Kill the process
                            kill_result = subprocess.run(
                                ['taskkill', '/PID', pid, '/F'],
                                capture_output=True,
                                text=True,
                                shell=True
                            )
                            if kill_result.returncode == 0:
                                print(f"{Colors.WARNING}Killed process {pid} on port {port}{Colors.ENDC}")
                                return True
            return False
        except Exception:
            return False
    
    @staticmethod
    def _kill_process_on_port_unix(port: int) -> bool:
        """Unix-specific port killing using psutil"""
        try:
            # Find processes using the port
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    # Get connections for this process
                    connections = proc.net_connections()
                    for conn in connections:
                        if hasattr(conn, 'laddr') and hasattr(conn.laddr, 'port') and conn.laddr.port == port:
                            print(f"{Colors.WARNING}Killing process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}{Colors.ENDC}")
                            proc.terminate()
                            proc.wait(timeout=5)
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired, psutil.ZombieProcess):
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
        """Monitor stderr for errors and info messages"""
        try:
            if self.process.stderr:
                for line in iter(self.process.stderr.readline, ''):
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        # Check if this is a DEBUG message from the backend
                        if '[DEBUG]' in line:
                            # Treat DEBUG messages as debug level (not errors)
                            self.logger.debug(f"[{self.name}] {line}")
                            # Don't print debug messages to console to reduce noise
                        elif line.startswith('INFO:'):
                            # Remove the INFO: prefix and treat as info message
                            info_message = line[5:].strip()
                            self.logger.info(f"[{self.name}] {info_message}")
                            print(f"{Colors.OKBLUE}[{self.name}] {info_message}{Colors.ENDC}")
                        elif line.startswith('WARNING:'):
                            # Remove the WARNING: prefix and treat as warning message
                            warning_message = line[8:].strip()
                            self.logger.warning(f"[{self.name}] {warning_message}")
                            print(f"{Colors.WARNING}[{self.name}] {warning_message}{Colors.ENDC}")
                        elif line.startswith('ERROR:'):
                            # Remove the ERROR: prefix and treat as error message
                            error_message = line[6:].strip()
                            self.error_queue.put(error_message)
                            self.logger.error(f"[{self.name}] {error_message}")
                            print(f"{Colors.FAIL}[{self.name}] {error_message}{Colors.ENDC}")
                        else:
                            # Default to treating as error if no log level prefix
                            self.error_queue.put(line)
                            self.logger.error(f"[{self.name}] {line}")
                            print(f"{Colors.FAIL}[{self.name}] {line}{Colors.ENDC}")
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
    
    def __init__(self, logger: Logger, quiet: bool = False):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.monitors: Dict[str, ProcessMonitor] = {}
        self.threads = []
        self.running = True
        self.logger = logger
        self.quiet = quiet
        
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
        """Enhanced cleanup with better process termination and duplicate prevention"""
        # Prevent double cleanup
        if hasattr(self, '_cleanup_called'):
            return
        self._cleanup_called = True
        
        self.running = False
        
        if not self.quiet:
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
                if not self.quiet:
                    print(f"{Colors.WARNING}Error terminating {name}: {e}{Colors.ENDC}")
        
        # Wait for threads
        if not self.quiet:
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
    def wait_for_backend(host: str, port: int, logger: Logger, timeout: int = 30, quiet: bool = False) -> bool:
        """Wait for backend to be ready with detailed error reporting"""
        url = f"http://{host}:{port}"
        start_time = time.time()
        
        if not quiet:
            print(f"{Colors.OKCYAN}Waiting for backend at {url}...{Colors.ENDC}")
        logger.info(f"Waiting for backend at {url}")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{url}/docs", timeout=2)
                if response.status_code == 200:
                    if not quiet:
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
            if not quiet:
                print(".", end="", flush=True)
        
        if not quiet:
            print(f"\n{Colors.FAIL}✗ Backend failed to start within {timeout} seconds{Colors.ENDC}")
        logger.error(f"Backend failed to start within {timeout} seconds")
        return False
    
    @staticmethod
    def wait_for_frontend(host: str, port: int, logger: Logger, timeout: int = 30, quiet: bool = False) -> bool:
        """Wait for frontend to be ready with detailed error reporting"""
        url = f"http://{host}:{port}"
        start_time = time.time()
        
        if not quiet:
            print(f"{Colors.OKCYAN}Waiting for frontend at {url}...{Colors.ENDC}")
        logger.info(f"Waiting for frontend at {url}")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    if not quiet:
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
            if not quiet:
                print(".", end="", flush=True)
        
        if not quiet:
            print(f"\n{Colors.FAIL}✗ Frontend failed to start within {timeout} seconds{Colors.ENDC}")
        logger.error(f"Frontend failed to start within {timeout} seconds")
        return False

class QRCodeGenerator:
    """Generates QR codes for mobile access with PWA installation support"""
    
    @staticmethod
    def generate_qr_code(url: str, output_path: Optional[str] = None, center: bool = False, term_width: int = 80) -> None:
        """Generate and display QR code for mobile access with improved terminal compatibility"""
        # Point directly to the main app
        app_url = url
        
        # Create QR code with appropriate size for terminal
        qr = qrcode.QRCode(
            version=1,
            error_correction=constants.ERROR_CORRECT_L,
            box_size=1,  # Small box size for terminal display
            border=1,    # Minimal border
        )
        qr.add_data(app_url)
        qr.make(fit=True)
        
        # Save image if path provided
        if output_path:
            try:
                img = qr.make_image(fill_color="black", back_color="white")
                with open(output_path, 'wb') as f:
                    img.save(f)
                print(f"{Colors.OKGREEN}QR code saved to: {output_path}{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.WARNING}Could not save QR code image: {e}{Colors.ENDC}")
        
        # Display QR code in terminal with proper formatting
        print(f"\n{Colors.HEADER}{'='*50}{Colors.ENDC}")
        print(f"{Colors.HEADER}📱 MOBILE ACCESS QR CODE{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
        print(f"{Colors.OKBLUE}URL: {app_url}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Scan this QR code to access the 2048 AI app{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
        
        # Get QR code matrix
        matrix = qr.get_matrix()
        
        # Calculate QR code width for centering
        qr_width = len(matrix[0]) * 2  # Each cell is 2 characters wide
        
        # Display QR code with proper centering and terminal compatibility
        for row in matrix:
            # Create line with appropriate characters
            line = ''.join(['██' if cell else '  ' for cell in row])
            
            # Center the line if requested
            if center and term_width > qr_width:
                padding = (term_width - qr_width) // 2
                line = ' ' * padding + line
            
            print(line)
        
        # Footer with instructions
        print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}📱 Scan this QR code to access the 2048 AI app on your device!{Colors.ENDC}")
        print(f"{Colors.WARNING}💡 iOS users: Tap the share button (📤) then Add to Home Screen{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")

# NOTE: Removed GUI-related classes `LoadingAnimation` and `QRCodeWindow` which contained severe syntax errors and were unused elsewhere in the launcher.
# The launcher continues to operate entirely via the console.

class Launcher:
    """Enhanced launcher with robust error handling and monitoring"""
    
    def __init__(self, dev_mode: bool = False, force_ports: bool = False,
                 lan_only: bool = False, tunnel_only: bool = False, tunnel_type: str = "quick",
                 tunnel_name: str = "2048-bot", tunnel_domain: Optional[str] = None,
                 no_tunnel_fallback: bool = False, backend_port: int = 8000,
                 frontend_port: int = 5173, host: str = "0.0.0.0", no_qr: bool = False,
                 no_color: bool = False, quiet: bool = False, skip_build: bool = False,
                 skip_deps: bool = False, cloudflared_path: Optional[str] = None,
                 timeout: int = 30):
        self.logger = Logger()
        self.process_manager = ProcessManager(self.logger, quiet=quiet)
        self.backend_port = backend_port
        self.frontend_port = frontend_port
        self.host_ip = None
        self.dev_mode = dev_mode
        self.force_ports = force_ports
        
        # Default to QR-only mode unless dev mode is explicitly requested
        if not dev_mode and not no_qr:
            self.qr_only = True
        else:
            self.qr_only = False
        
        # Initialize console UI
        self.console_ui = ConsoleUI(quiet=quiet, no_color=no_color)
        
        # New tunnel and configuration options
        self.lan_only = lan_only
        self.tunnel_only = tunnel_only
        self.tunnel_type = tunnel_type
        self.tunnel_name = tunnel_name
        self.tunnel_domain = tunnel_domain
        self.no_tunnel_fallback = no_tunnel_fallback
        self.host = host
        self.no_qr = no_qr
        self.no_color = no_color
        self.quiet = quiet
        self.skip_build = skip_build
        self.skip_deps = skip_deps
        self.cloudflared_path = cloudflared_path or self._find_cloudflared()
        self.timeout = timeout
        
        # Tunnel state
        self.tunnel_process = None
        self.tunnel_url = None
    
    def _find_cloudflared(self) -> Optional[str]:
        """Find cloudflared binary in current directory or PATH"""
        # Check current directory first
        local_cloudflared = Path("./cloudflared.exe" if platform.system() == "Windows" else "./cloudflared")
        if local_cloudflared.exists():
            return str(local_cloudflared)
        
        # Check PATH
        return shutil.which("cloudflared")
    
    def _start_tunnel(self) -> Optional[str]:
        """Start cloudflared tunnel and return the public URL"""
        if not self.cloudflared_path:
            if not self.quiet:
                print(f"{Colors.WARNING}⚠️  cloudflared not found - skipping tunnel creation{Colors.ENDC}")
            self.logger.warning("cloudflared not found, skipping tunnel")
            return None
        
        if not self.quiet:
            print(f"{Colors.OKCYAN}🌐 Starting Cloudflare Tunnel...{Colors.ENDC}")
        self.logger.info("Starting Cloudflare Tunnel")
        
        try:
            if self.tunnel_type == "named":
                return self._start_named_tunnel()
            else:
                return self._start_quick_tunnel()
        except Exception as e:
            if not self.quiet:
                print(f"{Colors.FAIL}❌ Failed to start tunnel: {e}{Colors.ENDC}")
            self.logger.error(f"Failed to start tunnel: {e}")
            return None
    
    def _start_quick_tunnel(self) -> Optional[str]:
        """Start a quick tunnel (temporary, no account required)"""
        cmd = [self.cloudflared_path, "tunnel", "--url", f"http://localhost:{self.backend_port}"]
        
        if not self.quiet:
            print(f"{Colors.OKCYAN}  Starting quick tunnel...{Colors.ENDC}")
        
        try:
            self.tunnel_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Wait for tunnel URL
            import re
            tunnel_url = None
            start_time = time.time()
            
            while time.time() - start_time < self.timeout:
                if not self.tunnel_process.stdout:
                    break
                line = self.tunnel_process.stdout.readline()
                if not line:
                    break
                
                if not self.quiet and self.dev_mode:
                    print(f"{Colors.OKCYAN}  [cloudflared] {line.strip()}{Colors.ENDC}")
                
                # Look for tunnel URL
                url_match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                if url_match:
                    tunnel_url = url_match.group(0)
                    break
            
            if tunnel_url:
                self.tunnel_url = tunnel_url
                if not self.quiet:
                    print(f"{Colors.OKGREEN}✅ Quick tunnel created: {tunnel_url}{Colors.ENDC}")
                self.logger.info(f"Quick tunnel created: {tunnel_url}")
                return tunnel_url
            else:
                if not self.quiet:
                    print(f"{Colors.FAIL}❌ Failed to get tunnel URL within {self.timeout}s{Colors.ENDC}")
                self.logger.error("Failed to get tunnel URL")
                return None
                
        except Exception as e:
            if not self.quiet:
                print(f"{Colors.FAIL}❌ Quick tunnel failed: {e}{Colors.ENDC}")
            self.logger.error(f"Quick tunnel failed: {e}")
            return None
    
    def _start_named_tunnel(self) -> Optional[str]:
        """Start a named tunnel (requires Cloudflare account setup)"""
        if not self.quiet:
            print(f"{Colors.OKCYAN}  Starting named tunnel: {self.tunnel_name}{Colors.ENDC}")
        
        # Check if tunnel exists and is configured
        config_path = Path.home() / ".cloudflared" / "config.yml"
        if not config_path.exists():
            if not self.quiet:
                print(f"{Colors.WARNING}⚠️  No cloudflared config found at {config_path}{Colors.ENDC}")
            
            if not self.no_tunnel_fallback:
                if not self.quiet:
                    print(f"{Colors.OKCYAN}  Falling back to quick tunnel...{Colors.ENDC}")
                return self._start_quick_tunnel()
            else:
                if not self.quiet:
                    print(f"{Colors.FAIL}❌ Named tunnel not configured and fallback disabled{Colors.ENDC}")
                return None
        
        # Build tunnel URL
        if self.tunnel_domain:
            tunnel_url = f"https://{self.tunnel_domain}"
        else:
            tunnel_url = f"https://{self.tunnel_name}.cfargotunnel.com"
        
        try:
            cmd = [self.cloudflared_path, "tunnel", "--config", str(config_path), "run", self.tunnel_name]
            
            self.tunnel_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Wait for tunnel to establish
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                if not self.tunnel_process.stdout:
                    break
                line = self.tunnel_process.stdout.readline()
                if not line:
                    break
                
                if not self.quiet and self.dev_mode:
                    print(f"{Colors.OKCYAN}  [cloudflared] {line.strip()}{Colors.ENDC}")
                
                # Check for successful connection
                if "Connection established" in line or "Registered tunnel connection" in line:
                    self.tunnel_url = tunnel_url
                    if not self.quiet:
                        print(f"{Colors.OKGREEN}✅ Named tunnel running: {tunnel_url}{Colors.ENDC}")
                    self.logger.info(f"Named tunnel running: {tunnel_url}")
                    return tunnel_url
            
            # If we get here, tunnel didn't establish in time
            if not self.quiet:
                print(f"{Colors.FAIL}❌ Named tunnel failed to establish within {self.timeout}s{Colors.ENDC}")
            
            if not self.no_tunnel_fallback:
                if not self.quiet:
                    print(f"{Colors.OKCYAN}  Falling back to quick tunnel...{Colors.ENDC}")
                self._stop_tunnel()
                return self._start_quick_tunnel()
            else:
                return None
                
        except Exception as e:
            if not self.quiet:
                print(f"{Colors.FAIL}❌ Named tunnel failed: {e}{Colors.ENDC}")
            self.logger.error(f"Named tunnel failed: {e}")
            
            if not self.no_tunnel_fallback:
                if not self.quiet:
                    print(f"{Colors.OKCYAN}  Falling back to quick tunnel...{Colors.ENDC}")
                return self._start_quick_tunnel()
            else:
                return None
    
    def _stop_tunnel(self):
        """Stop the tunnel process"""
        if self.tunnel_process:
            try:
                self.tunnel_process.terminate()
                self.tunnel_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.tunnel_process.kill()
            except Exception as e:
                self.logger.error(f"Error stopping tunnel: {e}")
            finally:
                self.tunnel_process = None
                self.tunnel_url = None
        
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        self.logger.info("Checking dependencies")
        
        # Update progress
        self.console_ui.update_progress("Checking dependencies", 0.1, "Verifying project structure...")
        self.console_ui.update_micro_progress("Checking project directories", 0.0)
        
        # Check if we're in the right directory
        if not os.path.exists("backend") or not os.path.exists("frontend"):
            error_msg = "Error: Please run this script from the project root directory"
            self.logger.error(error_msg)
            return False
        
        self.console_ui.update_micro_progress("Project structure verified", 0.3)
        
        # Check Poetry
        try:
            subprocess.run(["poetry", "--version"], check=True, capture_output=True, shell=True)
            self.console_ui.update_micro_progress("Poetry found", 0.5)
        except (subprocess.CalledProcessError, FileNotFoundError):
            error_msg = "Poetry not found. Please install Poetry first."
            self.logger.error(error_msg)
            return False
        
        # Check Node.js
        try:
            subprocess.run(["node", "--version"], check=True, capture_output=True, shell=True)
            self.console_ui.update_micro_progress("Node.js found", 0.7)
        except (subprocess.CalledProcessError, FileNotFoundError):
            error_msg = "Node.js not found. Please install Node.js first."
            self.logger.error(error_msg)
            return False
        
        # Check npm
        try:
            subprocess.run(["npm", "--version"], check=True, capture_output=True, shell=True)
            self.console_ui.update_micro_progress("npm found", 0.9)
        except (subprocess.CalledProcessError, FileNotFoundError):
            error_msg = "npm not found. Please install npm first."
            self.logger.error(error_msg)
            return False
        
        self.console_ui.update_micro_progress("All dependencies verified", 1.0)
        self.logger.info("All dependencies found")
        return True
    
    def setup_network(self) -> bool:
        """Setup network configuration with port management"""
        self.logger.info("Setting up network configuration")
        
        # Update progress
        self.console_ui.update_progress("Setting up network", 0.2, "Discovering network interfaces...")
        self.console_ui.update_micro_progress("Scanning network adapters", 0.0)
        
        # Find the best IP address
        self.host_ip = NetworkDiscovery.find_best_ip()
        if not self.host_ip:
            error_msg = "Could not determine local IP address"
            self.logger.error(error_msg)
            return False
        
        self.console_ui.update_micro_progress(f"Found IP: {self.host_ip}", 0.5)
        self.logger.info(f"Using IP address: {self.host_ip}")
        
        # Check and manage ports
        self.console_ui.update_micro_progress("Checking port availability", 0.7)
        if not self._setup_ports():
            return False
        
        self.console_ui.update_micro_progress("Network configuration ready", 1.0)
        return True
    
    def _setup_ports(self) -> bool:
        """Setup and verify port availability with enhanced error handling"""
        ports_to_check = [
            (self.backend_port, "Backend"),
            (self.frontend_port, "Frontend"),
            (self.frontend_port + 1, "HMR")  # Hot Module Replacement
        ]
        
        for port, name in ports_to_check:
            # Add detailed logging
            self.logger.debug(f"Checking port {port} ({name})")
            
            if PortManager.is_port_in_use(port):
                self.logger.warning(f"Port {port} ({name}) appears to be in use")
                
                if self.force_ports:
                    self.logger.info(f"Attempting to free port {port} ({name})")
                    if PortManager.kill_process_on_port(port):
                        self.logger.info(f"Successfully freed port {port}")
                        # Verify port is now free
                        time.sleep(0.5)  # Give OS time to release port
                        if not PortManager.is_port_in_use(port):
                            self.logger.info(f"Port {port} confirmed free")
                            continue
                        else:
                            self.logger.error(f"Port {port} still appears in use after killing process")
                            if not self.quiet:
                                print(f"{Colors.FAIL}Port {port} ({name}) still in use after cleanup attempt{Colors.ENDC}")
                            return False
                    else:
                        self.logger.error(f"Could not free port {port} ({name})")
                        if not self.quiet:
                            print(f"{Colors.FAIL}Could not free port {port} ({name}){Colors.ENDC}")
                        return False
                else:
                    error_msg = f"Port {port} ({name}) is in use. Use --force-ports to attempt to free it."
                    self.logger.error(error_msg)
                    if not self.quiet:
                        print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
                    return False
        
        self.logger.info("All ports are available")
        return True
    
    def install_dependencies(self) -> bool:
        """Install project dependencies with enhanced progress tracking"""
        self.logger.info("Installing dependencies")
        
        # Update main progress
        self.console_ui.update_progress("Installing dependencies", 0.3, "Checking existing installations...")
        
        # Check if dependencies are already installed
        backend_installed = self._check_backend_dependencies()
        frontend_installed = self._check_frontend_dependencies()
        
        if backend_installed and frontend_installed:
            self.console_ui.update_progress("Installing dependencies", 1.0, "✓ Dependencies already installed")
            self.console_ui.update_micro_progress("All dependencies ready", 1.0)
            self.logger.info("Dependencies already installed, skipping installation")
            return True
        
        # Install backend dependencies if needed
        if not backend_installed:
            self.console_ui.update_progress("Installing dependencies", 0.4, "Installing backend dependencies...")
            
            try:
                # Start backend installation
                self.console_ui.update_micro_progress("Installing Python packages", 0.0)
                
                backend_process = subprocess.Popen(
                    ["poetry", "install"], 
                    cwd="backend", 
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Monitor backend installation progress
                line_count = 0
                while backend_process.poll() is None:
                    if backend_process.stdout is None:
                        break
                    line = backend_process.stdout.readline()
                    if not line:
                        break
                    
                    line_count += 1
                    # Update micro-progress based on line count (rough estimate)
                    micro_progress = min(0.9, line_count / 50)
                    self.console_ui.update_micro_progress("Installing Python packages", micro_progress)
                    
                    # Log to file but don't spam console
                    self.logger.debug(f"[Backend Install] {line.strip()}")
                
                # Wait for process to complete and get return code
                backend_process.wait()
                
                # Check result
                if backend_process.returncode != 0:
                    # Capture any remaining output for debugging
                    remaining_output = ""
                    if backend_process.stdout:
                        remaining_output = backend_process.stdout.read()
                        if remaining_output:
                            self.logger.error(f"[Backend Install] Remaining output: {remaining_output}")
                    
                    error_msg = f"Failed to install backend dependencies (exit code: {backend_process.returncode})"
                    self.logger.error(error_msg)
                    return False
                
                self.logger.info("Backend dependencies installed successfully")
                self.console_ui.update_micro_progress("Backend packages installed", 1.0)
                
            except Exception as e:
                error_msg = f"Failed to install backend dependencies: {e}"
                self.logger.error(error_msg)
                return False
        else:
            self.console_ui.update_micro_progress("Backend dependencies already installed", 1.0)
        
        # Install frontend dependencies if needed
        if not frontend_installed:
            self.console_ui.update_progress("Installing dependencies", 0.7, "Installing frontend dependencies...")
            
            try:
                # Start frontend installation
                self.console_ui.update_micro_progress("Installing Node.js packages", 0.0)
                
                frontend_process = subprocess.Popen(
                    ["npm", "install"], 
                    cwd="frontend", 
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Monitor frontend installation progress
                line_count = 0
                while frontend_process.poll() is None:
                    if frontend_process.stdout is None:
                        break
                    line = frontend_process.stdout.readline()
                    if not line:
                        break
                    
                    line_count += 1
                    # Update micro-progress based on line count (rough estimate)
                    micro_progress = min(0.9, line_count / 100)
                    self.console_ui.update_micro_progress("Installing Node.js packages", micro_progress)
                    
                    # Log to file but don't spam console
                    self.logger.debug(f"[Frontend Install] {line.strip()}")
                
                # Wait for process to complete and get return code
                frontend_process.wait()
                
                # Check result
                if frontend_process.returncode != 0:
                    # Capture any remaining output for debugging
                    remaining_output = ""
                    if frontend_process.stdout:
                        remaining_output = frontend_process.stdout.read()
                        if remaining_output:
                            self.logger.error(f"[Frontend Install] Remaining output: {remaining_output}")
                    
                    error_msg = f"Failed to install frontend dependencies (exit code: {frontend_process.returncode})"
                    self.logger.error(error_msg)
                    return False
                
                self.logger.info("Frontend dependencies installed successfully")
                self.console_ui.update_micro_progress("Frontend packages installed", 1.0)
                
            except Exception as e:
                error_msg = f"Failed to install frontend dependencies: {e}"
                self.logger.error(error_msg)
                return False
        else:
            self.console_ui.update_micro_progress("Frontend dependencies already installed", 1.0)
        
        # Clear micro-progress
        self.console_ui.update_micro_progress("", 0.0)
        
        return True
    
    def _check_backend_dependencies(self) -> bool:
        """Check if backend dependencies are already installed"""
        try:
            # Check if poetry.lock exists and is recent
            lock_file = Path("backend/poetry.lock")
            if not lock_file.exists():
                return False
            
            # Check if virtual environment exists
            venv_path = Path("backend/.venv")
            if not venv_path.exists():
                return False
            
            # Quick check if key packages are available
            result = subprocess.run(
                ["poetry", "run", "python", "-c", "import fastapi, uvicorn, torch"],
                cwd="backend",
                shell=True,
                capture_output=True,
                text=True
            )
            
            return result.returncode == 0
        except Exception:
            return False
    
    def _check_frontend_dependencies(self) -> bool:
        """Check if frontend dependencies are already installed"""
        try:
            # Check if node_modules exists
            node_modules = Path("frontend/node_modules")
            if not node_modules.exists():
                return False
            
            # Check if package-lock.json exists
            lock_file = Path("frontend/package-lock.json")
            if not lock_file.exists():
                return False
            
            # Quick check if key packages are available
            result = subprocess.run(
                ["npm", "list", "react", "vite"],
                cwd="frontend",
                shell=True,
                capture_output=True,
                text=True
            )
            
            return result.returncode == 0
        except Exception:
            return False
    
    def start_backend(self) -> bool:
        """Start the backend server with enhanced progress tracking"""
        self.logger.info("Starting backend server")
        
        # Update progress
        self.console_ui.update_progress("Starting backend", 0.7, "Initializing FastAPI server...")
        self.console_ui.update_micro_progress("Setting up environment", 0.0)
        
        # Set up environment variables for CORS
        backend_env = os.environ.copy()
        cors_origins = [
            f"http://localhost:{self.frontend_port}",
            f"http://127.0.0.1:{self.frontend_port}",
            f"http://{self.host_ip}:{self.frontend_port}"
        ]
        
        # Always include tunnel wildcard domains for tunnel support
        if not self.lan_only:
            cors_origins.extend([
                "https://*.trycloudflare.com",
                "https://*.cfargotunnel.com"
            ])
        
        backend_env["CORS_ORIGINS"] = ",".join(cors_origins)
        self.console_ui.update_micro_progress("Environment configured", 0.3)
        
        # Backend command
        backend_cmd = [
            "poetry", "run", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", str(self.backend_port),
            "--reload"
        ]
        
        self.console_ui.update_micro_progress("Starting uvicorn server", 0.6)
        
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
            self.console_ui.update_micro_progress("Server process started", 0.8)
            
            # Wait for backend to be ready with progress updates
            if self.host_ip:
                self.console_ui.update_micro_progress("Waiting for server to respond", 0.9)
                
                # Enhanced health check with progress
                if not self._wait_for_backend_with_progress(self.host_ip, self.backend_port):
                    # Get recent errors for debugging
                    status = self.process_manager.get_process_status()
                    backend_status = status.get("Backend", {})
                    recent_errors = backend_status.get("recent_errors", [])
                    
                    if recent_errors:
                        print(f"{Colors.FAIL}Recent backend errors:{Colors.ENDC}")
                        for error in recent_errors[-3:]:  # Show last 3 errors
                            print(f"{Colors.FAIL}  {error}{Colors.ENDC}")
                    
                    return False
            
            self.console_ui.update_micro_progress("Backend server ready", 1.0)
            return True
            
        except Exception as e:
            error_msg = f"Failed to start backend: {e}"
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
    
    def _wait_for_backend_with_progress(self, host: str, port: int) -> bool:
        """Wait for backend with enhanced progress feedback"""
        url = f"http://{host}:{port}"
        start_time = time.time()
        timeout = 30
        
        if not self.quiet:
            print(f"{Colors.OKCYAN}Waiting for backend at {url}...{Colors.ENDC}")
        self.logger.info(f"Waiting for backend at {url}")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{url}/docs", timeout=2)
                if response.status_code == 200:
                    if not self.quiet:
                        print(f"{Colors.OKGREEN}✓ Backend is ready!{Colors.ENDC}")
                    self.logger.info("Backend is ready")
                    return True
            except requests.ConnectionError:
                pass
            except requests.Timeout:
                pass
            except Exception as e:
                self.logger.debug(f"Backend health check error: {e}")
            
            # Update micro-progress based on elapsed time
            elapsed = time.time() - start_time
            progress = min(0.95, elapsed / timeout)
            self.console_ui.update_micro_progress(f"Connecting to backend ({int(elapsed)}s)", progress)
            
            time.sleep(1)
            if not self.quiet:
                print(".", end="", flush=True)
        
        if not self.quiet:
            print(f"\n{Colors.FAIL}✗ Backend failed to start within {timeout} seconds{Colors.ENDC}")
        self.logger.error(f"Backend failed to start within {timeout} seconds")
        return False
    
    def start_frontend(self) -> bool:
        """Start the frontend server with enhanced progress tracking"""
        self.logger.info("Starting frontend server")
        
        # Update progress
        self.console_ui.update_progress("Starting frontend", 0.8, "Configuring Vite development server...")
        self.console_ui.update_micro_progress("Creating Vite configuration", 0.0)

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
                    name: '2048 Bot Training',
            short_name: '2048 AI',
            description: 'Real-time visualization for 2048 bot training',
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
      origin: ['http://{self.host_ip}:{self.frontend_port}', 'http://localhost:{self.frontend_port}', 'https://*.trycloudflare.com', 'https://*.cfargotunnel.com'],
      credentials: true
    }}
  }},
  define: {{
    global: 'globalThis',
    __BACKEND_URL__: JSON.stringify('{self.tunnel_url if self.tunnel_url else f"http://{self.host_ip}:{self.backend_port}"}')
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
            self.console_ui.update_micro_progress("Vite config created", 0.2)
        except Exception as e:
            error_msg = f"Failed to create Vite config: {e}"
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
        
        # Create a script to inject the backend URL
        backend_url_script = f"""
// Backend URL configuration
window.__BACKEND_URL__ = '{self.tunnel_url if self.tunnel_url else f"http://{self.host_ip}:{self.backend_port}"}';
console.log('Backend URL set to:', window.__BACKEND_URL__);
"""
        
        script_path = "frontend/public/backend-config.js"
        try:
            with open(script_path, "w") as f:
                f.write(backend_url_script)
            self.logger.info("Created backend config script")
            self.console_ui.update_micro_progress("Backend config created", 0.4)
        except Exception as e:
            error_msg = f"Failed to create backend config script: {e}"
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
        
        # After the build is complete, inject the backend URL into the built HTML
        dist_html_path = "frontend/dist/index.html"
        
        # Read the built HTML file
        try:
            with open(dist_html_path, "r") as f:
                html_content = f.read()
            self.console_ui.update_micro_progress("HTML template loaded", 0.6)
        except FileNotFoundError:
            error_msg = "Built HTML file not found. Make sure the frontend build completed successfully."
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
        
        # Inject the backend configuration script before the closing head tag
        backend_config_script = f"""
    <!-- Backend configuration script -->
    <script>
      // Backend URL configuration
      window.__BACKEND_URL__ = '{self.tunnel_url if self.tunnel_url else f"http://{self.host_ip}:{self.backend_port}"}';
      console.log('Backend URL set to:', window.__BACKEND_URL__);
    </script>
  </head>"""
        
        # Replace the closing head tag with our script + closing head tag
        html_content = html_content.replace("</head>", backend_config_script)
        
        # Write the modified HTML back to the dist folder
        try:
            with open(dist_html_path, "w") as f:
                f.write(html_content)
            self.logger.info("Updated built HTML with backend config script")
            self.console_ui.update_micro_progress("HTML updated with backend config", 0.8)
        except Exception as e:
            error_msg = f"Failed to update built HTML: {e}"
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
        
        if self.dev_mode:
            self.console_ui.update_micro_progress("Starting development server", 0.9)
            frontend_cmd = [
                "npm", "run", "dev", "--", 
                "--config", "vite.config.temp.ts",
                "--host", "0.0.0.0",
                "--port", str(self.frontend_port),
                "--force"
            ]
        else:
            # Build production bundle first
            self.console_ui.update_micro_progress("Building production bundle", 0.85)
            if not self.quiet:
                print(f"{Colors.OKCYAN}Building production bundle...{Colors.ENDC}")
            
            try:
                # Enhanced build process with progress monitoring
                build_process = subprocess.Popen([
                    "npm", "run", "build", "--", "--config", "vite.config.temp.ts"
                ], cwd="frontend", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                
                # Monitor build progress
                line_count = 0
                while build_process.poll() is None:
                    if build_process.stdout is None:
                        break
                    line = build_process.stdout.readline()
                    if not line:
                        break
                    
                    line_count += 1
                    # Update micro-progress based on build output
                    if "Building" in line:
                        self.console_ui.update_micro_progress("Compiling TypeScript", 0.87)
                    elif "chunks" in line:
                        self.console_ui.update_micro_progress("Bundling assets", 0.89)
                    elif "dist" in line:
                        self.console_ui.update_micro_progress("Writing output files", 0.91)
                    elif "built" in line:
                        self.console_ui.update_micro_progress("Build completed", 0.95)
                    
                    # Log to file but don't spam console
                    self.logger.debug(f"[Frontend Build] {line.strip()}")
                
                # Wait for the process to complete and get the return code
                build_process.wait()
                
                if build_process.returncode != 0:
                    error_msg = f"Production build failed (exit code: {build_process.returncode})"
                    if not self.quiet:
                        print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
                    self.logger.error(error_msg)
                    return False
                
                if not self.quiet:
                    print(f"{Colors.OKGREEN}✓ Production build completed{Colors.ENDC}")
                self.logger.info("Production build completed successfully")
                self.console_ui.update_micro_progress("Production build ready", 0.95)
                
            except Exception as e:
                error_msg = f"Production build failed: {e}"
                if not self.quiet:
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
            self.console_ui.update_micro_progress("Starting frontend server", 0.97)
            frontend_process = subprocess.Popen(
                frontend_cmd,
                cwd="frontend",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True
            )
            
            self.process_manager.add_process("Frontend", frontend_process)
            self.console_ui.update_micro_progress("Frontend process started", 0.98)
            
            # Wait for frontend to be ready with progress updates
            if self.host_ip:
                self.console_ui.update_micro_progress("Waiting for frontend to respond", 0.99)
                
                # Enhanced health check with progress
                if not self._wait_for_frontend_with_progress(self.host_ip, self.frontend_port):
                    # Get recent errors for debugging
                    status = self.process_manager.get_process_status()
                    frontend_status = status.get("Frontend", {})
                    recent_errors = frontend_status.get("recent_errors", [])
                    
                    if recent_errors:
                        print(f"{Colors.FAIL}Recent frontend errors:{Colors.ENDC}")
                        for error in recent_errors[-3:]:  # Show last 3 errors
                            print(f"{Colors.FAIL}  {error}{Colors.ENDC}")
                    
                    return False
            
            self.console_ui.update_micro_progress("Frontend server ready", 1.0)
            return True
            
        except Exception as e:
            error_msg = f"Failed to start frontend: {e}"
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
    
    def _wait_for_frontend_with_progress(self, host: str, port: int) -> bool:
        """Wait for frontend with enhanced progress feedback"""
        url = f"http://{host}:{port}"
        start_time = time.time()
        timeout = 30
        
        if not self.quiet:
            print(f"{Colors.OKCYAN}Waiting for frontend at {url}...{Colors.ENDC}")
        self.logger.info(f"Waiting for frontend at {url}")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    if not self.quiet:
                        print(f"{Colors.OKGREEN}✓ Frontend is ready!{Colors.ENDC}")
                    self.logger.info("Frontend is ready")
                    return True
            except requests.ConnectionError:
                pass
            except requests.Timeout:
                pass
            except Exception as e:
                self.logger.debug(f"Frontend health check error: {e}")
            
            # Update micro-progress based on elapsed time
            elapsed = time.time() - start_time
            progress = min(0.99, 0.99 + (elapsed / timeout) * 0.01)
            self.console_ui.update_micro_progress(f"Connecting to frontend ({int(elapsed)}s)", progress)
            
            time.sleep(1)
            if not self.quiet:
                print(".", end="", flush=True)
        
        if not self.quiet:
            print(f"\n{Colors.FAIL}✗ Frontend failed to start within {timeout} seconds{Colors.ENDC}")
        self.logger.error(f"Frontend failed to start within {timeout} seconds")
        return False
    
    def show_access_info(self):
        """Display access information and QR code"""
        if not self.quiet:
            print(f"\n{Colors.HEADER}🚀 2048 Bot Training Server Started!{Colors.ENDC}")
            print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
        
        # Determine URLs to display
        urls_to_show = []
        primary_url = None
        
        if not self.tunnel_only:
            # Show LAN URLs
            frontend_url = f"http://{self.host_ip}:{self.frontend_port}"
            backend_url = f"http://{self.host_ip}:{self.backend_port}"
            
            if not self.quiet:
                print(f"{Colors.OKGREEN}🏠 Local Network Access:{Colors.ENDC}")
                print(f"   Frontend: {frontend_url}")
                print(f"   Backend API: {backend_url}")
                print(f"   Backend Docs: {backend_url}/docs")
            
            urls_to_show.append(("LAN", frontend_url))
            if not primary_url:
                primary_url = frontend_url
            
            self.logger.info(f"LAN access - Frontend: {frontend_url}, Backend: {backend_url}")
        
        if self.tunnel_url:
            # Show tunnel URLs
            if not self.quiet:
                print(f"{Colors.OKGREEN}🌐 Public Tunnel Access:{Colors.ENDC}")
                print(f"   Frontend: {self.tunnel_url}")
                print(f"   Backend API: {self.tunnel_url}")
                print(f"   Backend Docs: {self.tunnel_url}/docs")
            
            urls_to_show.append(("Tunnel", self.tunnel_url))
            # Tunnel takes priority as primary URL for QR code
            primary_url = self.tunnel_url
            
            self.logger.info(f"Tunnel access: {self.tunnel_url}")
        
        # Generate QR code for primary URL
        if primary_url and not self.no_qr:
            if not self.quiet:
                print(f"\n{Colors.OKCYAN}📱 QR Code for Mobile Access:{Colors.ENDC}")
            
            QRCodeGenerator.generate_qr_code(primary_url, "mobile_access_qr.png")
            
            if not self.quiet:
                print(f"{Colors.OKCYAN}Scan the QR code above with your phone to access the app!{Colors.ENDC}")
        
        # Show usage instructions
        if not self.quiet:
            print(f"\n{Colors.HEADER}📋 Usage Instructions:{Colors.ENDC}")
            if self.tunnel_url:
                print(f"🌐 Remote access: Share the tunnel URL with anyone")
                print(f"📱 Mobile PWA: Install from tunnel URL for offline access")
            if not self.tunnel_only:
                print(f"🏠 Local access: Use LAN URL for faster local development")
            print(f"🎮 Start training: Click 'Start Training' in the web interface")
            print(f"\n{Colors.WARNING}Press Ctrl+C to stop the servers{Colors.ENDC}")
        
        # Log summary
        log_msg = "Servers started successfully"
        if urls_to_show:
            url_summary = ", ".join([f"{name}: {url}" for name, url in urls_to_show])
            log_msg += f" - {url_summary}"
        self.logger.info(log_msg)
    
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
        if self.qr_only:
            # Non-blocking launcher with background operations
            self._run_non_blocking()
        else:
            print(f"{Colors.HEADER}🚀 2048 Bot Training Launcher{Colors.ENDC}")
            print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
            self.logger.info("Launcher started")
            try:
                if not self.skip_deps and not self.check_dependencies():
                    return False
                if not self.setup_network():
                    return False
                if not self.skip_deps and not self.install_dependencies():
                    return False
                if not self.start_backend():
                    return False
                
                # Start tunnel if needed
                if not self.lan_only:
                    self.tunnel_url = self._start_tunnel()
                
                if not self.start_frontend():
                    return False
                self.show_access_info()
                try:
                    while self.process_manager.running:
                        time.sleep(1)
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
                self.process_manager.cleanup()
                self._stop_tunnel()
                self.console_ui.cleanup()
                temp_config = "frontend/vite.config.temp.ts"
                if os.path.exists(temp_config):
                    try:
                        os.remove(temp_config)
                        self.logger.info("Removed temporary Vite config")
                    except Exception as e:
                        self.logger.warning(f"Could not remove temporary config: {e}")
                if not self.qr_only:
                    print(f"{Colors.OKGREEN}✓ Cleanup completed{Colors.ENDC}")
                self.logger.info("Launcher cleanup completed")
    
    def _run_non_blocking(self):
        """Run launcher with non-blocking operations for smooth animations"""
        steps = [
            'Checking dependencies',
            'Setting up network', 
            'Installing dependencies',
            'Starting backend',
            'Starting frontend'
        ]
        
        # Progress tracking
        current_step = 0
        total_steps = len(steps)
        
        # Background operation state
        self.background_operations = {
            'current_step': 0,
            'status': 'idle',  # idle, running, completed, failed
            'progress': 0.0,
            'error': None,
            'result': None
        }
        
        try:
            # Start background operations thread
            background_thread = threading.Thread(target=self._run_background_operations, daemon=True)
            background_thread.start()
            
            # Main animation loop
            while self.background_operations['status'] != 'completed':
                # Update progress based on background operations
                step = steps[self.background_operations['current_step']]
                progress = (self.background_operations['current_step'] + self.background_operations['progress']) / total_steps
                
                if self.background_operations['status'] == 'failed':
                    self.console_ui.render_error_screen(step, str(self.background_operations['error']))
                    return False
                
                # Update UI with current progress
                self.console_ui.update_progress(step, progress, "Processing...")
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.016)  # ~60fps
            
            # Final QR code screen
            if self.tunnel_url:
                frontend_url = self.tunnel_url
                backend_url = self.tunnel_url
            else:
                frontend_url = f"http://{self.host_ip}:{self.frontend_port}"
                backend_url = f"http://{self.host_ip}:{self.backend_port}"
            
            self.console_ui.render_qr_screen(frontend_url, backend_url)
            
            # Keep the servers running until user interrupts
            try:
                while self.process_manager.running:
                    time.sleep(1)
                    if int(time.time()) % 30 == 0:
                        self.show_status()
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}Shutting down...{Colors.ENDC}")
                self.logger.info("Shutdown requested by user")
            
            return True
            
        except KeyboardInterrupt:
            self.console_ui.render_error_screen(steps[current_step], "Setup interrupted by user")
            return False
        except Exception as e:
            self.console_ui.render_error_screen(steps[current_step], str(e))
            return False
        finally:
            self.process_manager.cleanup()
            self._stop_tunnel()
            self.console_ui.cleanup()
            temp_config = "frontend/vite.config.temp.ts"
            if os.path.exists(temp_config):
                try:
                    os.remove(temp_config)
                    self.logger.info("Removed temporary Vite config")
                except Exception as e:
                    self.logger.warning(f"Could not remove temporary config: {e}")
            self.logger.info("Launcher cleanup completed")
    
    def _run_background_operations(self):
        """Run all launcher operations in background thread"""
        steps = [
            'Checking dependencies',
            'Setting up network', 
            'Installing dependencies',
            'Starting backend',
            'Starting frontend'
        ]
        
        try:
            for idx, step in enumerate(steps):
                self.background_operations['current_step'] = idx
                self.background_operations['status'] = 'running'
                self.background_operations['progress'] = 0.0
                self.background_operations['error'] = None
                
                success = False
                
                if step == 'Checking dependencies':
                    if self.skip_deps:
                        success = True
                        self.background_operations['progress'] = 1.0
                    else:
                        success = self._check_dependencies_async()
                elif step == 'Setting up network':
                    success = self._setup_network_async()
                elif step == 'Installing dependencies':
                    if self.skip_deps:
                        success = True
                        self.background_operations['progress'] = 1.0
                    else:
                        success = self._install_dependencies_async()
                elif step == 'Starting backend':
                    success = self._start_backend_async()
                    # Start tunnel if needed
                    if success and not self.lan_only:
                        self.background_operations['progress'] = 0.5
                        self.tunnel_url = self._start_tunnel()
                        self.background_operations['progress'] = 1.0
                elif step == 'Starting frontend':
                    success = self._start_frontend_async()
                
                if success:
                    self.background_operations['progress'] = 1.0
                    self.background_operations['status'] = 'completed' if idx == len(steps) - 1 else 'running'
                else:
                    self.background_operations['status'] = 'failed'
                    self.background_operations['error'] = f"{step} failed"
                    break
                
                # Small delay between steps to allow UI updates
                time.sleep(0.1)
                
        except Exception as e:
            self.background_operations['status'] = 'failed'
            self.background_operations['error'] = str(e)
    
    def _check_dependencies_async(self) -> bool:
        """Check dependencies with progress updates"""
        try:
            self.background_operations['progress'] = 0.2
            if not self.check_dependencies():
                return False
            self.background_operations['progress'] = 1.0
            return True
        except Exception as e:
            self.logger.error(f"Async dependency check failed: {e}")
            return False
    
    def _setup_network_async(self) -> bool:
        """Setup network with progress updates"""
        try:
            self.background_operations['progress'] = 0.3
            if not self.setup_network():
                return False
            self.background_operations['progress'] = 1.0
            return True
        except Exception as e:
            self.logger.error(f"Async network setup failed: {e}")
            return False
    
    def _install_dependencies_async(self) -> bool:
        """Install dependencies with progress updates"""
        try:
            # Check if dependencies are already installed
            backend_installed = self._check_backend_dependencies()
            frontend_installed = self._check_frontend_dependencies()
            
            if backend_installed and frontend_installed:
                self.background_operations['progress'] = 1.0
                return True
            
            # Install backend dependencies if needed
            if not backend_installed:
                self.background_operations['progress'] = 0.4
                if not self._install_backend_deps_async():
                    return False
            
            # Install frontend dependencies if needed
            if not frontend_installed:
                self.background_operations['progress'] = 0.7
                if not self._install_frontend_deps_async():
                    return False
            
            self.background_operations['progress'] = 1.0
            return True
            
        except Exception as e:
            self.logger.error(f"Async dependency installation failed: {e}")
            return False
    
    def _install_backend_deps_async(self) -> bool:
        """Install backend dependencies with progress updates"""
        try:
            backend_process = subprocess.Popen(
                ["poetry", "install"], 
                cwd="backend", 
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Monitor progress
            line_count = 0
            while backend_process.poll() is None:
                if backend_process.stdout is None:
                    break
                line = backend_process.stdout.readline()
                if not line:
                    break
                
                line_count += 1
                # Update progress based on line count
                progress = min(0.9, 0.4 + (line_count / 50) * 0.3)
                self.background_operations['progress'] = progress
                
                # Log to file
                self.logger.debug(f"[Backend Install] {line.strip()}")
            
            # Wait for completion
            backend_process.wait()
            
            if backend_process.returncode != 0:
                self.logger.error(f"Backend installation failed: {backend_process.returncode}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Backend installation failed: {e}")
            return False
    
    def _install_frontend_deps_async(self) -> bool:
        """Install frontend dependencies with progress updates"""
        try:
            frontend_process = subprocess.Popen(
                ["npm", "install"], 
                cwd="frontend", 
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Monitor progress
            line_count = 0
            while frontend_process.poll() is None:
                if frontend_process.stdout is None:
                    break
                line = frontend_process.stdout.readline()
                if not line:
                    break
                
                line_count += 1
                # Update progress based on line count
                progress = min(0.9, 0.7 + (line_count / 100) * 0.2)
                self.background_operations['progress'] = progress
                
                # Log to file
                self.logger.debug(f"[Frontend Install] {line.strip()}")
            
            # Wait for completion
            frontend_process.wait()
            
            if frontend_process.returncode != 0:
                self.logger.error(f"Frontend installation failed: {frontend_process.returncode}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Frontend installation failed: {e}")
            return False
    
    def _start_backend_async(self) -> bool:
        """Start backend with progress updates"""
        try:
            self.background_operations['progress'] = 0.3
            
            # Set up environment variables for CORS
            backend_env = os.environ.copy()
            cors_origins = [
                f"http://localhost:{self.frontend_port}",
                f"http://127.0.0.1:{self.frontend_port}",
                f"http://{self.host_ip}:{self.frontend_port}"
            ]
            
            if not self.lan_only:
                cors_origins.extend([
                    "https://*.trycloudflare.com",
                    "https://*.cfargotunnel.com"
                ])
            
            backend_env["CORS_ORIGINS"] = ",".join(cors_origins)
            self.background_operations['progress'] = 0.5
            
            # Start backend
            backend_cmd = [
                "poetry", "run", "uvicorn", "main:app",
                "--host", "0.0.0.0",
                "--port", str(self.backend_port),
                "--reload"
            ]
            
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
            self.background_operations['progress'] = 0.7
            
            # Wait for backend to be ready
            if self.host_ip:
                self.background_operations['progress'] = 0.8
                if not self._wait_for_backend_async(self.host_ip, self.backend_port):
                    return False
            
            self.background_operations['progress'] = 1.0
            return True
            
        except Exception as e:
            self.logger.error(f"Async backend startup failed: {e}")
            return False
    
    def _wait_for_backend_async(self, host: str, port: int) -> bool:
        """Wait for backend with progress updates"""
        url = f"http://{host}:{port}"
        start_time = time.time()
        timeout = 30
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{url}/docs", timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
            
            # Update progress based on elapsed time
            elapsed = time.time() - start_time
            progress = min(0.95, 0.8 + (elapsed / timeout) * 0.15)
            self.background_operations['progress'] = progress
            
            time.sleep(1)
        
        return False
    
    def _start_frontend_async(self) -> bool:
        """Start frontend with progress updates"""
        try:
            self.background_operations['progress'] = 0.3
            
            # Create temporary Vite config
            vite_config = f"""
import {{ defineConfig }} from 'vite'
import react from '@vitejs/plugin-react'
import {{ VitePWA }} from 'vite-plugin-pwa'

export default defineConfig({{
  plugins: [
    react({{
      refresh: false
    }}),
    VitePWA({{
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'favicon-16x16.png'],
      manifest: {{
        name: '2048 Bot Training',
        short_name: '2048 AI',
        description: 'Real-time visualization for 2048 bot training',
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
    hmr: {{
      port: {self.frontend_port + 1},
      host: '0.0.0.0'
    }},
    timeout: 30000,
    cors: {{
      origin: ['http://{self.host_ip}:{self.frontend_port}', 'http://localhost:{self.frontend_port}', 'https://*.trycloudflare.com', 'https://*.cfargotunnel.com'],
      credentials: true
    }}
  }},
  define: {{
    global: 'globalThis',
    __BACKEND_URL__: JSON.stringify('{self.tunnel_url if self.tunnel_url else f"http://{self.host_ip}:{self.backend_port}"}')
  }},
  build: {{
    target: 'es2015',
    minify: false,
    sourcemap: true
  }}
}})
"""
            
            config_path = "frontend/vite.config.temp.ts"
            with open(config_path, "w") as f:
                f.write(vite_config)
            
            self.background_operations['progress'] = 0.5
            
            # Build production bundle
            build_process = subprocess.Popen([
                "npm", "run", "build", "--", "--config", "vite.config.temp.ts"
            ], cwd="frontend", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            # Monitor build progress
            line_count = 0
            while build_process.poll() is None:
                if build_process.stdout is None:
                    break
                line = build_process.stdout.readline()
                if not line:
                    break
                
                line_count += 1
                progress = min(0.8, 0.5 + (line_count / 50) * 0.3)
                self.background_operations['progress'] = progress
                
                self.logger.debug(f"[Frontend Build] {line.strip()}")
            
            build_process.wait()
            
            if build_process.returncode != 0:
                self.logger.error(f"Frontend build failed: {build_process.returncode}")
                return False
            
            self.background_operations['progress'] = 0.9
            
            # Start frontend server
            frontend_cmd = [
                "npm", "run", "preview", "--", 
                "--config", "vite.config.temp.ts",
                "--host", "0.0.0.0",
                "--port", str(self.frontend_port)
            ]
            
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
            if self.host_ip:
                if not self._wait_for_frontend_async(self.host_ip, self.frontend_port):
                    return False
            
            self.background_operations['progress'] = 1.0
            return True
            
        except Exception as e:
            self.logger.error(f"Async frontend startup failed: {e}")
            return False
    
    def _wait_for_frontend_async(self, host: str, port: int) -> bool:
        """Wait for frontend with progress updates"""
        url = f"http://{host}:{port}"
        start_time = time.time()
        timeout = 30
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
            
            time.sleep(1)
        
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Launch 2048 training stack with various deployment options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py                           # Default: Clean QR-focused interface
  python launcher.py --dev                     # Development mode with verbose output
  python launcher.py --lan-only                # LAN access only (no tunnel)
  python launcher.py --tunnel-only             # Tunnel access only (no LAN)
  python launcher.py --tunnel-type named       # Use named tunnel (requires setup)
  python launcher.py --port 9000               # Custom backend port
  python launcher.py --no-qr                   # Skip QR code generation
        """
    )
    
    # Core operation modes
    mode_group = parser.add_argument_group('Operation Modes')
    mode_group.add_argument("--lan-only", action="store_true", 
                           help="LAN access only - no tunnel created (faster startup)")
    mode_group.add_argument("--tunnel-only", action="store_true", 
                           help="Tunnel access only - no LAN serving (cloud-first)")
    mode_group.add_argument("--dev", action="store_true", 
                           help="Development mode - hot reload, detailed logs, LAN only")
    mode_group.add_argument("--production", action="store_true", 
                           help="Production mode - optimized build, tunnel preferred")
    
    # Tunnel configuration
    tunnel_group = parser.add_argument_group('Tunnel Configuration')
    tunnel_group.add_argument("--tunnel-type", choices=["quick", "named"], default="quick",
                             help="Tunnel type: 'quick' (temporary) or 'named' (persistent)")
    tunnel_group.add_argument("--tunnel-name", default="2048-bot",
                             help="Named tunnel identifier (for --tunnel-type named)")
    tunnel_group.add_argument("--tunnel-domain", 
                             help="Custom domain for named tunnel (optional)")
    tunnel_group.add_argument("--no-tunnel-fallback", action="store_true",
                             help="Don't fallback to quick tunnel if named tunnel fails")
    
    # Network configuration
    network_group = parser.add_argument_group('Network Configuration')
    network_group.add_argument("--port", type=int, default=8000,
                              help="Backend server port (default: 8000)")
    network_group.add_argument("--frontend-port", type=int, default=5173,
                              help="Frontend dev server port (default: 5173)")
    network_group.add_argument("--host", default="0.0.0.0",
                              help="Backend server host (default: 0.0.0.0)")
    network_group.add_argument("--force-ports", action="store_true", 
                              help="Force kill processes using required ports")
    
    # Output and UI configuration
    ui_group = parser.add_argument_group('UI and Output')
    ui_group.add_argument("--no-qr", action="store_true", 
                         help="Skip QR code generation (default: enabled)")
    ui_group.add_argument("--no-color", action="store_true", 
                         help="Disable colored output")
    ui_group.add_argument("--quiet", action="store_true", 
                         help="Suppress non-essential output")
    ui_group.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                         default="INFO", help="Set logging level")
    
    # Advanced options
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument("--skip-build", action="store_true",
                               help="Skip frontend build step (use existing dist/)")
    advanced_group.add_argument("--skip-deps", action="store_true",
                               help="Skip dependency checks")
    advanced_group.add_argument("--cloudflared-path", 
                               help="Custom path to cloudflared binary")
    advanced_group.add_argument("--timeout", type=int, default=30,
                               help="Startup timeout in seconds")
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.lan_only and args.tunnel_only:
        parser.error("Cannot specify both --lan-only and --tunnel-only")
    
    if args.dev and args.production:
        parser.error("Cannot specify both --dev and --production")
    
    # Auto-configure based on mode
    if args.dev:
        args.lan_only = True  # Dev mode implies LAN only
        args.log_level = "DEBUG"
    
    if args.production:
        args.tunnel_type = "named"  # Production prefers named tunnels
    
    launcher = Launcher(
        dev_mode=args.dev,
        force_ports=args.force_ports,
        lan_only=args.lan_only,
        tunnel_only=args.tunnel_only,
        tunnel_type=args.tunnel_type,
        tunnel_name=args.tunnel_name,
        tunnel_domain=args.tunnel_domain,
        no_tunnel_fallback=args.no_tunnel_fallback,
        backend_port=args.port,
        frontend_port=args.frontend_port,
        host=args.host,
        no_qr=args.no_qr,
        no_color=args.no_color,
        quiet=args.quiet,
        skip_build=args.skip_build,
        skip_deps=args.skip_deps,
        cloudflared_path=args.cloudflared_path,
        timeout=args.timeout
    )
    success = launcher.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 