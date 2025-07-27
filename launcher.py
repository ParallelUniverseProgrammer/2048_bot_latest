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
import argparse
import logging
import queue
import shutil
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from urllib.parse import urlparse
from datetime import datetime

# Optional imports for enhanced features
try:
    import pystray
    PYTRAY_AVAILABLE = True
except ImportError:
    PYTRAY_AVAILABLE = False

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
    # Modern GUI imports
    import customtkinter as ctk
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "qrcode[pil]", "netifaces", "pillow", "customtkinter"], check=True, shell=True)
    # Re-import after installation
    import qrcode
    import qrcode.image.svg
    import netifaces
    from qrcode import constants
    # GUI imports for QR-only mode
    import tkinter as tk
    from tkinter import ttk, messagebox
    from PIL import Image, ImageTk
    # Modern GUI imports
    import customtkinter as ctk

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


class GUIWindow:
    """Compact desktop GUI window optimized for stability and efficiency"""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.window = None
        self.progress_bar = None
        self.status_label = None
        self.step_label = None
        self.error_label = None
        self.url_entry = None
        self.copy_button = None
        self.system_tray = None
        self.current_progress = 0.0
        self.target_progress = 0.0
        
        # Configure CustomTkinter for vibrant PC-forward appearance
        ctk.set_appearance_mode("dark")  # Force dark theme for consistency
        ctk.set_default_color_theme("blue")
        
        # Create compact main window
        self.window = ctk.CTk()
        self.window.title("2048 Bot Launcher")
        self.window.geometry("450x420")  # Increased height to accommodate all content
        self.window.resizable(False, False)  # Fixed size
        self.window.minsize(450, 420)
        self.window.maxsize(450, 420)
        
        # Set project icon
        try:
            icon_path = Path("project_icon.png")
            if icon_path.exists():
                if platform.system() == "Windows":
                    ico_path = Path("project_icon.ico")
                    if not ico_path.exists():
                        from PIL import Image
                        img = Image.open(icon_path)
                        img.save(ico_path, format='ICO')
                    self.window.iconbitmap(str(ico_path))
                else:
                    self.window.iconphoto(True, tk.PhotoImage(file=str(icon_path)))
        except Exception as e:
            self.logger.warning(f"Could not set window icon: {e}")
        
        # Center window on screen with CustomTkinter-specific fixes
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (450 // 2)
        y = (self.window.winfo_screenheight() // 2) - (420 // 2)
        
        # Force window constraints first to prevent square appearance
        self.window.minsize(450, 480)  # Increased height to accommodate buttons
        self.window.maxsize(450, 480)  # Increased height to accommodate buttons
        self.window.resizable(False, False)
        
        # Set geometry with delayed approach for Windows compatibility
        self.window.after(100, lambda: self._set_window_geometry(x, y))
        
        # Create main container with consistent padding
        self.main_frame = ctk.CTkFrame(self.window, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=12, pady=12)
        
        # Calculate available space for dynamic sizing
        self.window.update_idletasks()
        available_width = 450 - 24  # 426px
        available_height = 480 - 24  # 456px - increased for buttons
        
        # Force window to render and get actual dimensions
        self.window.update()
        actual_width = self.window.winfo_width()
        actual_height = self.window.winfo_height()
        
        # Use actual dimensions if available, otherwise fall back to calculated
        if actual_width > 0 and actual_height > 0:
            available_width = actual_width - 24
            available_height = actual_height - 24
            self.logger.info(f"Using actual window dimensions: {actual_width}x{actual_height}")
        else:
            self.logger.info(f"Using calculated dimensions: {available_width}x{available_height}")
        
        # Progress section (fixed height: ~60px)
        self.progress_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.progress_frame.pack(fill="x", pady=(0, 8))
        
        # Step label (large, prominent, centered)
        self.step_label = ctk.CTkLabel(
            self.progress_frame,
            text="Initializing...",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("black", "white")
        )
        self.step_label.pack(pady=(0, 8))
        
        # Progress bar (thicker, vibrant colors)
        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame, 
            height=20,
            progress_color="#3b82f6",  # Vibrant blue from style guide
            fg_color="#1e293b"  # Darker background
        )
        self.progress_bar.pack(fill="x", pady=(0, 8))
        self.progress_bar.set(0)
        
        # Error display (integrated into progress area)
        self.error_label = ctk.CTkLabel(
            self.progress_frame,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="red",
            wraplength=available_width - 24
        )
        
        # Status and controls section (fixed height, no expansion)
        self.status_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.status_frame.pack(fill="x")
        
        # Service status grid (fixed height: ~40px)
        self.status_grid = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        self.status_grid.pack(fill="x", pady=(0, 8))
        
        # Configure grid columns
        self.status_grid.grid_columnconfigure(0, weight=1)
        self.status_grid.grid_columnconfigure(1, weight=1)
        self.status_grid.grid_columnconfigure(2, weight=1)
        
        # Backend status widget (sophisticated indicator)
        self.backend_status_frame = ctk.CTkFrame(
            self.status_grid,
            fg_color="#1e293b",  # Dark background
            corner_radius=8,
            height=40
        )
        self.backend_status_frame.grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        self.backend_status_frame.grid_propagate(False)  # Maintain fixed height
        
        # Backend status indicator (animated dot)
        self.backend_indicator = ctk.CTkLabel(
            self.backend_status_frame,
            text="●",
            font=ctk.CTkFont(size=12),
            text_color="#f59e0b"  # Orange for starting
        )
        self.backend_indicator.pack(side="left", padx=(8, 4))
        
        # Backend status text
        self.backend_status = ctk.CTkLabel(
            self.backend_status_frame,
            text="Backend\nStarting...",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#f59e0b"
        )
        self.backend_status.pack(side="left", padx=(0, 8), fill="both", expand=True)
        
        # Frontend status widget (sophisticated indicator)
        self.frontend_status_frame = ctk.CTkFrame(
            self.status_grid,
            fg_color="#1e293b",  # Dark background
            corner_radius=8,
            height=40
        )
        self.frontend_status_frame.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        self.frontend_status_frame.grid_propagate(False)  # Maintain fixed height
        
        # Frontend status indicator (animated dot)
        self.frontend_indicator = ctk.CTkLabel(
            self.frontend_status_frame,
            text="●",
            font=ctk.CTkFont(size=12),
            text_color="#f59e0b"  # Orange for starting
        )
        self.frontend_indicator.pack(side="left", padx=(8, 4))
        
        # Frontend status text
        self.frontend_status = ctk.CTkLabel(
            self.frontend_status_frame,
            text="Frontend\nStarting...",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#f59e0b"
        )
        self.frontend_status.pack(side="left", padx=(0, 8), fill="both", expand=True)
        
        # Tunnel status widget (sophisticated indicator)
        self.tunnel_status_frame = ctk.CTkFrame(
            self.status_grid,
            fg_color="#1e293b",  # Dark background
            corner_radius=8,
            height=40
        )
        self.tunnel_status_frame.grid(row=0, column=2, padx=4, pady=4, sticky="ew")
        self.tunnel_status_frame.grid_propagate(False)  # Maintain fixed height
        
        # Tunnel status indicator (animated dot)
        self.tunnel_indicator = ctk.CTkLabel(
            self.tunnel_status_frame,
            text="●",
            font=ctk.CTkFont(size=12),
            text_color="#6b7280"  # Gray for not started
        )
        self.tunnel_indicator.pack(side="left", padx=(8, 4))
        
        # Tunnel status text
        self.tunnel_status = ctk.CTkLabel(
            self.tunnel_status_frame,
            text="Tunnel\nNot started",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color="#6b7280"
        )
        self.tunnel_status.pack(side="left", padx=(0, 8), fill="both", expand=True)
        
        # URL and QR display (centered)
        self.url_frame = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        self.url_frame.pack(fill="x", pady=(0, 8))
        
        # URL section (centered)
        self.url_label = ctk.CTkLabel(
            self.url_frame,
            text="Access URL:",
            font=ctk.CTkFont(size=10, weight="bold")
        )
        self.url_label.pack(pady=(0, 4))
        
        self.url_entry = ctk.CTkEntry(
            self.url_frame,
            font=ctk.CTkFont(size=9),
            height=24,
            placeholder_text="Generating..."
        )
        self.url_entry.pack(fill="x", pady=(0, 8))
        
        # QR code section (centered, fixed size)
        self.qr_frame = ctk.CTkFrame(self.url_frame, fg_color="transparent")
        self.qr_frame.pack(pady=(0, 4))
        
        self.qr_label = ctk.CTkLabel(
            self.qr_frame,
            text="QR Code:",
            font=ctk.CTkFont(size=9, weight="bold")
        )
        self.qr_label.pack(pady=(0, 4))
        
        # QR code placeholder (180x180 - larger to fill extra space)
        self.qr_placeholder_frame = ctk.CTkFrame(
            self.qr_frame,
            fg_color="#374151",  # Style guide background
            corner_radius=8,
            width=180,
            height=180
        )
        self.qr_placeholder_frame.pack(pady=(0, 0))
        self.qr_placeholder_frame.pack_propagate(False)  # Maintain size
        
        # QR image label (will show placeholder immediately)
        self.qr_image_label = ctk.CTkLabel(self.qr_frame, text="")
        
        # Create and show placeholder QR code immediately
        self._create_placeholder_qr()
        
        # Control buttons (ensure visibility with fixed positioning)
        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.button_frame.pack(fill="x", side="bottom", pady=(4, 0))  # Reduced padding
        
        # Configure button grid
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)
        self.button_frame.grid_columnconfigure(2, weight=1)
        self.button_frame.grid_columnconfigure(3, weight=1)
        
        # Calculate optimal button height - now with more space
        # Fixed elements: Progress(70) + Status(52) + URL(64) + QR(178) + Spacing(48) = 412px
        # Available: 456px, Remaining: 44px for buttons + padding
        optimal_button_height = 32  # Back to comfortable size
        
        # Stop button (vibrant red, optimal size)
        self.stop_button = ctk.CTkButton(
            self.button_frame,
            text="Stop",
            font=ctk.CTkFont(size=10, weight="bold"),
            height=optimal_button_height,
            fg_color="#ef4444",  # Vibrant red from style guide
            hover_color="#dc2626",  # Darker red
            command=self._stop_services
        )
        self.stop_button.grid(row=0, column=0, padx=2, pady=4, sticky="ew")
        
        # Restart button (vibrant blue, optimal size)
        self.restart_button = ctk.CTkButton(
            self.button_frame,
            text="Restart",
            font=ctk.CTkFont(size=10, weight="bold"),
            height=optimal_button_height,
            fg_color="#3b82f6",  # Vibrant blue from style guide
            hover_color="#2563eb",  # Darker blue
            command=self._restart_services
        )
        self.restart_button.grid(row=0, column=1, padx=2, pady=4, sticky="ew")
        
        # Logs button (vibrant gray, optimal size)
        self.logs_button = ctk.CTkButton(
            self.button_frame,
            text="Logs",
            font=ctk.CTkFont(size=10, weight="bold"),
            height=optimal_button_height,
            fg_color="#6b7280",  # Vibrant gray from style guide
            hover_color="#4b5563",  # Darker gray
            command=self._view_logs
        )
        self.logs_button.grid(row=0, column=2, padx=2, pady=4, sticky="ew")
        
        # Copy URL button (vibrant green, optimal size)
        self.copy_button = ctk.CTkButton(
            self.button_frame,
            text="Copy URL",
            font=ctk.CTkFont(size=10, weight="bold"),
            height=optimal_button_height,
            fg_color="#22c55e",  # Vibrant green from style guide
            hover_color="#16a34a",  # Darker green
            command=self._copy_url
        )
        self.copy_button.grid(row=0, column=3, padx=2, pady=4, sticky="ew")
        
        # Initialize system tray
        self._init_system_tray()
        
        # Bind window close event
        self.window.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Bind keyboard shortcuts
        self.window.bind("<Control-s>", lambda e: self._stop_services())
        self.window.bind("<Control-r>", lambda e: self._restart_services())
        self.window.bind("<Control-l>", lambda e: self._view_logs())
        self.window.bind("<Control-c>", lambda e: self._copy_url())
        
        # Focus on window
        self.window.focus_force()
        
        # Start progress animation
        self._start_progress_animation()
    
    def _set_window_geometry(self, x: int, y: int):
        """Set window geometry with proper CustomTkinter handling"""
        try:
            # Force window update before setting geometry
            self.window.update_idletasks()
            
            # Set the geometry
            self.window.geometry(f"450x480+{x}+{y}")
            
            # Force another update to ensure geometry is applied
            self.window.update()
            
            # Double-check constraints are still enforced
            self.window.minsize(450, 480)
            self.window.maxsize(450, 480)
            
            self.logger.info("Window geometry set successfully")
        except Exception as e:
            self.logger.error(f"Failed to set window geometry: {e}")
            # Fallback: try setting geometry directly
            try:
                self.window.geometry(f"450x480+{x}+{y}")
            except Exception as e2:
                self.logger.error(f"Fallback geometry setting also failed: {e2}")
    
    def _create_placeholder_qr(self):
        """Create a blurred placeholder QR code"""
        self.logger.info("Creating placeholder QR code...")
        try:
            from PIL import Image, ImageDraw, ImageFilter
            
            # Create a simple QR-like pattern (larger size)
            size = 180
            img = Image.new('RGB', (size, size), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw a simple grid pattern that looks like a QR code
            for i in range(0, size, 18):
                for j in range(0, size, 18):
                    if (i + j) % 36 == 0:  # Create a pattern
                        draw.rectangle([i, j, i+17, j+17], fill='black')
            
            # Add corner squares (like real QR codes)
            draw.rectangle([0, 0, 44, 44], fill='black')
            draw.rectangle([5, 5, 39, 39], fill='white')
            draw.rectangle([10, 10, 34, 34], fill='black')
            
            draw.rectangle([size-45, 0, size-1, 44], fill='black')
            draw.rectangle([size-40, 5, size-6, 39], fill='white')
            draw.rectangle([size-35, 10, size-11, 34], fill='black')
            
            draw.rectangle([0, size-45, 44, size-1], fill='black')
            draw.rectangle([5, size-40, 39, size-6], fill='white')
            draw.rectangle([10, size-35, 34, size-11], fill='black')
            
            # Blur the image to make it look like a placeholder
            img = img.filter(ImageFilter.GaussianBlur(radius=2))
            
            # Convert to CTkImage (larger size)
            qr_photo = ctk.CTkImage(light_image=img, dark_image=img, size=(180, 180))
            
            # Show the placeholder immediately
            self.qr_placeholder_frame.pack_forget()
            self.qr_image_label.configure(image=qr_photo, text="")
            self.qr_image_label.image = qr_photo  # Keep reference
            self.qr_image_label.pack(pady=(0, 4))
            
            # Force update to ensure it's visible
            self.window.update()
            self.logger.info("Placeholder QR code created and displayed successfully")
            
        except ImportError:
            # Fallback to simple text if PIL not available
            self.qr_placeholder_label = ctk.CTkLabel(
                self.qr_placeholder_frame,
                text="QR\nCode",
                font=ctk.CTkFont(size=8),
                text_color=("gray", "lightgray")
            )
            self.qr_placeholder_label.pack(expand=True)
            self.window.update()
        except Exception as e:
            self.logger.error(f"Error creating placeholder QR: {e}")
            # Ensure we always show something
            self.qr_placeholder_label = ctk.CTkLabel(
                self.qr_placeholder_frame,
                text="QR\nCode",
                font=ctk.CTkFont(size=8),
                text_color=("gray", "lightgray")
            )
            self.qr_placeholder_label.pack(expand=True)
            self.window.update()
    
    def _start_progress_animation(self):
        """Start smooth progress bar animation"""
        def animate_progress():
            if hasattr(self, 'current_progress') and hasattr(self, 'target_progress'):
                if abs(self.current_progress - self.target_progress) > 0.01:
                    self.current_progress += (self.target_progress - self.current_progress) * 0.1
                    self.progress_bar.set(self.current_progress)
            self.window.after(50, animate_progress)  # 20 FPS
        
        animate_progress()
    
    def _init_system_tray(self):
        """Initialize system tray icon"""
        try:
            import pystray
            from PIL import Image
            
            # Create tray icon
            icon_path = Path("project_icon.png")
            if icon_path.exists():
                tray_image = Image.open(icon_path)
            else:
                # Create a simple colored square as fallback
                tray_image = Image.new('RGB', (64, 64), color='blue')
            
            # Create tray menu
            menu = pystray.Menu(
                pystray.MenuItem("Show Window", self._show_from_tray),
                pystray.MenuItem("Stop Services", self._stop_services),
                pystray.MenuItem("Copy URL", self._copy_url),
                pystray.MenuItem("View Logs", self._view_logs),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Exit", self._quit_from_tray)
            )
            
            self.system_tray = pystray.Icon("2048_bot", tray_image, "2048 Bot Launcher", menu)
            
        except ImportError:
            self.logger.warning("pystray not available, system tray disabled")
            self.system_tray = None
        except Exception as e:
            self.logger.error(f"Failed to initialize system tray: {e}")
            self.system_tray = None
    
    def _show_from_tray(self, icon=None, item=None):
        """Show window from system tray"""
        self.show()
        if self.system_tray:
            self.system_tray.stop()
    
    def _quit_from_tray(self, icon=None, item=None):
        """Quit application from system tray"""
        self._on_closing()
    
    def _copy_url(self):
        """Copy URL to clipboard"""
        try:
            url = self.url_entry.get()
            if url and url != "Generating...":
                self.window.clipboard_clear()
                self.window.clipboard_append(url)
                self.window.update()
                
                # Show temporary success message
                original_text = self.copy_button.cget("text")
                self.copy_button.configure(text="Copied!")
                self.window.after(2000, lambda: self.copy_button.configure(text=original_text))
                
                self.logger.info(f"URL copied to clipboard: {url}")
        except Exception as e:
            self.logger.error(f"Failed to copy URL: {e}")
    
    def _stop_services(self):
        """Stop all services"""
        try:
            self.logger.info("Stop services requested")
            if hasattr(self, 'on_stop_requested'):
                self.on_stop_requested()
        except Exception as e:
            self.logger.error(f"Error stopping services: {e}")
    
    def _restart_services(self):
        """Restart all services"""
        try:
            self.logger.info("Restart services requested")
            if hasattr(self, 'on_restart_requested'):
                self.on_restart_requested()
        except Exception as e:
            self.logger.error(f"Error restarting services: {e}")
    
    def _view_logs(self):
        """Open log file"""
        try:
            import subprocess
            import platform
            
            log_file = "launcher.log"
            if platform.system() == "Windows":
                subprocess.run(["notepad", log_file], check=False)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", "-a", "TextEdit", log_file], check=False)
            else:  # Linux
                subprocess.run(["xdg-open", log_file], check=False)
        except Exception as e:
            self.logger.error(f"Error opening logs: {e}")
    
    def _on_closing(self):
        """Handle window closing"""
        try:
            # Minimize to tray instead of closing
            if self.system_tray and PYTRAY_AVAILABLE:
                self.hide()
                self.system_tray.run()
            else:
                # Signal to launcher to cleanup
                if hasattr(self, 'on_closing'):
                    self.on_closing()
                self.window.destroy()
        except Exception as e:
            self.logger.error(f"Error during window closing: {e}")
            try:
                self.window.destroy()
            except:
                pass
    
    def update_progress(self, step: str, progress: float, status: str = "", error: str = ""):
        """Update progress display with smooth animation"""
        try:
            self.step_label.configure(text=step)
            self.target_progress = progress
            
            if error:
                self.error_label.configure(text=f"Error: {error}")
                self.error_label.pack(pady=(4, 0))
                # Change progress bar color to indicate error
                self.progress_bar.configure(progress_color="red")
            else:
                self.error_label.pack_forget()
                # Restore normal progress bar color
                self.progress_bar.configure(progress_color="#3b82f6")  # Vibrant blue
                
            self.window.update()
        except Exception as e:
            self.logger.error(f"Error updating progress: {e}")
    
    def show_access_info(self, frontend_url: str, backend_url: str):
        """Show access information with QR code"""
        try:
            # Update URL entry
            self.url_entry.delete(0, "end")
            self.url_entry.insert(0, frontend_url)
            
            # Generate QR code
            try:
                import qrcode
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_L,
                    box_size=5,  # Larger boxes for 180x180 size
                    border=4,
                )
                qr.add_data(frontend_url)
                qr.make(fit=True)
                
                qr_image = qr.make_image(fill_color="black", back_color="white")
                
                # Convert to PIL Image
                if hasattr(qr_image, '_img'):
                    pil_image = qr_image._img
                elif hasattr(qr_image, 'convert'):
                    pil_image = qr_image
                else:
                    pil_image = qr_image.convert('RGB')
                
                # Convert to CTkImage for 180x180 size
                qr_photo = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(180, 180))
                
                # Hide placeholder and show QR code
                self.qr_placeholder_frame.pack_forget()
                self.qr_image_label.configure(image=qr_photo, text="")
                self.qr_image_label.image = qr_photo  # Keep reference
                self.qr_image_label.pack(anchor="w", pady=(0, 4))
                
            except ImportError:
                self.logger.warning("qrcode library not available, skipping QR generation")
            except Exception as e:
                self.logger.error(f"Error generating QR code: {e}")
            
            # Update status to show ready
            self.step_label.configure(text="Ready!")
            self.progress_bar.set(1.0)
            
            # Show success in status
            self.backend_status.configure(text="Backend\nReady", text_color="green")
            self.frontend_status.configure(text="Frontend\nReady", text_color="green")
            
            self.window.update()
            self.logger.info(f"Access info displayed: {frontend_url}")
        except Exception as e:
            self.logger.error(f"Error showing access info: {e}")
    
    def update_status(self, backend_status: str, frontend_status: str, tunnel_status: str):
        """Update service status with sophisticated indicators and animations"""
        try:
            def get_status_config(status: str) -> tuple[str, str, str]:
                """Get color, indicator symbol, and animation state for status"""
                status_lower = status.lower()
                if "running" in status_lower or "ready" in status_lower:
                    return "#00ff00", "●", "stable"  # green
                elif "starting" in status_lower or "connecting" in status_lower:
                    return "#ffa500", "●", "pulsing"  # orange
                elif "error" in status_lower or "failed" in status_lower:
                    return "#ff0000", "●", "error"  # red
                else:
                    return "#808080", "○", "stable"  # gray
            
            # Update backend status
            backend_color, backend_symbol, backend_animation = get_status_config(backend_status)
            self.backend_indicator.configure(
                text=backend_symbol,
                text_color=backend_color
            )
            self.backend_status.configure(
                text=f"Backend\n{backend_status}",
                text_color=backend_color
            )
            
            # Update frontend status
            frontend_color, frontend_symbol, frontend_animation = get_status_config(frontend_status)
            self.frontend_indicator.configure(
                text=frontend_symbol,
                text_color=frontend_color
            )
            self.frontend_status.configure(
                text=f"Frontend\n{frontend_status}",
                text_color=frontend_color
            )
            
            # Update tunnel status
            tunnel_color, tunnel_symbol, tunnel_animation = get_status_config(tunnel_status)
            self.tunnel_indicator.configure(
                text=tunnel_symbol,
                text_color=tunnel_color
            )
            self.tunnel_status.configure(
                text=f"Tunnel\n{tunnel_status}",
                text_color=tunnel_color
            )
            
            # Start animations for pulsing indicators
            self._start_status_animations(backend_animation, frontend_animation, tunnel_animation)
            
            self.window.update()
        except Exception as e:
            self.logger.error(f"Error updating status: {e}")
    
    def _start_status_animations(self, backend_anim: str, frontend_anim: str, tunnel_anim: str):
        """Start appropriate animations for status indicators"""
        try:
            # Cancel any existing animations
            if hasattr(self, '_animation_jobs'):
                for job in self._animation_jobs:
                    self.window.after_cancel(job)
            
            self._animation_jobs = []
            
            # Backend animation
            if backend_anim == "pulsing":
                self._animate_indicator(self.backend_indicator, "pulse")
            elif backend_anim == "error":
                self._animate_indicator(self.backend_indicator, "error")
            
            # Frontend animation
            if frontend_anim == "pulsing":
                self._animate_indicator(self.frontend_indicator, "pulse")
            elif frontend_anim == "error":
                self._animate_indicator(self.frontend_indicator, "error")
            
            # Tunnel animation
            if tunnel_anim == "pulsing":
                self._animate_indicator(self.tunnel_indicator, "pulse")
            elif tunnel_anim == "error":
                self._animate_indicator(self.tunnel_indicator, "error")
                
        except Exception as e:
            self.logger.error(f"Error starting status animations: {e}")
    
    def _animate_indicator(self, indicator: ctk.CTkLabel, animation_type: str):
        """Animate a status indicator"""
        try:
            if animation_type == "pulse":
                # Pulsing animation
                def pulse_animation(alpha=1.0, direction=-1):
                    if not hasattr(self, '_animation_jobs'):
                        return
                    
                    # Update opacity by changing text color alpha
                    current_color = indicator.cget("text_color")
                    if isinstance(current_color, str):
                        # Handle color names and convert to hex if needed
                        color_map = {
                            "red": "#ff0000",
                            "green": "#00ff00", 
                            "blue": "#0000ff",
                            "orange": "#ffa500",
                            "yellow": "#ffff00",
                            "purple": "#800080",
                            "gray": "#808080",
                            "grey": "#808080",
                            "black": "#000000",
                            "white": "#ffffff"
                        }
                        
                        # If it's a color name, convert to hex
                        if current_color.lower() in color_map:
                            current_color = color_map[current_color.lower()]
                        
                        # Only try to parse as hex if it looks like a hex color
                        if current_color.startswith('#') and len(current_color) == 7:
                            try:
                                color = current_color.lstrip('#')
                                r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
                                # Apply alpha by darkening the color
                                r = int(r * alpha)
                                g = int(g * alpha)
                                b = int(b * alpha)
                                new_color = f"#{r:02x}{g:02x}{b:02x}"
                                indicator.configure(text_color=new_color)
                            except (ValueError, IndexError):
                                # If hex parsing fails, just use the original color
                                pass
                    
                    # Schedule next frame
                    if direction == -1 and alpha > 0.3:
                        alpha -= 0.1
                    elif direction == -1 and alpha <= 0.3:
                        direction = 1
                    elif direction == 1 and alpha < 1.0:
                        alpha += 0.1
                    elif direction == 1 and alpha >= 1.0:
                        direction = -1
                    
                    job = self.window.after(100, lambda: pulse_animation(alpha, direction))
                    if hasattr(self, '_animation_jobs'):
                        self._animation_jobs.append(job)
                
                pulse_animation()
                
            elif animation_type == "error":
                # Error animation (rapid blinking)
                def error_animation(visible=True):
                    if not hasattr(self, '_animation_jobs'):
                        return
                    
                    if visible:
                        indicator.configure(text="●")
                    else:
                        indicator.configure(text="")
                    
                    job = self.window.after(300, lambda: error_animation(not visible))
                    if hasattr(self, '_animation_jobs'):
                        self._animation_jobs.append(job)
                
                error_animation()
                
        except Exception as e:
            self.logger.error(f"Error animating indicator: {e}")
    
    def show_error(self, error: str):
        """Show error message integrated into progress area"""
        try:
            self.error_label.configure(text=f"Error: {error}")
            self.error_label.pack(pady=(4, 0))
            self.progress_bar.configure(progress_color="red")
            self.window.update()
        except Exception as e:
            self.logger.error(f"Error showing error: {e}")
    
    def hide_error(self):
        """Hide error message"""
        try:
            self.error_label.pack_forget()
            self.progress_bar.configure(progress_color=("blue", "lightblue"))
            self.window.update()
        except Exception as e:
            self.logger.error(f"Error hiding error: {e}")
    
    def show(self):
        """Show the window"""
        try:
            self.window.deiconify()
            self.window.lift()
            self.window.focus_force()
        except Exception as e:
            self.logger.error(f"Error showing window: {e}")
    
    def hide(self):
        """Hide the window"""
        try:
            self.window.withdraw()
        except Exception as e:
            self.logger.error(f"Error hiding window: {e}")
    
    def destroy(self):
        """Destroy the window and cleanup"""
        try:
            # Stop any running animations
            self._stop_status_animations()
            
            if self.system_tray and PYTRAY_AVAILABLE:
                self.system_tray.stop()
            self.window.destroy()
        except Exception as e:
            self.logger.error(f"Error destroying window: {e}")
            try:
                self.window.destroy()
            except:
                pass
            
            if self.window:
                self.window.destroy()
        except Exception as e:
            self.logger.error(f"Error destroying window: {e}")
    
    def _stop_status_animations(self):
        """Stop all status indicator animations"""
        try:
            if hasattr(self, '_animation_jobs'):
                for job in self._animation_jobs:
                    try:
                        self.window.after_cancel(job)
                    except:
                        pass
                self._animation_jobs = []
        except Exception as e:
            self.logger.error(f"Error stopping status animations: {e}")


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
        """Get status of all processes with debouncing to prevent rapid state changes"""
        status = {}
        current_time = time.time()
        
        for name, process in self.processes.items():
            monitor = self.monitors.get(name)
            
            # Get current process state
            poll_result = process.poll()
            is_running = poll_result is None
            
            # Initialize debouncing for this process if not exists
            if not hasattr(self, '_status_debounce'):
                self._status_debounce = {}
            if name not in self._status_debounce:
                self._status_debounce[name] = {
                    'last_state': is_running,
                    'last_change_time': current_time,
                    'stable_duration': 1.0  # Require 1 second of stable state
                }
            
            debounce_info = self._status_debounce[name]
            
            # Check if state has changed
            if is_running != debounce_info['last_state']:
                # State changed, update timestamp
                debounce_info['last_change_time'] = current_time
                debounce_info['last_state'] = is_running
            
            # Only consider state stable if it hasn't changed for the debounce duration
            time_since_change = current_time - debounce_info['last_change_time']
            stable_state = time_since_change >= debounce_info['stable_duration']
            
            # Use the stable state, or the current state if we're still in debounce period
            final_running_state = debounce_info['last_state'] if stable_state else is_running
            
            status[name] = {
                'pid': process.pid,
                'running': final_running_state,
                'returncode': poll_result,
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
        
        # Clear debounce state
        if hasattr(self, '_status_debounce'):
            self._status_debounce.clear()
        
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
    """Generate QR codes for mobile access"""
    
    @staticmethod
    def generate_qr_code(url: str, output_path: Optional[str] = None, center: bool = False, term_width: int = 80) -> None:
        """Generate a QR code for the given URL"""
        try:
            import qrcode
            from PIL import Image
            
            # Create QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=1,
                border=2,
            )
            qr.add_data(url)
            qr.make(fit=True)
            
            # Create image
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Save to file if requested
            if output_path:
                img.save(output_path)
            
            # Display in terminal if possible
            if center:
                # Center the QR code in terminal
                qr_width = img.width
                padding = max(0, (term_width - qr_width) // 2)
                print(" " * padding)
            
            # Convert to ASCII art for terminal display
            width, height = img.size
            for y in range(0, height, 2):
                line = ""
                for x in range(width):
                    # Check if pixel is black (QR code data)
                    if img.getpixel((x, y)) == 0:
                        line += "█"
                    else:
                        line += " "
                print(line)
            
        except ImportError:
            # Fallback if qrcode library not available
            print(f"QR Code for: {url}")
            print("(Install 'qrcode' and 'pillow' packages for visual QR code)")


class ProgressMapper:
    """Maps double progress bar system to unified progress for GUI"""
    
    # Step definitions with weights (must sum to 1.0)
    STEPS = [
        ("Checking dependencies", 0.08),    # 8% of total progress
        ("Setting up network", 0.04),       # 4% of total progress  
        ("Installing dependencies", 0.32),  # 32% of total progress
        ("Building frontend assets", 0.16), # 16% of total progress
        ("Starting backend", 0.24),         # 24% of total progress
        ("Starting frontend", 0.16),        # 16% of total progress
    ]
    
    def __init__(self):
        """Initialize progress mapper"""
        self.current_step_index = 0
        self.step_progress = 0.0
        self.micro_progress = 0.0
        self.current_step_name = ""
        self.micro_step_name = ""
        
        # Validate step weights sum to 1.0
        total_weight = sum(weight for _, weight in self.STEPS)
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Step weights must sum to 1.0, got {total_weight}")
    
    def update_step_progress(self, step_name: str, progress: float) -> float:
        """Update step progress and return unified progress (0.0-1.0)"""
        # Find step index
        step_index = None
        for i, (name, _) in enumerate(self.STEPS):
            if name == step_name:
                step_index = i
                break
        
        if step_index is None:
            # Unknown step, assume it's the current step
            step_index = self.current_step_index
        
        # Update state
        self.current_step_index = step_index
        self.current_step_name = step_name
        self.step_progress = max(0.0, min(1.0, progress))
        
        return self._calculate_unified_progress()
    
    def update_micro_progress(self, micro_step: str, progress: float) -> float:
        """Update micro progress and return unified progress (0.0-1.0)"""
        self.micro_step_name = micro_step
        self.micro_progress = max(0.0, min(1.0, progress))
        
        return self._calculate_unified_progress()
    
    def _calculate_unified_progress(self) -> float:
        """Calculate unified progress from step and micro progress"""
        # Calculate base progress from completed steps
        base_progress = 0.0
        for i in range(self.current_step_index):
            base_progress += self.STEPS[i][1]
        
        # Add progress within current step
        current_step_weight = self.STEPS[self.current_step_index][1]
        step_contribution = current_step_weight * self.step_progress
        
        # Add micro progress contribution (weighted by 20% of current step)
        micro_weight = current_step_weight * 0.2
        micro_contribution = micro_weight * self.micro_progress
        
        unified_progress = base_progress + step_contribution + micro_contribution
        
        return max(0.0, min(1.0, unified_progress))
    
    def get_current_step_info(self) -> tuple[str, float, str, float]:
        """Get current step information for debugging"""
        return (
            self.current_step_name,
            self.step_progress,
            self.micro_step_name,
            self.micro_progress
        )
    
    def reset(self):
        """Reset progress mapper state"""
        self.current_step_index = 0
        self.step_progress = 0.0
        self.micro_progress = 0.0
        self.current_step_name = ""
        self.micro_step_name = ""


class Launcher:
    """Enhanced launcher with robust error handling and monitoring"""
    
    def __init__(self, dev_mode: bool = False, force_ports: bool = False,
                 lan_only: bool = False, tunnel_only: bool = False, tunnel_type: str = "quick",
                 tunnel_name: str = "2048-bot", tunnel_domain: Optional[str] = None,
                 no_tunnel_fallback: bool = False, backend_port: int = 8000,
                 frontend_port: int = 5173, host: str = "0.0.0.0", no_qr: bool = False,
                 no_color: bool = False, quiet: bool = False, skip_build: bool = False,
                 skip_deps: bool = False, cloudflared_path: Optional[str] = None,
                 timeout: int = 30, gui: bool = False):
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
        
        # GUI mode
        self.gui_mode = gui
        self.gui_window = None
        if self.gui_mode:
            self.gui_window = GUIWindow(self.logger)
            # Set up callbacks
            self.gui_window.on_stop_requested = self._stop_services
            self.gui_window.on_restart_requested = self._restart_services
            self.gui_window.on_closing = self._cleanup_gui
        
        # Background operations state
        self.background_operations = {
            'current_step': 0,
            'status': 'idle',  # idle, running, completed, failed
            'progress': 0.0,
            'error': None,
            'result': None
        }
        
        # Progress mapper for unified GUI progress
        self.progress_mapper = ProgressMapper()
    
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
    
    def _stop_services(self):
        """Stop all services (GUI callback)"""
        try:
            self.logger.info("Stop services requested from GUI")
            self.cleanup()
            if self.gui_window:
                self.gui_window.update_status("Stopped", "Stopped", "Stopped")
        except Exception as e:
            self.logger.error(f"Error stopping services: {e}")
    
    def _restart_services(self):
        """Restart all services (GUI callback)"""
        try:
            self.logger.info("Restart services requested from GUI")
            # This would require more complex restart logic
            # For now, just show a message
            if self.gui_window:
                self.gui_window.show_error("Restart functionality not yet implemented")
        except Exception as e:
            self.logger.error(f"Error restarting services: {e}")
    
    def _cleanup_gui(self):
        """Cleanup when GUI window is closed"""
        try:
            self.logger.info("GUI window closed, cleaning up")
            self.cleanup()
        except Exception as e:
            self.logger.error(f"Error during GUI cleanup: {e}")
    
    def _update_gui_progress(self, step: str, progress: float, status: str = "", error: str = ""):
        """Update GUI progress display with unified progress mapping"""
        if self.gui_mode and self.gui_window:
            try:
                # Use progress mapper to get unified progress
                unified_progress = self.progress_mapper.update_step_progress(step, progress)
                self.gui_window.update_progress(step, unified_progress, status, error)
            except Exception as e:
                self.logger.error(f"Error updating GUI progress: {e}")
    
    def _update_gui_status(self, backend_status: str, frontend_status: str, tunnel_status: str):
        """Update GUI status display"""
        if self.gui_mode and self.gui_window:
            try:
                self.gui_window.update_status(backend_status, frontend_status, tunnel_status)
            except Exception as e:
                self.logger.error(f"Error updating GUI status: {e}")
    
    def _update_gui_micro_progress(self, micro_step: str, progress: float):
        """Update GUI micro progress with unified progress mapping"""
        if self.gui_mode and self.gui_window:
            try:
                # Use progress mapper to get unified progress including micro progress
                unified_progress = self.progress_mapper.update_micro_progress(micro_step, progress)
                # Get current step info for display
                current_step, step_progress, _, _ = self.progress_mapper.get_current_step_info()
                self.gui_window.update_progress(current_step, unified_progress, f"{micro_step}...")
            except Exception as e:
                self.logger.error(f"Error updating GUI micro progress: {e}")
    
    def _show_gui_access_info(self, frontend_url: str, backend_url: str):
        """Show access info in GUI"""
        self.logger.info(f"Launcher: _show_gui_access_info called with frontend: {frontend_url}, backend: {backend_url}")
        self.logger.info(f"Launcher: gui_mode={self.gui_mode}, gui_window exists={self.gui_window is not None}")
        
        if self.gui_mode and self.gui_window:
            try:
                self.logger.info("Launcher: Calling gui_window.show_access_info")
                self.gui_window.show_access_info(frontend_url, backend_url)
                self.logger.info("Launcher: gui_window.show_access_info completed successfully")
            except Exception as e:
                self.logger.error(f"Error showing GUI access info: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            self.logger.warning("Launcher: GUI mode or window not available for access info display")
    
    def cleanup(self):
        """Cleanup launcher resources"""
        try:
            self.logger.info("Starting launcher cleanup")
            
            # Stop all processes
            if hasattr(self, 'process_manager'):
                self.process_manager.cleanup()
            
            # Stop tunnel
            if hasattr(self, 'tunnel_process') and self.tunnel_process:
                self._stop_tunnel()
            
            # Cleanup console UI
            if hasattr(self, 'console_ui'):
                self.console_ui.cleanup()
            
            # Cleanup GUI
            if self.gui_mode and self.gui_window:
                self.gui_window.destroy()
            
            # Remove temporary config
            temp_config = "frontend/vite.config.temp.ts"
            if os.path.exists(temp_config):
                try:
                    os.remove(temp_config)
                    self.logger.info("Removed temporary Vite config")
                except Exception as e:
                    self.logger.warning(f"Could not remove temporary config: {e}")
            
            self.logger.info("Launcher cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
        
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        self.logger.info("Checking dependencies")
        
        # Update progress
        self._update_gui_progress("Checking dependencies", 0.1, "Verifying project structure...")
        if not self.gui_mode:
            self.console_ui.update_progress("Checking dependencies", 0.1, "Verifying project structure...")
            self.console_ui.update_micro_progress("Checking project directories", 0.0)
        
        # Check if we're in the right directory
        if not os.path.exists("backend") or not os.path.exists("frontend"):
            error_msg = "Error: Please run this script from the project root directory"
            self.logger.error(error_msg)
            return False
        
        if not self.gui_mode:
            self.console_ui.update_micro_progress("Project structure verified", 0.3)
        
        # Check Poetry
        try:
            subprocess.run(["poetry", "--version"], check=True, capture_output=True, shell=True)
            if not self.gui_mode:
                self.console_ui.update_micro_progress("Poetry found", 0.5)
        except (subprocess.CalledProcessError, FileNotFoundError):
            error_msg = "Poetry not found. Please install Poetry first."
            self.logger.error(error_msg)
            return False
        
        # Check Node.js
        try:
            subprocess.run(["node", "--version"], check=True, capture_output=True, shell=True)
            if not self.gui_mode:
                self.console_ui.update_micro_progress("Node.js found", 0.7)
        except (subprocess.CalledProcessError, FileNotFoundError):
            error_msg = "Node.js not found. Please install Node.js first."
            self.logger.error(error_msg)
            return False
        
        # Check npm
        try:
            subprocess.run(["npm", "--version"], check=True, capture_output=True, shell=True)
            if not self.gui_mode:
                self.console_ui.update_micro_progress("npm found", 0.9)
        except (subprocess.CalledProcessError, FileNotFoundError):
            error_msg = "npm not found. Please install npm first."
            self.logger.error(error_msg)
            return False
        
        if not self.gui_mode:
            self.console_ui.update_micro_progress("All dependencies verified", 1.0)
        self.logger.info("All dependencies found")
        return True
    
    def setup_network(self) -> bool:
        """Setup network configuration with port management"""
        self.logger.info("Setting up network configuration")
        
        # Update progress
        self._update_gui_progress("Setting up network", 0.2, "Discovering network interfaces...")
        if not self.gui_mode:
            self.console_ui.update_progress("Setting up network", 0.2, "Discovering network interfaces...")
            self.console_ui.update_micro_progress("Scanning network adapters", 0.0)
        
        # Find the best IP address
        self.host_ip = NetworkDiscovery.find_best_ip()
        if not self.host_ip:
            error_msg = "Could not determine local IP address"
            self.logger.error(error_msg)
            return False
        
        if not self.gui_mode:
            self.console_ui.update_micro_progress(f"Found IP: {self.host_ip}", 0.5)
        self.logger.info(f"Using IP address: {self.host_ip}")
        
        # Check and manage ports
        if not self.gui_mode:
            self.console_ui.update_micro_progress("Checking port availability", 0.7)
        if not self._setup_ports():
            return False
        
        if not self.gui_mode:
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
    
    def _build_frontend_assets(self) -> bool:
        """Build frontend assets to ensure they exist for backend startup"""
        self.logger.info("Building frontend assets for backend")
        
        # Update progress
        self.console_ui.update_progress("Building frontend assets", 0.75, "Creating production build...")
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
            self.logger.info("Created temporary Vite config for asset build")
            self.console_ui.update_micro_progress("Vite config created", 0.2)
        except Exception as e:
            error_msg = f"Failed to create Vite config: {e}"
            print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
            return False
        
        # Build production bundle
        self.console_ui.update_micro_progress("Building production bundle", 0.4)
        if not self.quiet:
            print(f"{Colors.OKCYAN}Building frontend assets...{Colors.ENDC}")
        
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
                    self.console_ui.update_micro_progress("Compiling TypeScript", 0.5)
                elif "chunks" in line:
                    self.console_ui.update_micro_progress("Bundling assets", 0.6)
                elif "dist" in line:
                    self.console_ui.update_micro_progress("Writing output files", 0.7)
                elif "built" in line:
                    self.console_ui.update_micro_progress("Build completed", 0.8)
                
                # Log to file but don't spam console
                self.logger.debug(f"[Frontend Asset Build] {line.strip()}")
            
            # Wait for the process to complete and get the return code
            build_process.wait()
            
            if build_process.returncode != 0:
                error_msg = f"Frontend asset build failed (exit code: {build_process.returncode})"
                if not self.quiet:
                    print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
                self.logger.error(error_msg)
                return False
            
            if not self.quiet:
                print(f"{Colors.OKGREEN}✓ Frontend assets built successfully{Colors.ENDC}")
            self.logger.info("Frontend assets built successfully")
            self.console_ui.update_micro_progress("Assets ready", 0.9)
            
            # Verify that the assets directory exists
            assets_dir = "frontend/dist/assets"
            if not os.path.exists(assets_dir):
                error_msg = f"Assets directory not found after build: {assets_dir}"
                print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
                self.logger.error(error_msg)
                return False
            
            self.console_ui.update_micro_progress("Assets verified", 1.0)
            return True
            
        except Exception as e:
            error_msg = f"Frontend asset build failed: {e}"
            if not self.quiet:
                print(f"{Colors.FAIL}✗ {error_msg}{Colors.ENDC}")
            self.logger.error(error_msg)
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
        
        # Show in GUI if in GUI mode
        if self.gui_mode and primary_url:
            self._show_gui_access_info(primary_url, backend_url if not self.tunnel_only else self.tunnel_url)
        
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
        if self.gui_mode:
            # GUI mode - show window and run in background
            return self._run_gui_mode()
        elif self.qr_only:
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
                
                # Build frontend first to ensure static assets exist for backend
                if not self.dev_mode and not self.skip_build:
                    if not self._build_frontend_assets():
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
    
    def _run_gui_mode(self):
        """Run launcher in GUI mode"""
        try:
            # Show the GUI window
            self.gui_window.show()
            
            # Start background operations
            background_thread = threading.Thread(target=self._run_background_operations, daemon=True)
            background_thread.start()
            
            # Start status update thread
            status_thread = threading.Thread(target=self._update_gui_status_periodic, daemon=True)
            status_thread.start()
            
            # Start GUI main loop
            self.gui_window.window.mainloop()
            
            return True
        except Exception as e:
            error_msg = f"GUI mode error: {e}"
            self.logger.error(error_msg)
            if self.gui_window:
                self.gui_window.show_error(error_msg)
            return False
        finally:
            # Cleanup
            if self.gui_window:
                self.gui_window.destroy()
            self.process_manager.cleanup()
            self._stop_tunnel()
            temp_config = "frontend/vite.config.temp.ts"
            if os.path.exists(temp_config):
                try:
                    os.remove(temp_config)
                    self.logger.info("Removed temporary Vite config")
                except Exception as e:
                    self.logger.warning(f"Could not remove temporary config: {e}")
            self.logger.info("GUI launcher cleanup completed")
    
    def _update_gui_status_periodic(self):
        """Periodically update GUI status"""
        while self.gui_mode and self.gui_window:
            try:
                # Get process status
                status = self.process_manager.get_process_status()
                
                backend_status = "Running" if status.get("Backend", {}).get("running", False) else "Stopped"
                frontend_status = "Running" if status.get("Frontend", {}).get("running", False) else "Stopped"
                tunnel_status = "Running" if self.tunnel_url else "Not started"
                
                self._update_gui_status(backend_status, frontend_status, tunnel_status)
                
                time.sleep(2)  # Update every 2 seconds
            except Exception as e:
                self.logger.error(f"Error in periodic status update: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _run_non_blocking(self):
        """Run launcher with non-blocking operations for smooth animations"""
        steps = [
            'Checking dependencies',
            'Setting up network', 
            'Installing dependencies',
            'Building frontend assets',
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
            'Building frontend assets',
            'Starting backend',
            'Starting frontend'
        ]
        
        try:
            # Reset progress mapper for new run
            if self.gui_mode:
                self.progress_mapper.reset()
            
            for idx, step in enumerate(steps):
                self.background_operations['current_step'] = idx
                self.background_operations['status'] = 'running'
                self.background_operations['progress'] = 0.0
                self.background_operations['error'] = None
                
                # Update GUI progress with unified mapping
                if self.gui_mode:
                    self._update_gui_progress(step, 0.0, "Processing...")
                
                success = False
                
                if step == 'Checking dependencies':
                    if self.skip_deps:
                        success = True
                        self.background_operations['progress'] = 1.0
                        if self.gui_mode:
                            self._update_gui_progress(step, 1.0, "Skipped")
                    else:
                        success = self._check_dependencies_async()
                elif step == 'Setting up network':
                    success = self._setup_network_async()
                elif step == 'Installing dependencies':
                    if self.skip_deps:
                        success = True
                        self.background_operations['progress'] = 1.0
                        if self.gui_mode:
                            self._update_gui_progress(step, 1.0, "Skipped")
                    else:
                        success = self._install_dependencies_async()
                elif step == 'Building frontend assets':
                    if self.dev_mode or self.skip_build:
                        success = True
                        self.background_operations['progress'] = 1.0
                        if self.gui_mode:
                            self._update_gui_progress(step, 1.0, "Skipped (dev mode or skip_build)")
                    else:
                        success = self._build_frontend_assets()
                elif step == 'Starting backend':
                    success = self._start_backend_async()
                    if success and self.gui_mode:
                        self._update_gui_status("Running", "Starting...", "Not started")
                    # Start tunnel if needed
                    if success and not self.lan_only:
                        self.background_operations['progress'] = 0.5
                        if self.gui_mode:
                            self._update_gui_progress(step, 0.7, "Starting tunnel...")
                        self.tunnel_url = self._start_tunnel()
                        self.background_operations['progress'] = 1.0
                        if self.gui_mode:
                            tunnel_status = "Running" if self.tunnel_url else "Failed"
                            self._update_gui_status("Running", "Starting...", tunnel_status)
                elif step == 'Starting frontend':
                    success = self._start_frontend_async()
                    if success and self.gui_mode:
                        tunnel_status = "Running" if self.tunnel_url else "Not started"
                        self._update_gui_status("Running", "Running", tunnel_status)
                
                if success:
                    self.background_operations['progress'] = 1.0
                    
                    # Show access info in GUI when completed
                    if idx == len(steps) - 1:
                        self.background_operations['status'] = 'completed'
                        self.logger.info("Launcher: All background operations completed, showing access info")
                        if self.gui_mode:
                            frontend_url = self.tunnel_url if self.tunnel_url else f"http://{self.host_ip}:{self.frontend_port}"
                            backend_url = self.tunnel_url if self.tunnel_url else f"http://{self.host_ip}:{self.backend_port}"
                            self.logger.info(f"Launcher: Determined URLs - frontend: {frontend_url}, backend: {backend_url}")
                            self._show_gui_access_info(frontend_url, backend_url)
                            self._update_gui_progress("Ready!", 1.0, "All services running")
                            self.logger.info("Launcher: GUI access info and progress updated")
                        else:
                            self.logger.info("Launcher: Not in GUI mode, skipping GUI access info")
                    else:
                        self.background_operations['status'] = 'running'
                else:
                    self.background_operations['status'] = 'failed'
                    self.background_operations['error'] = f"{step} failed"
                    if self.gui_mode:
                        self.gui_window.show_error(f"{step} failed")
                    break
                
                # Small delay between steps to allow UI updates
                time.sleep(0.1)
                
        except Exception as e:
            self.background_operations['status'] = 'failed'
            self.background_operations['error'] = str(e)
            if self.gui_mode:
                self.gui_window.show_error(f"Background operation error: {str(e)}")
    
    def _check_dependencies_async(self) -> bool:
        """Check dependencies with progress updates"""
        try:
            if self.gui_mode:
                self._update_gui_progress("Checking dependencies", 0.2, "Verifying project structure...")
            self.background_operations['progress'] = 0.2
            if not self.check_dependencies():
                return False
            if self.gui_mode:
                self._update_gui_progress("Checking dependencies", 1.0, "Dependencies verified")
            self.background_operations['progress'] = 1.0
            return True
        except Exception as e:
            self.logger.error(f"Async dependency check failed: {e}")
            return False
    
    def _setup_network_async(self) -> bool:
        """Setup network with progress updates"""
        try:
            if self.gui_mode:
                self._update_gui_progress("Setting up network", 0.3, "Configuring network settings...")
            self.background_operations['progress'] = 0.3
            if not self.setup_network():
                return False
            if self.gui_mode:
                self._update_gui_progress("Setting up network", 1.0, "Network configured")
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
                if self.gui_mode:
                    self._update_gui_progress("Installing dependencies", 1.0, "Dependencies already installed")
                self.background_operations['progress'] = 1.0
                return True
            
            # Install backend dependencies if needed
            if not backend_installed:
                if self.gui_mode:
                    self._update_gui_progress("Installing dependencies", 0.4, "Installing backend dependencies...")
                self.background_operations['progress'] = 0.4
                if not self._install_backend_deps_async():
                    return False
            
            # Install frontend dependencies if needed
            if not frontend_installed:
                if self.gui_mode:
                    self._update_gui_progress("Installing dependencies", 0.7, "Installing frontend dependencies...")
                self.background_operations['progress'] = 0.7
                if not self._install_frontend_deps_async():
                    return False
            
            if self.gui_mode:
                self._update_gui_progress("Installing dependencies", 1.0, "All dependencies installed successfully")
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
                if self.gui_mode:
                    self._update_gui_micro_progress(f"Installing backend dependencies (line {line_count})", progress)
                
                # Log to file
                self.logger.debug(f"[Backend Install] {line.strip()}")
            
            # Wait for completion
            backend_process.wait()
            
            if backend_process.returncode != 0:
                self.logger.error(f"Backend installation failed: {backend_process.returncode}")
                return False
            
            if self.gui_mode:
                self._update_gui_progress("Installing dependencies", 0.5, "Backend dependencies installed")
            
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
                if self.gui_mode:
                    self._update_gui_micro_progress(f"Installing frontend dependencies (line {line_count})", progress)
                
                # Log to file
                self.logger.debug(f"[Frontend Install] {line.strip()}")
            
            # Wait for completion
            frontend_process.wait()
            
            if frontend_process.returncode != 0:
                self.logger.error(f"Frontend installation failed: {frontend_process.returncode}")
                return False
            
            if self.gui_mode:
                self._update_gui_progress("Installing dependencies", 1.0, "All dependencies installed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Frontend installation failed: {e}")
            return False
    
    def _start_backend_async(self) -> bool:
        """Start backend with progress updates"""
        try:
            if self.gui_mode:
                self._update_gui_progress("Starting backend", 0.3, "Initializing backend server...")
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
                if self.gui_mode:
                    self._update_gui_progress("Starting backend", 0.8, "Waiting for backend to be ready...")
                self.background_operations['progress'] = 0.8
                if not self._wait_for_backend_async(self.host_ip, self.backend_port):
                    return False
            
            if self.gui_mode:
                self._update_gui_progress("Starting backend", 1.0, "Backend server ready")
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
            if self.gui_mode:
                self._update_gui_progress("Starting frontend", 0.3, "Initializing frontend server...")
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
                if self.gui_mode:
                    self._update_gui_progress("Starting frontend", 0.8, "Waiting for frontend to be ready...")
                if not self._wait_for_frontend_async(self.host_ip, self.frontend_port):
                    return False
            
            if self.gui_mode:
                self._update_gui_progress("Starting frontend", 1.0, "Frontend server ready")
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
  python launcher.py --gui                     # Desktop GUI window interface
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
    ui_group.add_argument("--gui", action="store_true", 
                         help="Launch with desktop GUI window (disables console output)")
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
        timeout=args.timeout,
        gui=args.gui
    )
    success = launcher.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 