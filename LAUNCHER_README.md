# 2048 Bot Training Launcher

A modern, feature-rich launcher for the 2048 Bot Training system with **ultra-smooth animations**, **desktop GUI support**, and professional terminal UI.

## Overview

The launcher provides a **smooth, professional experience** for starting the 2048 Bot Training system, featuring:

- **Desktop GUI Mode** - Modern desktop window with system tray integration
- **Ultra-smooth 60fps progress animations** with precise timing
- **Non-blocking architecture** - UI never freezes during operations
- **Enhanced terminal compatibility** with Unicode/ASCII fallbacks
- **Background operations** - All heavy work runs in background threads
- **Professional visual effects** - Smooth easing, proper timing, polished animations

## Features

### 🖥️ **Desktop GUI Mode**
- **Modern desktop window** with CustomTkinter for native appearance
- **Service management controls** - Stop, restart, and view logs directly from GUI
- **One-click URL copying** with visual feedback and clipboard integration
- **Real-time status indicators** with animated status lights
- **Progress tracking** with smooth progress bars and step indicators
- **Cross-platform compatibility** - Works on Windows, macOS, and Linux

**Note**: System tray integration is implemented but may have compatibility issues on some platforms. The desktop window provides full functionality for service management and monitoring.

### 🎯 **Ultra-Smooth Animations**
- **60fps animation loop** with precise `time.perf_counter()` timing
- **Double-buffered rendering** - No screen flickering or visual artifacts
- **Smooth progress interpolation** with natural easing (0.12 factor)
- **Micro-progress system** for detailed operation feedback
- **Content change detection** - Only re-renders when necessary

### 🚀 **Non-Blocking Architecture**
- **Background operations** - Dependency installation, server startup, network setup
- **Smooth UI updates** - Main thread runs at consistent 60fps
- **Real-time progress** - Live feedback during all operations
- **Professional feel** - No stuttering or freezing during heavy operations

### 🎨 **Modern Console UI**
- **Smooth 60fps animations** without interruption
- **Smart screen management** using cursor positioning instead of screen clearing
- **Enhanced terminal compatibility** with automatic Unicode/ASCII detection
- **Professional visual effects** including:
  - Animated spinners and progress bars
  - Typing effects for status messages
  - Pulse effects for error messages
  - Smooth transitions between states

### 🌐 **Network & Access**
- **Automatic port management** with conflict resolution
- **Network discovery** for optimal IP address selection
- **Cloudflare Tunnel integration** for public access
- **QR code generation** for mobile access
- **Multiple access methods** (LAN, tunnel, localhost)

### 🔧 **Dependency Management**
- **Poetry integration** for Python dependencies
- **npm integration** for frontend dependencies
- **Automatic dependency checking** and installation
- **Progress tracking** for installation operations

## Enhanced Features

### **Desktop GUI Mode**
Launch with `python launcher.py --gui` for a modern desktop experience:

- **Professional Window Interface** - Clean, modern window with project branding
- **Service Management** - One-click stop/restart services and view logs
- **URL Management** - Copy access URLs with one click and visual feedback
- **Status Monitoring** - Real-time status indicators for backend, frontend, and tunnel
- **Progress Tracking** - Smooth progress bars with step-by-step feedback
- **Error Handling** - Clear error display with actionable information

**Note**: System tray integration is implemented but may have compatibility issues on some platforms. The desktop window provides full functionality for service management and monitoring.

### **Ultra-Smooth Animation System**
The launcher features a **professional-grade animation system** that rivals modern desktop applications:

- **Precise 60fps timing** using `time.perf_counter()` for microsecond accuracy
- **Double-buffered rendering** eliminates screen flickering and visual artifacts
- **Content change detection** with hash-based diffing for efficient updates
- **Smooth progress interpolation** with natural easing curves
- **Micro-progress bars** for detailed operation feedback
- **Animation state management** for smooth transitions

### **Non-Blocking Architecture**
All heavy operations run in background threads, ensuring the UI remains **always responsive**:

- **Background dependency installation** - Poetry and npm operations
- **Background server startup** - Backend and frontend servers
- **Background network setup** - Port checking and IP discovery
- **Real-time progress updates** - Live feedback during all operations
- **Smooth UI animations** - Main thread never blocks

### **Terminal Compatibility**
Works seamlessly across all terminal environments:

- **Unicode support detection** with graceful ASCII fallbacks
- **Color support detection** with dynamic color scheme adaptation
- **Responsive layout** that adapts to any terminal size
- **Animation fallbacks** for basic terminals
- **Cross-platform compatibility** (Windows, macOS, Linux)

## Usage

### Basic Usage
```bash
# Start with QR code for mobile access (console mode)
python launcher.py

# Launch with desktop GUI window
python launcher.py --gui

# LAN-only mode (no tunnel)
python launcher.py --lan-only

# Development mode with verbose output
python launcher.py --dev-mode
```

### Advanced Options
```bash
# Custom ports
python launcher.py --backend-port 8001 --frontend-port 3000

# Named tunnel
python launcher.py --tunnel-type named --tunnel-name my-2048-bot

# Skip dependency installation
python launcher.py --skip-deps

# Skip build process
python launcher.py --skip-build

# Force kill processes using required ports
python launcher.py --force-ports
```

### GUI Mode Features
When using `--gui` mode, you get access to:

- **Desktop Window** - Professional window interface with project branding
- **Service Controls** - Stop, restart, and view logs directly from the GUI
- **URL Copying** - One-click copying of access URLs with visual feedback
- **Status Monitoring** - Real-time status indicators for all services
- **Progress Tracking** - Smooth progress bars with detailed step information

**Note**: System tray integration is implemented but may have compatibility issues on some platforms. The desktop window provides full functionality for service management and monitoring.

## Technical Architecture

### **GUI Architecture**
```python
# Modern CustomTkinter-based GUI
class GUIWindow:
    def __init__(self, logger: Logger):
        # Configure for vibrant PC-forward appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create compact main window
        self.window = ctk.CTk()
        self.window.title("2048 Bot Launcher")
        self.window.geometry("450x420")
        
        # Service management and status monitoring
        self._setup_service_controls()
```

### **Animation System**
```python
# Ultra-precise 60fps timing
frame_interval = 1.0 / 60.0  # 16.67ms
current_time = time.perf_counter()

# Smooth progress interpolation
easing_factor = 0.12  # Natural movement
progress_diff = self.target_progress - self.current_progress
self.current_progress += progress_diff * easing_factor
```

### **Non-Blocking Operations**
```python
# Background operations thread
background_thread = threading.Thread(target=self._run_background_operations)
background_thread.start()

# Main thread - always smooth 60fps
while self.background_operations['status'] != 'completed':
    self.console_ui.update_progress(step, progress)
    time.sleep(0.016)  # 60fps - NO BLOCKING
```

### **Double-Buffered Rendering**
```python
# Content change detection
current_content_hash = hash(f"{self.current_step}{self.current_progress:.3f}...")

# Cursor positioning instead of screen clearing
if not hasattr(self, '_first_render'):
    self.clear_screen()  # Only on first render
else:
    self.move_cursor(1, 1)  # Smooth updates
```

## Performance Features

- ✅ **60fps animations** - Consistent frame rate with precise timing
- ✅ **No screen flickering** - Double-buffered rendering eliminates visual artifacts
- ✅ **Smooth progress bars** - Natural easing with overshoot prevention
- ✅ **Responsive UI** - Main thread never blocks, always responsive
- ✅ **Background processing** - Heavy operations don't affect animations
- ✅ **Memory efficient** - Content change detection prevents unnecessary renders
- ✅ **CPU optimized** - Ultra-precise sleep timing for better responsiveness
- ✅ **Desktop GUI** - Native window interface with system tray integration

## Compatibility

- ✅ **Windows** - PowerShell, Command Prompt, Windows Terminal, GUI mode
- ✅ **macOS** - Terminal, iTerm2, Alacritty, GUI mode
- ✅ **Linux** - GNOME Terminal, Konsole, Alacritty, GUI mode
- ✅ **Basic terminals** - ASCII fallbacks for limited environments
- ✅ **Remote terminals** - SSH, WSL, Docker containers
- ✅ **Desktop environments** - Full GUI support with service management

**Note**: System tray integration is implemented but may have compatibility issues on some platforms. The desktop window provides full functionality for service management and monitoring.

## Requirements

- Python 3.8+
- Poetry (for backend dependencies)
- Node.js 16+ and npm (for frontend dependencies)
- Cloudflared (optional, for tunnel access)
- CustomTkinter (for GUI mode) - automatically installed if missing

## Installation

1. Clone the repository
2. Install Python dependencies: `poetry install`
3. Install frontend dependencies: `cd frontend && npm install`
4. Run the launcher: `python launcher.py` or `python launcher.py --gui`

## GUI Mode Dependencies

The GUI mode automatically installs required dependencies:
- `customtkinter` - Modern GUI framework
- `pillow` - Image processing for icons and QR codes

**Note**: System tray integration (`pystray`) is implemented but may have compatibility issues on some platforms. The desktop window provides full functionality for service management and monitoring.

If you encounter issues with GUI mode, manually install:
```bash
pip install customtkinter pillow
```

---

**The launcher now provides a truly professional, ultra-smooth experience that rivals modern desktop applications with both console and GUI modes!** 🚀 