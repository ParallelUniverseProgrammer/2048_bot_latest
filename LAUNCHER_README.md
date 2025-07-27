# 2048 Bot Training Launcher

A modern, feature-rich launcher for the 2048 Bot Training system with **ultra-smooth animations** and professional terminal UI.

## Overview

The launcher provides a **smooth, professional experience** for starting the 2048 Bot Training system, featuring:

- **Ultra-smooth 60fps progress animations** with precise timing
- **Non-blocking architecture** - UI never freezes during operations
- **Enhanced terminal compatibility** with Unicode/ASCII fallbacks
- **Background operations** - All heavy work runs in background threads
- **Professional visual effects** - Smooth easing, proper timing, polished animations

## Features

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
# Start with QR code for mobile access
python launcher.py

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
```

## Technical Architecture

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

## Compatibility

- ✅ **Windows** - PowerShell, Command Prompt, Windows Terminal
- ✅ **macOS** - Terminal, iTerm2, Alacritty
- ✅ **Linux** - GNOME Terminal, Konsole, Alacritty
- ✅ **Basic terminals** - ASCII fallbacks for limited environments
- ✅ **Remote terminals** - SSH, WSL, Docker containers

## Requirements

- Python 3.8+
- Poetry (for backend dependencies)
- Node.js 16+ and npm (for frontend dependencies)
- Cloudflared (optional, for tunnel access)

## Installation

1. Clone the repository
2. Install Python dependencies: `poetry install`
3. Install frontend dependencies: `cd frontend && npm install`
4. Run the launcher: `python launcher.py`

---

**The launcher now provides a truly professional, ultra-smooth experience that rivals modern desktop applications!** 🚀 