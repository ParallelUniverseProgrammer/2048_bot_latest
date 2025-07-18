# 2048 Bot Training Launcher

## Overview

The launcher script (`launcher.py`) is a comprehensive Python-based deployment tool that automatically:
- Launches the backend server with comprehensive error monitoring
- Waits for the backend to be ready with detailed health checks
- Launches the frontend development server with real-time output monitoring
- **Creates Cloudflare Tunnels for internet access** (Quick or Named tunnels)
- Configures both servers for LAN access with intelligent port management
- Generates QR codes for easy mobile access
- Handles network discovery and configuration
- Provides comprehensive logging and error reporting
- Ensures proper cleanup of all resources
- **Supports multiple deployment modes** (development, production, cloud-first)
- **Modern console UI** with smooth progress animations and non-scrolling display

## Features

- **Platform Agnostic**: Works on Windows, macOS, and Linux
- **Automatic Network Discovery**: Finds the best local IP address for LAN access
- **Cloudflare Tunnel Integration**: Automatic HTTPS tunnel creation for internet access
- **Multiple Deployment Modes**: Development, production, LAN-only, and tunnel-only modes
- **Mobile QR Code**: Generates QR codes for easy mobile access
- **Robust Error Handling**: Comprehensive error checking, logging, and recovery
- **Process Monitoring**: Real-time monitoring of server output and health
- **Port Management**: Intelligent port conflict detection and resolution
- **Dependency Management**: Automatically installs required dependencies
- **Clean Shutdown**: Properly terminates all processes and frees ports on exit
- **Comprehensive Logging**: File-based logging with multiple verbosity levels
- **Status Monitoring**: Real-time display of process status and recent errors
- **Tunnel Fallback**: Automatic fallback from named to quick tunnels
- **Production Ready**: Named tunnels with persistent URLs and auto-reconnect
- **Modern Console UI**: Smooth progress animations, non-scrolling display, and QR-focused interface

## Prerequisites

Before using the launcher, ensure you have:

1. **Python 3.9+** installed
2. **Poetry** installed for Python dependency management
3. **Node.js** and **npm** installed for frontend development
4. **Git** (optional, for version control)
5. **cloudflared** (optional, for tunnel functionality - auto-downloaded if missing)

## Installation

1. Clone the repository and navigate to the project root
2. Run the launcher script:
   ```bash
   python launcher.py
   ```

The launcher will automatically:
- Install missing Python packages (`qrcode`, `netifaces`, `psutil`)
- Install backend dependencies via Poetry
- Install frontend dependencies via npm

## Usage

### Basic Usage

Simply run the launcher from the project root directory:

```bash
python launcher.py
```

This will start both servers with automatic Cloudflare Tunnel creation for internet access. The launcher now features a modern, non-scrolling console interface that displays smooth progress animations and prominently shows the QR code for mobile access.

### Deployment Modes

```bash
# Development mode (LAN only, hot reload)
python launcher.py --dev

# LAN access only (no tunnel, faster startup)
python launcher.py --lan-only

# Tunnel access only (cloud-first deployment)
python launcher.py --tunnel-only

# Production mode (named tunnel, optimized build)
python launcher.py --production
```

### Advanced Usage Options

```bash
# Development mode with Hot Module Replacement (HMR)
python launcher.py --dev

# Force kill processes using required ports (resolves port conflicts)
python launcher.py --force-ports

# Set logging verbosity
python launcher.py --log-level DEBUG

# Custom tunnel configuration
python launcher.py --tunnel-type named --tunnel-name my-2048-bot

# Custom ports
python launcher.py --port 9000 --frontend-port 3000

# Skip QR code generation
python launcher.py --no-qr

# Quiet mode (minimal output)
python launcher.py --quiet

# Development mode (verbose output)
python launcher.py --dev

# Combine options
python launcher.py --dev --force-ports --log-level INFO
```

### Command Line Options

#### Operation Modes
| Option | Description |
|--------|-------------|
| `--dev` | Development mode - LAN only with hot reload |
| `--lan-only` | LAN access only - no tunnel created (faster startup) |
| `--tunnel-only` | Tunnel access only - no LAN serving (cloud-first) |
| `--production` | Production mode - named tunnel with optimized build |

#### Tunnel Configuration
| Option | Description |
|--------|-------------|
| `--tunnel-type` | Tunnel type: `quick` (temporary) or `named` (persistent) |
| `--tunnel-name` | Named tunnel identifier (default: "2048-bot") |
| `--tunnel-domain` | Custom domain for named tunnel (optional) |
| `--no-tunnel-fallback` | Don't fallback to quick tunnel if named tunnel fails |

#### Network Configuration
| Option | Description |
|--------|-------------|
| `--port` | Backend server port (default: 8000) |
| `--frontend-port` | Frontend dev server port (default: 5173) |
| `--host` | Backend server host (default: 0.0.0.0) |
| `--force-ports` | Force kill processes using required ports |

#### UI and Output
| Option | Description |
|--------|-------------|
| `--no-qr` | Skip QR code generation (default: enabled) |
| `--no-color` | Disable colored output |
| `--quiet` | Suppress non-essential output |
| `--log-level` | Set logging level: DEBUG, INFO, WARNING, ERROR |

#### Advanced Options
| Option | Description |
|--------|-------------|
| `--skip-build` | Skip frontend build step (use existing dist/) |
| `--skip-deps` | Skip dependency checks |
| `--cloudflared-path` | Custom path to cloudflared binary |
| `--timeout` | Startup timeout in seconds (default: 30) |

### What the Launcher Does

1. **Dependency Check**: Verifies Poetry, Node.js, npm, and cloudflared are installed
2. **Network Discovery**: Finds the best local IP address for LAN access
3. **Port Management**: 
   - Checks that required ports are available
   - Optionally kills conflicting processes with `--force-ports`
   - Provides detailed error messages for port conflicts
4. **Backend Launch**: 
   - Starts the FastAPI server with external access enabled
   - Monitors stdout/stderr for real-time error reporting
   - Waits for server to be ready with health checks
5. **Tunnel Creation** (if enabled):
   - Creates Cloudflare Tunnel for internet access
   - Supports Quick Tunnel (temporary) or Named Tunnel (persistent)
   - Automatic fallback from named to quick tunnel
   - Waits for tunnel URL to be available
6. **Frontend Launch**: 
   - Starts the Vite development server with LAN access
   - Monitors build process and server output
   - Handles both dev and production modes
7. **QR Code Generation**: Creates QR codes for mobile access (LAN and/or tunnel)
8. **Status Monitoring**: 
   - Real-time display of process status
   - Shows recent errors and warnings
   - Automatic status updates every 30 seconds

### Mobile Access

Once the launcher is running, you'll see:
- **Modern console interface** with smooth progress animations
- **Prominent QR code display** in the center of the terminal
- QR codes saved as `mobile_access_qr.png`
- Access URLs for both desktop and mobile
- Real-time status of all servers and tunnels

#### LAN Access
To access on mobile via LAN:
1. Ensure your phone is on the same Wi-Fi network
2. Scan the LAN QR code with your phone's camera
3. The app will open in your mobile browser

#### Internet Access (Tunnel)
To access from anywhere via tunnel:
1. Scan the tunnel QR code with your phone's camera
2. The app will open in your mobile browser
3. Install as PWA for offline access
4. Share the tunnel URL with others for remote access

#### PWA Installation
- **iOS**: Tap the share button (📤) then "Add to Home Screen"
- **Android**: Tap "Install App" when prompted
- **Desktop**: Click the install icon in the browser address bar

### Network Configuration

The launcher automatically:
- Detects all available network interfaces
- Prioritizes common private network ranges (192.168.x.x, 10.x.x.x, 172.x.x.x)
- Avoids virtual adapters (WSL, Hyper-V, VPN) when possible
- Configures CORS headers for secure cross-origin requests
- Sets up WebSocket connections for real-time updates

### Tunnel Configuration

The launcher supports two types of Cloudflare Tunnels:

#### Quick Tunnel (Default)
- **No account required** - works immediately
- **Temporary URL** - changes each time you restart
- **Perfect for demos** and quick testing
- **Automatic fallback** if named tunnel fails

#### Named Tunnel (Production)
- **Requires Cloudflare account** setup
- **Persistent URL** - same URL every time
- **Production ready** with auto-reconnect
- **Custom domains** supported
- **Better performance** and reliability

To set up a named tunnel:
```bash
# Login to Cloudflare (one-time setup)
cloudflared tunnel login

# Create a named tunnel
cloudflared tunnel create 2048-bot

# Route DNS (optional - uses cfargotunnel.com by default)
cloudflared tunnel route dns 2048-bot your-domain.com

# Use with launcher
python launcher.py --tunnel-type named
```

## Enhanced Features

### Process Monitoring

The launcher provides comprehensive process monitoring:

```
📊 Process Status
==============================
🟢 Backend: Running (PID: 1234)
🟢 Frontend: Running (PID: 5678)
   Recent errors:
   - Connection timeout on port 8000
```

### Error Reporting

When servers fail to start, the launcher shows:
- Recent error messages from the failed process
- Detailed dependency installation errors
- Network configuration issues
- Port conflict resolution

### Logging System

All operations are logged to `launcher.log` with:
- Timestamps and log levels
- Process output and errors
- Network discovery results
- Cleanup operations

### Port Management

The launcher handles port conflicts intelligently:
- **Detection**: Checks if ports are in use before starting
- **Resolution**: Optionally kills conflicting processes
- **Cleanup**: Ensures ports are freed on exit
- **Fallback**: Can find alternative ports if needed

## Configuration

### Default Ports
- **Backend**: 8000
- **Frontend**: 5173
- **HMR (Hot Module Replacement)**: 5174

### Tunnel URLs
- **Quick Tunnel**: `https://[random].trycloudflare.com`
- **Named Tunnel**: `https://[name].cfargotunnel.com` or custom domain

### Environment Variables

The launcher sets the following environment variables:
- `CORS_ORIGINS`: Comma-separated list of allowed origins for CORS
- `__BACKEND_URL__`: Backend URL for the frontend (set via Vite config)

### Logging Configuration

Log levels available:
- **DEBUG**: Detailed debugging information
- **INFO**: General information about operations
- **WARNING**: Warning messages about potential issues
- **ERROR**: Error messages and failures

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Use the force-ports option
   python launcher.py --force-ports
   
   # Or manually check what's using the port
   netstat -an | findstr :8000  # Windows
   netstat -an | grep :8000     # Linux/macOS
   ```

2. **Tunnel Creation Fails**
   ```bash
   # Check if cloudflared is installed
   cloudflared --version
   
   # Try LAN-only mode
   python launcher.py --lan-only
   
   # Check firewall settings for outbound connections
   ```

3. **Named Tunnel Not Working**
   ```bash
   # Verify Cloudflare login
   cloudflared tunnel list
   
   # Check tunnel configuration
   cat ~/.cloudflared/config.yml
   
   # Use quick tunnel fallback
   python launcher.py --tunnel-type quick
   ```

4. **Network Discovery Issues**
   - Ensure you're connected to a network
   - Check firewall settings
   - Try connecting to the IP address manually
   - Check the log file for detailed network information

5. **Mobile Access Issues**
   - Ensure phone and computer are on the same network (for LAN access)
   - Check that the mobile browser supports the features used
   - Try accessing the URL directly instead of scanning the QR code
   - Check the launcher logs for connection issues
   - For tunnel access, ensure internet connectivity on mobile device

6. **Dependency Issues**
   - Ensure Poetry is properly installed and configured
   - Check that Node.js and npm are in your PATH
   - Try running the installation steps manually
   - Check the log file for detailed error messages

7. **Process Termination Issues**
   - The launcher now has enhanced cleanup
   - Check `launcher.log` for cleanup details
   - Use `--force-ports` to resolve port conflicts
   - Restart your terminal if processes are stuck

### Manual Setup

If the launcher fails, you can run the servers manually:

1. **Backend**:
   ```bash
   cd backend
   poetry install
   poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev -- --host 0.0.0.0 --port 5173
   ```

### Debugging with Logs

The launcher creates detailed logs in `launcher.log`:
```bash
# View recent logs
tail -f launcher.log

# Search for errors
grep ERROR launcher.log

# Check process output
grep "\[Backend\]" launcher.log
```

### Logs and Debugging

The launcher provides colored output for different types of messages:
- 🟢 **Green**: Success messages
- 🔵 **Blue**: Information messages
- 🟡 **Yellow**: Warning messages
- 🔴 **Red**: Error messages

Process output is captured and logged for debugging purposes.

## Security Considerations

- **LAN Mode**: Only allows connections from the local network
- **Tunnel Mode**: Uses Cloudflare's secure HTTPS tunnels with automatic SSL
- **CORS**: Configured to only allow specific origins
- **Data Privacy**: All training data remains on your local machine
- **Process Monitoring**: Doesn't expose sensitive information
- **Tunnel Security**: Cloudflare tunnels provide enterprise-grade security

## Advanced Usage

### Custom Configuration

You can modify the launcher script to:
- Change default ports
- Add custom network interfaces
- Modify CORS settings
- Add additional startup checks
- Customize logging behavior

### Integration with CI/CD

The launcher can be integrated into development workflows:
- Use as a pre-commit hook for testing
- Include in automated testing pipelines
- Use for demonstration environments
- Monitor server health in production-like environments

### Development Workflow

For development, use the `--dev` flag:
```bash
python launcher.py --dev --log-level DEBUG
```

This provides:
- Traditional verbose output (instead of the clean QR interface)
- Hot Module Replacement for frontend changes
- Detailed logging for debugging
- Real-time error reporting
- Faster development iteration

## File Structure

The launcher creates and manages:
- `launcher.log` - Comprehensive operation log
- `mobile_access_qr.png` - QR code for mobile access
- `frontend/vite.config.temp.ts` - Temporary Vite configuration (auto-removed)

## Contributing

To contribute to the launcher:
1. Fork the repository
2. Create a feature branch
3. Test your changes across platforms
4. Ensure all new features include proper error handling and logging
5. Submit a pull request

## License

This launcher is part of the 2048 Bot Training project and follows the same license. 