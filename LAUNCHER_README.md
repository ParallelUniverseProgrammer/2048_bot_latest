# 2048 Transformer Training Launcher

## Overview

The launcher script (`launcher.py`) is a robust Python-based development tool that automatically:
- Launches the backend server with comprehensive error monitoring
- Waits for the backend to be ready with detailed health checks
- Launches the frontend development server with real-time output monitoring
- Configures both servers for LAN access with intelligent port management
- Generates a QR code for easy mobile access
- Handles network discovery and configuration
- Provides comprehensive logging and error reporting
- Ensures proper cleanup of all resources

## Features

- **Platform Agnostic**: Works on Windows, macOS, and Linux
- **Automatic Network Discovery**: Finds the best local IP address for LAN access
- **Mobile QR Code**: Generates a QR code for easy mobile access
- **Robust Error Handling**: Comprehensive error checking, logging, and recovery
- **Process Monitoring**: Real-time monitoring of server output and health
- **Port Management**: Intelligent port conflict detection and resolution
- **Dependency Management**: Automatically installs required dependencies
- **Clean Shutdown**: Properly terminates all processes and frees ports on exit
- **Comprehensive Logging**: File-based logging with multiple verbosity levels
- **Status Monitoring**: Real-time display of process status and recent errors

## Prerequisites

Before using the launcher, ensure you have:

1. **Python 3.9+** installed
2. **Poetry** installed for Python dependency management
3. **Node.js** and **npm** installed for frontend development
4. **Git** (optional, for version control)

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

### Advanced Usage Options

```bash
# Development mode with Hot Module Replacement (HMR)
python launcher.py --dev

# Force kill processes using required ports (resolves port conflicts)
python launcher.py --force-ports

# Set logging verbosity
python launcher.py --log-level DEBUG

# Combine options
python launcher.py --dev --force-ports --log-level INFO
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--dev` | Run frontend in Vite dev mode with Hot Module Replacement |
| `--force-ports` | Force kill processes using required ports (8000, 5173, 5174) |
| `--log-level` | Set logging level: DEBUG, INFO, WARNING, ERROR (default: INFO) |

### What the Launcher Does

1. **Dependency Check**: Verifies Poetry, Node.js, and npm are installed
2. **Network Discovery**: Finds the best local IP address for LAN access
3. **Port Management**: 
   - Checks that ports 8000 (backend), 5173 (frontend), and 5174 (HMR) are available
   - Optionally kills conflicting processes with `--force-ports`
   - Provides detailed error messages for port conflicts
4. **Backend Launch**: 
   - Starts the FastAPI server with external access enabled
   - Monitors stdout/stderr for real-time error reporting
   - Waits for server to be ready with health checks
5. **Frontend Launch**: 
   - Starts the Vite development server with LAN access
   - Monitors build process and server output
   - Handles both dev and production modes
6. **QR Code Generation**: Creates a QR code for mobile access
7. **Status Monitoring**: 
   - Real-time display of process status
   - Shows recent errors and warnings
   - Automatic status updates every 30 seconds

### Mobile Access

Once the launcher is running, you'll see:
- A QR code displayed in the terminal
- The QR code saved as `mobile_access_qr.png`
- Access URLs for both desktop and mobile
- Real-time status of both servers

To access on mobile:
1. Ensure your phone is on the same Wi-Fi network
2. Scan the QR code with your phone's camera
3. The app will open in your mobile browser

### Network Configuration

The launcher automatically:
- Detects all available network interfaces
- Prioritizes common private network ranges (192.168.x.x, 10.x.x.x, 172.x.x.x)
- Avoids virtual adapters (WSL, Hyper-V, VPN) when possible
- Configures CORS headers for secure cross-origin requests
- Sets up WebSocket connections for real-time updates

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

2. **Network Discovery Issues**
   - Ensure you're connected to a network
   - Check firewall settings
   - Try connecting to the IP address manually
   - Check the log file for detailed network information

3. **Mobile Access Issues**
   - Ensure phone and computer are on the same network
   - Check that the mobile browser supports the features used
   - Try accessing the URL directly instead of scanning the QR code
   - Check the launcher logs for connection issues

4. **Dependency Issues**
   - Ensure Poetry is properly installed and configured
   - Check that Node.js and npm are in your PATH
   - Try running the installation steps manually
   - Check the log file for detailed error messages

5. **Process Termination Issues**
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

- The launcher only allows connections from the local network
- CORS is configured to only allow specific origins
- No external internet access is required
- All data remains on the local network
- Process monitoring doesn't expose sensitive information

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

This launcher is part of the 2048 Transformer Training project and follows the same license. 