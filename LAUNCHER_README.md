# 2048 Transformer Training Launcher

## Overview

The launcher script (`launcher.py`) is a Python-based development tool that automatically:
- Launches the backend server
- Waits for the backend to be ready
- Launches the frontend development server
- Configures both servers for LAN access
- Generates a QR code for easy mobile access
- Handles network discovery and configuration

## Features

- **Platform Agnostic**: Works on Windows, macOS, and Linux
- **Automatic Network Discovery**: Finds the best local IP address for LAN access
- **Mobile QR Code**: Generates a QR code for easy mobile access
- **Robust Error Handling**: Comprehensive error checking and recovery
- **Dependency Management**: Automatically installs required dependencies
- **Clean Shutdown**: Properly terminates all processes on exit

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

### What the Launcher Does

1. **Dependency Check**: Verifies Poetry, Node.js, and npm are installed
2. **Network Discovery**: Finds the best local IP address for LAN access
3. **Port Availability**: Checks that ports 8000 (backend) and 5173 (frontend) are available
4. **Backend Launch**: Starts the FastAPI server with external access enabled
5. **Frontend Launch**: Starts the Vite development server with LAN access
6. **QR Code Generation**: Creates a QR code for mobile access
7. **Status Display**: Shows access URLs and connection information

### Mobile Access

Once the launcher is running, you'll see:
- A QR code displayed in the terminal
- The QR code saved as `mobile_access_qr.png`
- Access URLs for both desktop and mobile

To access on mobile:
1. Ensure your phone is on the same Wi-Fi network
2. Scan the QR code with your phone's camera
3. The app will open in your mobile browser

### Network Configuration

The launcher automatically:
- Detects all available network interfaces
- Prioritizes common private network ranges (192.168.x.x, 10.x.x.x, 172.x.x.x)
- Configures CORS headers for secure cross-origin requests
- Sets up WebSocket connections for real-time updates

## Configuration

### Default Ports
- **Backend**: 8000
- **Frontend**: 5173

### Environment Variables

The launcher sets the following environment variables:
- `CORS_ORIGINS`: Comma-separated list of allowed origins for CORS
- `__BACKEND_URL__`: Backend URL for the frontend (set via Vite config)

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   - Stop any existing servers running on ports 8000 or 5173
   - Use `netstat -an | grep :8000` to check port usage

2. **Network Discovery Issues**
   - Ensure you're connected to a network
   - Check firewall settings
   - Try connecting to the IP address manually

3. **Mobile Access Issues**
   - Ensure phone and computer are on the same network
   - Check that the mobile browser supports the features used
   - Try accessing the URL directly instead of scanning the QR code

4. **Dependency Issues**
   - Ensure Poetry is properly installed and configured
   - Check that Node.js and npm are in your PATH
   - Try running the installation steps manually

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

### Logs and Debugging

The launcher provides colored output for different types of messages:
- 🟢 **Green**: Success messages
- 🔵 **Blue**: Information messages
- 🟡 **Yellow**: Warning messages
- 🔴 **Red**: Error messages

Process output is captured and can be viewed if needed for debugging.

## Security Considerations

- The launcher only allows connections from the local network
- CORS is configured to only allow specific origins
- No external internet access is required
- All data remains on the local network

## Advanced Usage

### Custom Configuration

You can modify the launcher script to:
- Change default ports
- Add custom network interfaces
- Modify CORS settings
- Add additional startup checks

### Integration with CI/CD

The launcher can be integrated into development workflows:
- Use as a pre-commit hook for testing
- Include in automated testing pipelines
- Use for demonstration environments

## Contributing

To contribute to the launcher:
1. Fork the repository
2. Create a feature branch
3. Test your changes across platforms
4. Submit a pull request

## License

This launcher is part of the 2048 Transformer Training project and follows the same license. 