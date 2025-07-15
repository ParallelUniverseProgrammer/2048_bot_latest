# Scripts Directory

This directory contains utility scripts for the 2048 AI Bot project.

## Utility Scripts

- **fix_windows_firewall.py** - Automatically configures Windows Firewall rules for mobile access
- **network_troubleshooter.py** - Comprehensive network troubleshooting tool for diagnosing mobile connectivity issues

## Running Scripts

To run the utility scripts, execute them from the project root directory:

```bash
python scripts/fix_windows_firewall.py
python scripts/network_troubleshooter.py
```

### Script Descriptions

#### fix_windows_firewall.py
- Automatically adds Windows Firewall rules for Python and the application ports
- Helps resolve mobile device connectivity issues on Windows systems
- Should be run as Administrator for full functionality

#### network_troubleshooter.py
- Advanced network diagnostics for mobile device connectivity
- Tests network interfaces, port availability, and routing
- Provides detailed troubleshooting information for connection issues
- Especially useful for diagnosing Spectrum router configuration problems 

## Generating PWA Icons

To automatically generate and overwrite all required PWA icons from a single source image (`generated-image.png` in `frontend/public/`), run:

```sh
python generate_pwa_icons.py
```

This will create/overwrite the following files in `frontend/public/`:
- pwa-192x192.png
- pwa-512x512.png
- apple-touch-icon.png
- favicon-16x16.png
- favicon.ico (multi-size)

Make sure you have the Pillow library installed:
```sh
pip install pillow
``` 