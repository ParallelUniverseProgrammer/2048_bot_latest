# 🌐 Remote Access Roadmap: Cloudflare Tunnel Integration

> A comprehensive implementation plan to transform the 2048 Bot from a LAN-only development tool into an internet-reachable, HTTPS-secured service using Cloudflare Tunnel.

## 📋 Overview

This roadmap provides a complete playbook for adding Cloudflare Tunnel to the 2048 Bot, enabling:
- **Internet Accessibility** - Access from anywhere via HTTPS
- **Zero Configuration** - Automatic tunnel setup and QR code generation
- **Production Ready** - Named tunnels with auto-reconnect and monitoring
- **Mobile PWA Support** - Seamless installation and offline functionality

## 🎯 Implementation Phases

### Phase 1: Foundation Setup (Week 1)
**Goal**: Establish Cloudflare Tunnel infrastructure

#### 1.1 Vocabulary & Architecture
- **Node**: The machine hosting FastAPI + PWA static files
- **Tunnel URL**: Public hostname (e.g., `https://demo.example.com` or `https://orange-lion.trycloudflare.com`)
- **cloudflared**: Single-binary Cloudflare connector

#### 1.2 Tunnel Strategy Decision
**Quick Tunnel** (Development/Demo):
- Command: `cloudflared tunnel --url http://localhost:8000`
- Pros: Zero Cloudflare account, <10s setup
- Cons: Random URL each run, 200 concurrent request cap, no uptime SLA

**Named Tunnel** (Production):
- Persistent ID + credentials JSON, auto-reconnect, no request cap
- Custom subdomain support
- Requires free Cloudflare account + domain ownership

**Recommendation**: Implement both with Named Tunnel as primary, Quick Tunnel as fallback.

#### 1.3 Bootstrap Implementation
```bash
# Step 1: Install cloudflared (>= 2025.6.5 for UDP engine)
# Linux
curl -L https://pkg.cloudflare.com/cloudflared/install.sh | sudo bash

# macOS
brew install cloudflared

# Windows
# Download cloudflared-windows-amd64.exe and add to PATH

# Step 2: Authenticate
cloudflared tunnel login

# Step 3: Create named tunnel
cloudflared tunnel create 2048-bot

# Step 4: Map hostname (replace with your domain)
cloudflared tunnel route dns 2048-bot api.yourdomain.com
```

#### 1.4 Configuration File
```yaml
# ~/.cloudflared/config.yml
tunnel: 2048-bot
credentials-file: ~/.cloudflared/2048-bot.json

ingress:
  - hostname: api.yourdomain.com
    service: http://localhost:8000     # FastAPI port
  - service: http_status:404           # catch-all
```

### Phase 2: PWA Origin Unification (Week 1)
**Goal**: Ensure single HTTPS origin for all PWA functionality

#### 2.1 Frontend Updates
- Update `vite.config.ts` to use relative base URL (`/`)
- Ensure all `fetch()` calls use relative paths (`/api/v1/...`) or hardcoded HTTPS URLs
- Remove any LAN IP literals from service worker
- Verify localhost functionality when accessing `http://127.0.0.1:8000`

#### 2.2 Service Worker Updates
```typescript
// Ensure all API calls are relative or HTTPS
const apiCall = async (endpoint: string) => {
  const response = await fetch(`/api/${endpoint}`, {
    // ... options
  });
  return response.json();
};
```

### Phase 3: Enhanced Launcher Integration (Week 2)
**Goal**: Integrate tunnel management into the existing launcher

#### 3.1 Launcher Enhancement
```python
# launcher.py additions
import json
import re
import subprocess
import shutil
from pathlib import Path

CLOUDFLARED_BIN = "cloudflared"
LOCAL_PORT = 8000

def ensure_cloudflared():
    """Verify cloudflared is available on PATH"""
    if not shutil.which(CLOUDFLARED_BIN):
        raise SystemExit("cloudflared not on PATH – install from https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/")

def start_tunnel(named=True) -> str:
    """Start Cloudflare tunnel and return public URL"""
    if named and Path.home().joinpath(".cloudflared/config.yml").exists():
        proc = subprocess.Popen(
            [
                CLOUDFLARED_BIN,
                "tunnel",
                "--config",
                str(Path.home() / ".cloudflared" / "config.yml"),
                "--format",
                "json",
                "run",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
    else:  # Quick Tunnel fallback
        proc = subprocess.Popen(
            [
                CLOUDFLARED_BIN,
                "tunnel",
                "--url",
                f"http://localhost:{LOCAL_PORT}",
                "--format",
                "json",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )

    # Wait for first JSON log line containing "routeHostname"
    for line in proc.stdout:
        try:
            msg = json.loads(line)
            if "routeHostname" in msg:
                return msg["routeHostname"]
        except Exception:
            continue
    raise RuntimeError("tunnel failed to start")

def display_qr_code(url: str):
    """Generate and display QR code for mobile access"""
    try:
        import qrcode
        qr = qrcode.QRCode()
        qr.add_data(f"https://{url}")
        qr.print_ascii(invert=True)
    except ImportError:
        print(f"Install qrcode: pip install qrcode[pil]")
        print(f"Mobile access URL: https://{url}")

def main():
    """Enhanced main function with tunnel support"""
    ensure_cloudflared()
    
    # Start FastAPI backend
    api = start_fastapi()
    
    # Start tunnel
    try:
        public_url = start_tunnel(named=True)
        print(f"✓ Tunnel live at https://{public_url}")
        display_qr_code(public_url)
    except Exception as e:
        print(f"⚠ Named tunnel failed: {e}")
        print("🔄 Falling back to Quick Tunnel...")
        public_url = start_tunnel(named=False)
        print(f"✓ Quick tunnel live at https://{public_url}")
        display_qr_code(public_url)
    
    # Wait for API to complete
    api.wait()
```

#### 3.2 QR Code Generation
- Encode `https://api.yourdomain.com` (not IP addresses)
- Optional: Append `/#install` for PWA installation prompt
- Use `qrcode` or `segno` library with ANSI rendering for headless servers

### Phase 4: Service Persistence (Week 2)
**Goal**: Make tunnel survive system restarts and run as background service

#### 4.1 Linux (systemd)
```ini
# /etc/systemd/system/2048-bot-tunnel.service
[Unit]
Description=2048 Bot Cloudflare Tunnel
After=network-online.target

[Service]
User=ubuntu
ExecStart=/usr/bin/cloudflared tunnel --config /home/ubuntu/.cloudflared/config.yml run
Restart=always
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

#### 4.2 Windows Service
```powershell
cloudflared service install --config C:\Users\me\.cloudflared\config.yml
```

#### 4.3 macOS (Homebrew)
```bash
brew services start cloudflared
```

### Phase 5: Performance Optimization (Week 3)
**Goal**: Optimize for LAN performance and reduce latency

#### 5.1 Split-Horizon DNS (Optional)
- Run dnsmasq or Pi-hole on the node
- Advertise via DHCP option 6
- Override `api.yourdomain.com → 192.168.x.y`
- PWA continues using identical HTTPS origin

#### 5.2 Hair-Pin Routing
- Default behavior: devices loop out to Cloudflare and back (~40ms extra)
- Acceptable for most use cases

### Phase 6: Security & Monitoring (Week 3)
**Goal**: Implement security features and observability

#### 6.1 Security Features
- **HTTPS**: Auto-renewed by Cloudflare (no local TLS files)
- **Bot Protection**: Toggle "Bot Fight Mode" in Cloudflare dashboard
- **WAF**: Web Application Firewall rules
- **Rate Limiting**: Per-hostname configuration
- **Zero Trust**: Optional Google Workspace/GitHub authentication

#### 6.2 Monitoring Implementation
```python
# Add to launcher for metrics endpoint
def start_tunnel_with_metrics(named=True) -> str:
    """Start tunnel with metrics endpoint"""
    metrics_args = ["--metrics", "127.0.0.1:2000"] if named else []
    
    # ... existing tunnel logic with metrics_args
```

#### 6.3 Observability Tools
- **Real-time metrics**: `curl http://127.0.0.1:2000/metrics` (Prometheus format)
- **Diagnostics**: `cloudflared diag` for comprehensive troubleshooting
- **Logs**: JSON-formatted logs for machine parsing

### Phase 7: Self-Healing & Resilience (Week 4)
**Goal**: Implement robust error handling and recovery

#### 7.1 Failure Modes
- **Tunnel Exit**: systemd restarts in 5s (configurable)
- **Network Flaps**: cloudflared maintains 4 long-lived TCP + QUIC sessions
- **Quick Tunnel Fallback**: Automatic fallback when named tunnel fails

#### 7.2 Recovery Strategies
```python
def resilient_tunnel_start():
    """Start tunnel with automatic fallback"""
    try:
        return start_tunnel(named=True)
    except Exception as e:
        print(f"Named tunnel failed: {e}")
        print("Falling back to Quick Tunnel...")
        return start_tunnel(named=False)
```

### Phase 8: Distribution & Packaging (Week 4)
**Goal**: Package everything for easy deployment

#### 8.1 Bundle cloudflared
- Include platform-specific `cloudflared` binary in `bin/` directory
- Add to `pyproject.toml` dependencies
- Post-install hook for tunnel setup

#### 8.2 Installation Script
```python
# setup.py or pyproject.toml post-install
def post_install_setup():
    """Guide user through tunnel setup"""
    config_path = Path.home() / ".cloudflared" / "2048-bot.json"
    if not config_path.exists():
        print("🔧 First-time setup required:")
        print("1. Run: cloudflared tunnel login")
        print("2. Run: cloudflared tunnel create 2048-bot")
        print("3. Run: cloudflared tunnel route dns 2048-bot api.yourdomain.com")
```

#### 8.3 Documentation
- Single outbound port requirement: TCP 7844 + UDP 7844 (QUIC)
- Domain ownership verification process
- Troubleshooting guide

## 🚀 Implementation Checklist

### Pre-Implementation
- [ ] Cloudflare account created
- [ ] Domain ownership verified
- [ ] cloudflared ≥ 2025.6.5 installed on target OSes
- [ ] Named tunnel created and configured

### Development
- [ ] Named tunnel config.yml committed to repo (outside Docker)
- [ ] Launcher updated with JSON log parsing
- [ ] QR code generation implemented
- [ ] PWA uses single HTTPS origin
- [ ] Service worker cache warmed on first run

### Deployment
- [ ] systemd/Windows service enabled with `Restart=always`
- [ ] Monitoring hooks expose metrics
- [ ] `cloudflared diag` artifacts collected
- [ ] Quick Tunnel fallback tested

### Documentation
- [ ] Disaster recovery plan documented
- [ ] Alert system configured
- [ ] User guide updated
- [ ] Troubleshooting section added

## 📊 Success Metrics

### Technical Metrics
- **Uptime**: 99.9% tunnel availability
- **Latency**: <100ms additional latency over direct LAN
- **Recovery Time**: <30s from tunnel failure to Quick Tunnel fallback
- **Mobile Compatibility**: 100% PWA functionality on iOS/Android

### User Experience Metrics
- **Setup Time**: <5 minutes from fresh install to internet access
- **QR Code Success Rate**: >95% successful mobile installations
- **Cross-Platform**: Seamless experience across Windows, macOS, Linux

## 🔧 Troubleshooting Guide

### Common Issues
1. **cloudflared not found**: Install from official Cloudflare repository
2. **Authentication failed**: Run `cloudflared tunnel login` and follow browser flow
3. **Tunnel won't start**: Check config.yml syntax and credentials file
4. **PWA not working**: Verify single origin configuration and service worker

### Diagnostic Commands
```bash
# Check tunnel status
cloudflared tunnel info 2048-bot

# View logs
cloudflared tunnel --config ~/.cloudflared/config.yml run

# Generate diagnostics
cloudflared diag

# Test connectivity
curl -I https://api.yourdomain.com
```

## 🎯 Future Enhancements

### Advanced Features
- **Load Balancing**: Multiple tunnel endpoints
- **Geographic Routing**: Route users to nearest tunnel
- **Custom Domains**: Support for vanity URLs
- **Analytics**: Detailed usage metrics and insights

### Integration Opportunities
- **CI/CD**: Automated tunnel deployment
- **Monitoring**: Integration with Prometheus/Grafana
- **Alerting**: Slack/Discord notifications for tunnel status
- **Backup**: Automatic tunnel configuration backup

---

**Timeline**: 4 weeks for complete implementation
**Effort**: Medium complexity, high impact
**Dependencies**: Cloudflare account, domain ownership, cloudflared binary 