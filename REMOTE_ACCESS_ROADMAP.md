# 🌐 Remote-Access Roadmap  
Cloudflare Tunnel Integration for the “2048 Bot” PWA


## 📑 Table of Contents
1. Conceptual Overview  
2. Phase-by-Phase Plan  
 2.0 Prerequisites  
 2.1 Quick Tunnel (no domain)  
 2.2 Named Tunnel (custom domain, production-ready)  
 2.3 PWA Origin Unification  
 2.4 Launcher Enhancement  
 2.5 Persistence & Self-Healing  
 2.6 Performance & Split DNS (optional)  
 2.7 Security & Observability  
 2.8 Packaging & Distribution  
3. Implementation Checklists  
4. Success Criteria  
5. Troubleshooting Quick-Reference  
6. Future Enhancements

---

## 1 Conceptual Overview
- Node  
 The machine that runs FastAPI and serves the PWA’s static assets.  
- cloudflared  
 Single-binary connector that dials Cloudflare’s edge over outbound TCP 7844
 and UDP 7844 (QUIC). No inbound firewall changes required.  
- Tunnel URL  
 The public hostname that browsers use. It is **HTTPS** and therefore satisfies
 the PWA secure-context requirement.

Two operating modes matter:
1. Quick Tunnel – zero account, disposable URL under  
 `*.trycloudflare.com`. Perfect for demos, CI jobs, and fallback when the named
 tunnel cannot start.  
2. Named Tunnel – persistent ID, automatic reconnection, you choose either  
 a domain you own or Cloudflare’s free `*.cfargotunnel.com`. Recommended for
 anyone who cares about a repeatable URL.

---

## 2 Phase-by-Phase Plan

### 2.0 Prerequisites  (≈1 hour, once per machine)
- Install cloudflared ≥ 2025.6.5

```bash
# Linux
curl -L https://pkg.cloudflare.com/cloudflared/install.sh | sudo bash
# macOS
brew install cloudflared
# Windows (PowerShell)
winget install --id Cloudflare.cloudflared
```

- Verify: `cloudflared --version`
- Python ≥ 3.9, `uvicorn`, and your existing FastAPI stack
- Optional: a registered domain in a Cloudflare account for the Named Tunnel

---

### 2.1 Phase 1 Quick Tunnel (Developer-centric)  Week 1
Goal: Any contributor can expose the 2048 Bot from a café Wi-Fi in <30 s.

```bash
cloudflared tunnel --url http://localhost:8000 --format json
```

Parse the first log line containing `"url":"https://x.trycloudflare.com"`.  
Use that URL in QR generation.  
Limitations to remember:
- Random subdomain each run
- Cloudflare currently caps Quick Tunnel at ≈100 concurrent HTTP requests
- Intended for non-production use

Keep Quick Tunnel as a hard fallback in all later automation.

---

### 2.2 Phase 2 Named Tunnel (Stable URL)  Week 1
Goal: A repeatable HTTPS origin suitable for bookmarks, service-worker cache,
and push notifications.

```
cloudflared tunnel login                       # browser OAuth dance
cloudflared tunnel create 2048-bot             # stores credentials file
cloudflared tunnel route dns 2048-bot \
        public-2048bot.cfargotunnel.com        # free, no domain needed
# OR, if you own a domain:
cloudflared tunnel route dns 2048-bot api.yourdomain.com
```

Create the config file:

```yaml
# ~/.cloudflared/config.yml
tunnel: 2048-bot
credentials-file: ~/.cloudflared/2048-bot.json
ingress:
  - hostname: public-2048bot.cfargotunnel.com     # or api.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

Validation:

```bash
cloudflared tunnel --config ~/.cloudflared/config.yml run --format json
curl https://public-2048bot.cfargotunnel.com/healthz
```

---

### 2.3 Phase 3 PWA Origin Unification  Week 1
Goal: The client never relies on `192.168.x.y` again.

Steps:
- In `vite.config.ts` set `base: '/'`.  
- Ensure every `fetch()` is either relative (`/api/...`) or hard-coded to
 `https://public-2048bot.cfargotunnel.com`.  
- Update service-worker cache names to avoid collision with any legacy
 `localhost` versions.  
- Manually test:  
 `http://127.0.0.1:8000` → still works in dev;  
 `https://public-2048bot.cfargotunnel.com` → works via tunnel.

---

### 2.4 Phase 4 Launcher Enhancement  Week 2
Goal: One command boots FastAPI, starts a tunnel, prints an installable QR.

```python
# launcher.py  (80-column wrapped, Flake8-clean)

import json
import shutil
import subprocess
from pathlib import Path

CLOUDFLARED = "cloudflared"
LOCAL_PORT = 8000


def ensure_cloudflared() -> None:
    if not shutil.which(CLOUDFLARED):
        raise SystemExit(
            "cloudflared not found. Install from "
            "https://developers.cloudflare.com/cloudflare-one/connections/"
            "connect-apps/install-and-setup/installation/"
        )


def start_fastapi():
    return subprocess.Popen(
        ["uvicorn", "app:api", "--port", str(LOCAL_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def start_tunnel(named: bool = True) -> str:
    cfg = Path.home() / ".cloudflared" / "config.yml"
    cmd = (
        [CLOUDFLARED, "tunnel", "--config", str(cfg), "--format", "json", "run"]
        if named and cfg.exists()
        else [
            CLOUDFLARED,
            "tunnel",
            "--url",
            f"http://localhost:{LOCAL_PORT}",
            "--format",
            "json",
        ]
    )
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        try:
            msg = json.loads(line)
            if "routeHostname" in msg:
                return msg["routeHostname"]
        except ValueError:
            continue
    raise RuntimeError("cloudflared did not yield a hostname")


def print_qr(url: str) -> None:
    try:
        import qrcode

        qr = qrcode.QRCode()
        qr.add_data(f"https://{url}")
        qr.make()
        qr.print_ascii(invert=True)
    except ImportError:
        print("pip install qrcode[pil] for QR output")
    print(f"Open → https://{url}")


def main() -> None:
    ensure_cloudflared()
    api = start_fastapi()
    try:
        hostname = start_tunnel(named=True)
    except Exception as err:
        print(f"Named tunnel failed: {err}\nFalling back to Quick Tunnel")
        hostname = start_tunnel(named=False)
    print_qr(hostname)
    api.wait()


if __name__ == "__main__":
    main()
```

Notes:
- `--format json` future-proofs the log parsing.
- Fallback to Quick Tunnel is automatic.
- The printed QR encodes the **hostname only**; scheme and path are implied
 by the phone’s default HTTPS handler.

---

### 2.5 Phase 5 Persistence & Self-Healing  Week 2
- Linux systemd

```ini
# /etc/systemd/system/2048-tunnel.service
[Unit]
Description=2048 Bot Cloudflare Tunnel
After=network-online.target

[Service]
User=ubuntu
ExecStart=/usr/bin/cloudflared tunnel \
  --config /home/ubuntu/.cloudflared/config.yml run
Restart=always
RestartSec=3
# raise ulimit so HTTP/2 multiplexing is never starved
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

- Windows:  
 `cloudflared service install --config %USERPROFILE%\.cloudflared\config.yml`

- macOS (Homebrew):  
 `brew services start cloudflared`

Self-healing facts:
- cloudflared maintains four simultaneous QUIC/TCP sessions; one loss is not a
 disconnect.  
- systemd restarts within 3 s if the binary exits non-zero.

---

### 2.6 Phase 6 Performance & Split-DNS (optional)  Week 3
Default: hair-pinning via Cloudflare adds ≈40 ms; acceptable for turn-based
games like 2048.

Optional optimisation:
- Run dnsmasq/Pi-hole on the node.  
- Serve a local A-record  
 `public-2048bot.cfargotunnel.com → 192.168.x.y`.  
- Mobile devices still see the certificate for the public name, so HTTPS stays
 valid.

---

### 2.7 Phase 7 Security & Observability  Week 3
Security toggles (all click-enabled in the CF dashboard):
- Universal SSL (already on)  
- Bot Fight Mode  
- WAF managed rules  
- Rate Limiting  
- Zero Trust Access → force Google / GitHub OAuth before FastAPI sees traffic

Observability:
- Start cloudflared with metrics endpoint  
 `--metrics 127.0.0.1:2000`  
 `curl http://127.0.0.1:2000/metrics` → Prometheus
- Deep bundle: `cloudflared diag > diag.zip`
- FastAPI already exports `/healthz` for uptime checks (add if missing).

---

### 2.8 Phase 8 Packaging & Distribution  Week 4
- Embed platform-specific cloudflared binaries in `bin/` or instruct users to
 install via package manager; both flows are documented.  
- Post-install message:

```
First-time tunnel setup:
1 cloudflared tunnel login
2 cloudflared tunnel create 2048-bot
3 cloudflared tunnel route dns 2048-bot public-2048bot.cfargotunnel.com
```

- Publish `launcher.py` entry-point as a `console_scripts` target in
 `pyproject.toml`.  
- Ship a `docker-compose.yml` for contributors who want container isolation;
 cloudflared can run side-car in the same compose file.

---

## 3 Implementation Checklists

### Development-Time
- [ ] cloudflared installed and on PATH  
- [ ] Quick Tunnel command verified on each dev OS  
- [ ] PWA origin audit: no plain-IP fetches remain  
- [ ] Service-worker cache bust performed

### Staging / Production
- [ ] Named Tunnel credentials file stored in the untracked secrets folder  
- [ ] `config.yml` syntax validated by `cloudflared tunnel --validate`  
- [ ] systemd/Windows service enabled and `systemctl status` is green  
- [ ] Metrics endpoint scraped by Prometheus or Vigil

### Disaster Recovery
- [ ] Quick Tunnel fallback tested (disconnect the machine from the internet
 then restore)  
- [ ] `cloudflared diag` produces a ZIP and retains <25 MB of disk  
- [ ] Single outbound firewall rule documented: TCP/UDP 7844 open

---

## 4 Success Criteria

Technical:
- Tunnel availability ≥ 99.9 % over a rolling 30 days
- Additional latency ≤ 100 ms p95 relative to LAN
- Mean Time To Recovery \(<30\text{ s}\) after unexpected tunnel exit

User Experience:
- First-run to public URL <5 min with a fresh laptop
- QR scan success rate ≥ 95 % across iOS 17 + Android 14
- PWA install prompt appears within 2 s of first load

---

## 5 Troubleshooting Quick-Reference

| Symptom | Command / Fix |
|---------|---------------|
| `cloudflared: command not found` | Re-install; confirm `which cloudflared` |
| Browser SSL error | Clock skew; sync system time |
| 525 Handshake Fail | Cloudflare paused SSL for the zone; unpause |
| `ERR_TUNNEL_CONNECTION_FAILED` | Corporate proxy blocks UDP 7844; add `--no-quic` |

Diagnostics:

```bash
cloudflared tunnel info 2048-bot
cloudflared tunnel run --loglevel debug
curl -v https://public-2048bot.cfargotunnel.com/healthz
```

---

## 6 Future Enhancements
- Multi-origin load balancing via Cloudflare LB
- Geographic routing to the closest tunnel node
- WebPush for move notifications, piggybacking on the same origin
- Slack / Discord webhook for `systemd` restart events
- Automated release in CI: start Quick Tunnel, run end-to-end tests against the
 public URL, then tear down

---

Implementation window: **4 weeks**, medium complexity, no public IP required.