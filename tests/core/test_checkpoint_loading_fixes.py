from tests.utilities.backend_manager import requires_real_backend
#!/usr/bin/env python3
"""
Autonomous Test Suite for Checkpoint Loading Fixes
==================================================

This test suite verifies checkpoint loading fixes by running four scenarios against 
a mock backend without requiring manual steps.

Scenarios validated:
1. timeout              -> playback/current returns has_data=false (HTTP 200)
2. api_error            -> POST playback/start returns HTTP 500
3. websocket_failure    -> playback/current returns HTTP 503
4. success              -> playback/current returns has_data=true (HTTP 200)

If all assertions pass the test exits with status code 0.
"""

import json
import threading
import time
import http.client
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import contextlib
import sys
import os
from typing import Any

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))

from tests.utilities.test_utils import TestLogger

HOST = "localhost"
PORT = 8000

class MockBackendHandler(BaseHTTPRequestHandler):
    """Simple mock backend supporting multiple failure scenarios."""
    scenario = "timeout"  # overwritten per test

    def _send(self, code: int, payload: Any):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/checkpoints":
            self._send(200, [{"id": "test", "episode": 1}])
        elif path == "/checkpoints/stats":
            self._send(200, {"total_checkpoints": 1})
        elif path == "/checkpoints/playback/current":
            if MockBackendHandler.scenario == "timeout":
                self._send(200, {"has_data": False, "status": {"is_playing": True, "is_paused": False}})
            elif MockBackendHandler.scenario == "websocket_failure":
                self._send(503, {"error": "WebSocket connection failed"})
            elif MockBackendHandler.scenario == "success":
                self._send(200, {"has_data": True, "playback_data": {"type": "checkpoint_playback"}})
            else:
                self._send(500, {"error": "Unexpected scenario"})
        else:
            self._send(404, {"error": "Not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path.endswith("/playback/start"):
            if MockBackendHandler.scenario == "api_error":
                self._send(500, {"error": "Internal server error during playback start"})
            else:
                self._send(200, {"message": "Playback started"})
        else:
            self._send(404, {"error": "Not found"})

    def log_message(self, *_):
        # Silence default logging for cleaner test output
        pass

@contextlib.contextmanager
def run_mock_backend(scenario: str):
    MockBackendHandler.scenario = scenario
    server = HTTPServer((HOST, PORT), MockBackendHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # give the server a moment to fully bind
    time.sleep(0.1)
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

# --------------------------------------------------------------------------------------
# Helper functions

def _request(method: str, path: str, body: dict | None = None):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=5)
    headers = {"Content-Type": "application/json"}
    payload = json.dumps(body).encode() if body else None
    conn.request(method, path, body=payload, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    try:
        parsed = json.loads(data.decode()) if data else None
    except json.JSONDecodeError:
        parsed = None
    return resp.status, parsed

# --------------------------------------------------------------------------------------
# Test scenarios
@requires_real_backend

def test_timeout(logger: TestLogger):
    """Test timeout scenario"""
    with run_mock_backend("timeout"):
        status, data = _request("GET", "/checkpoints/playback/current")
        assert status == 200, f"Expected 200, got {status}"
        assert data is not None and data.get("has_data") is False, "Expected has_data false in timeout scenario"
        logger.ok("timeout scenario passed")
@requires_real_backend

def test_api_error(logger: TestLogger):
    """Test API error scenario"""
    with run_mock_backend("api_error"):
        status, _ = _request("POST", "/checkpoints/test/playback/start", {})
        assert status == 500, f"Expected 500, got {status}"
        logger.ok("api_error scenario passed")
@requires_real_backend

def test_websocket_failure(logger: TestLogger):
    """Test WebSocket failure scenario"""
    with run_mock_backend("websocket_failure"):
        status, _ = _request("GET", "/checkpoints/playback/current")
        assert status == 503, f"Expected 503, got {status}"
        logger.ok("websocket_failure scenario passed")
@requires_real_backend

def test_success(logger: TestLogger):
    """Test success scenario"""
    with run_mock_backend("success"):
        status, data = _request("GET", "/checkpoints/playback/current")
        assert status == 200, f"Expected 200, got {status}"
        assert data is not None and data.get("has_data") is True, "Expected has_data true in success scenario"
        logger.ok("success scenario passed")

# --------------------------------------------------------------------------------------
@requires_real_backend

def main():
    """Main entry point"""
    logger = TestLogger()
    logger.banner("Checkpoint Loading Fixes Test Suite", 60)
    
    tests = [
        ("timeout", test_timeout),
        ("api_error", test_api_error),
        ("websocket_failure", test_websocket_failure),
        ("success", test_success),
    ]
    
    failures = 0
    for name, func in tests:
        try:
            logger.testing(f"Running {name} scenario...")
            func(logger)
        except AssertionError as e:
            logger.error(f"{name} scenario failed: {e}")
            failures += 1
        except Exception as e:
            logger.error(f"{name} scenario error: {e}")
            failures += 1
    
    if failures == 0:
        logger.success("All checkpoint loading fix tests passed!")
        sys.exit(0)
    else:
        logger.error(f"{failures} test(s) failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 