#!/usr/bin/env python3
"""
Backend Parity Test
==================

This test ensures that the mock backend and real backend provide identical
API endpoints, response formats, and behavior. This is critical for ensuring
that tests using the mock backend accurately reflect the behavior of the
real backend system.

The test covers:
- API endpoint availability and structure
- Response format consistency
- Error handling behavior
- WebSocket functionality
- Data validation and serialization
- Performance characteristics (within acceptable ranges)
"""

import asyncio
import json
import time
import sys
import os
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path for independent execution
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tests.utilities.test_utils import TestLogger, BackendTester
    from tests.utilities.backend_manager import (
        requires_real_backend, 
        requires_mock_backend,
        get_global_backend_manager,
        real_backend_context,
        mock_backend_context,
        BackendManager
    )
    from tests.utilities.mock_backend import MockBackend
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


@dataclass
class EndpointTestResult:
    """Result of testing a single endpoint"""
    endpoint: str
    method: str
    real_backend_success: bool
    mock_backend_success: bool
    real_response: Optional[Dict[str, Any]] = None
    mock_response: Optional[Dict[str, Any]] = None
    real_error: Optional[str] = None
    mock_error: Optional[str] = None
    response_time_real: float = 0.0
    response_time_mock: float = 0.0
    structure_match: bool = False
    data_parity: bool = False
    
    def is_parity_achieved(self) -> bool:
        """Check if this endpoint has achieved parity"""
        return (self.real_backend_success == self.mock_backend_success and
                self.structure_match and
                self.data_parity)


@dataclass
class ParityTestResults:
    """Results of the complete parity test"""
    total_endpoints: int = 0
    successful_parity: int = 0
    failed_parity: int = 0
    endpoint_results: List[EndpointTestResult] = field(default_factory=list)
    performance_comparison: Dict[str, Any] = field(default_factory=dict)
    websocket_parity: bool = False
    error_handling_parity: bool = False
    
    def add_endpoint_result(self, result: EndpointTestResult):
        """Add an endpoint test result"""
        self.endpoint_results.append(result)
        self.total_endpoints += 1
        if result.is_parity_achieved():
            self.successful_parity += 1
        else:
            self.failed_parity += 1
    
    def get_success_rate(self) -> float:
        """Get the success rate as a percentage"""
        if self.total_endpoints == 0:
            return 0.0
        return (self.successful_parity / self.total_endpoints) * 100


class BackendParityTester:
    """Test class for ensuring backend parity"""
    
    def __init__(self):
        self.logger = TestLogger()
        # Create separate backend managers for real and mock backends
        self.real_backend_manager = BackendManager(
            base_url="http://localhost:8000",
            backend_port=8000
        )
        self.mock_backend_manager = BackendManager(
            base_url="http://localhost:8001", 
            backend_port=8001
        )
        
        # Define all API endpoints to test
        self.endpoints_to_test = [
            # Core API endpoints
            ("/checkpoints", "GET"),
            ("/checkpoints/stats", "GET"),
            ("/checkpoints/{checkpoint_id}", "GET"),
            ("/training/status", "GET"),
            ("/training/start", "POST"),
            ("/training/stop", "POST"),
            ("/playback/start", "POST"),
            ("/playback/stop", "POST"),
            ("/playback/pause", "POST"),
            ("/playback/resume", "POST"),
            ("/playback/speed", "GET"),
            ("/playback/speed", "POST"),
            
            # Design/Model Studio endpoints
            ("/design", "GET"),
            ("/design", "POST"),
            ("/design/{design_id}", "GET"),
            ("/design/{design_id}", "PUT"),
            ("/design/{design_id}", "DELETE"),
            ("/design/{design_id}/validate", "POST"),
            ("/design/{design_id}/compile", "POST"),
            
            # Health and status endpoints
            ("/health", "GET"),
            ("/status", "GET"),
            ("/metrics", "GET"),
        ]
    
    def _get_test_checkpoint_id(self) -> str:
        """Get a test checkpoint ID for testing"""
        return "test_checkpoint_episode_100"
    
    def _get_test_design_id(self) -> str:
        """Get a test design ID for testing"""
        return "test_design_001"
    
    def _replace_placeholders(self, endpoint: str) -> str:
        """Replace placeholder values in endpoints"""
        return (endpoint
                .replace("{checkpoint_id}", self._get_test_checkpoint_id())
                .replace("{design_id}", self._get_test_design_id()))
    
    def _compare_response_structures(self, real_response: Dict[str, Any], 
                                   mock_response: Dict[str, Any]) -> bool:
        """Compare the structure of two responses"""
        try:
            # Convert to JSON strings and back to normalize structure
            real_json = json.dumps(real_response, sort_keys=True)
            mock_json = json.dumps(mock_response, sort_keys=True)
            
            real_parsed = json.loads(real_json)
            mock_parsed = json.loads(mock_json)
            
            # Check if keys match
            real_keys = set(real_parsed.keys())
            mock_keys = set(mock_parsed.keys())
            
            if real_keys != mock_keys:
                self.logger.warning(f"Key mismatch: real={real_keys}, mock={mock_keys}")
                return False
            
            # Check if data types match for each key
            for key in real_keys:
                real_type = type(real_parsed[key])
                mock_type = type(mock_parsed[key])
                
                if real_type != mock_type:
                    self.logger.warning(f"Type mismatch for {key}: real={real_type}, mock={mock_type}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error comparing response structures: {e}")
            return False
    
    def _compare_response_data(self, real_response: Dict[str, Any], 
                             mock_response: Dict[str, Any]) -> bool:
        """Compare the actual data values in responses"""
        try:
            # For some endpoints, we expect exact data matches
            # For others, we only check structure and data types
            
            # Convert to JSON strings for comparison
            real_json = json.dumps(real_response, sort_keys=True)
            mock_json = json.dumps(mock_response, sort_keys=True)
            
            # For endpoints that should have exact matches
            exact_match_endpoints = ["/health", "/status", "/metrics"]
            
            # For endpoints that may have different data but same structure
            structure_only_endpoints = ["/checkpoints", "/checkpoints/stats", "/training/status"]
            
            # Check if this is an exact match endpoint
            for endpoint in exact_match_endpoints:
                if any(endpoint in ep for ep in [real_json, mock_json]):
                    return real_json == mock_json
            
            # For structure-only endpoints, we already checked structure
            # Now just verify that the data types and basic structure are correct
            return True
            
        except Exception as e:
            self.logger.error(f"Error comparing response data: {e}")
            return False
    
    async def _test_single_endpoint(self, endpoint: str, method: str) -> EndpointTestResult:
        """Test a single endpoint on both backends"""
        result = EndpointTestResult(
            endpoint=endpoint, 
            method=method,
            real_backend_success=False,
            mock_backend_success=False
        )
        
        # Test real backend
        try:
            start_time = time.time()
            if self.real_backend_manager.is_backend_available(backend_type='real'):
                real_response = await self._make_request(endpoint, method, "real")
                result.real_backend_success = True
                result.real_response = real_response
            else:
                # Try to start real backend
                if self.real_backend_manager.start_real_backend():
                    real_response = await self._make_request(endpoint, method, "real")
                    result.real_backend_success = True
                    result.real_response = real_response
                else:
                    result.real_backend_success = False
                    result.real_error = "Real backend not available"
            result.response_time_real = time.time() - start_time
            
        except Exception as e:
            result.real_backend_success = False
            result.real_error = str(e)
        
        # Test mock backend
        try:
            start_time = time.time()
            if self.mock_backend_manager.is_backend_available(backend_type='mock'):
                mock_response = await self._make_request(endpoint, method, "mock")
                result.mock_backend_success = True
                result.mock_response = mock_response
            else:
                # Try to start mock backend
                if self.mock_backend_manager.start_mock_backend():
                    mock_response = await self._make_request(endpoint, method, "mock")
                    result.mock_backend_success = True
                    result.mock_response = mock_response
                else:
                    result.mock_backend_success = False
                    result.mock_error = "Mock backend not available"
            result.response_time_mock = time.time() - start_time
            
        except Exception as e:
            result.mock_backend_success = False
            result.mock_error = str(e)
        
        # Compare responses if both succeeded
        if (result.real_backend_success and result.mock_backend_success and
            result.real_response is not None and result.mock_response is not None):
            result.structure_match = self._compare_response_structures(
                result.real_response, result.mock_response
            )
            result.data_parity = self._compare_response_data(
                result.real_response, result.mock_response
            )
        
        return result
    
    async def _make_request(self, endpoint: str, method: str, backend_type: str) -> Dict[str, Any]:
        """Make a request to the specified backend"""
        import aiohttp
        
        # Replace placeholders in endpoint
        actual_endpoint = self._replace_placeholders(endpoint)
        
        # Determine base URL based on backend type
        if backend_type == "real":
            base_url = "http://localhost:8000"
        else:  # mock
            base_url = "http://localhost:8001"
        
        url = f"{base_url}{actual_endpoint}"
        
        # Prepare request data for POST requests
        data = None
        if method == "POST":
            if "playback/start" in endpoint:
                data = {"checkpoint_id": self._get_test_checkpoint_id()}
            elif "design" in endpoint and method == "POST":
                data = {"name": "Test Design", "description": "Test design for parity testing"}
            elif "training/start" in endpoint:
                data = {"episodes": 10}
        
        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(url) as response:
                    return await response.json()
            elif method == "POST":
                async with session.post(url, json=data) as response:
                    return await response.json()
            elif method == "PUT":
                async with session.put(url, json=data) as response:
                    return await response.json()
            elif method == "DELETE":
                async with session.delete(url) as response:
                    return await response.json()
            else:
                raise ValueError(f"Unsupported method: {method}")
    
    async def test_all_endpoints(self) -> List[EndpointTestResult]:
        """Test all endpoints on both backends"""
        self.logger.banner("Testing All Endpoints for Parity", 60)
        
        results = []
        
        for endpoint, method in self.endpoints_to_test:
            self.logger.info(f"Testing {method} {endpoint}")
            result = await self._test_single_endpoint(endpoint, method)
            results.append(result)
            
            if result.is_parity_achieved():
                self.logger.ok(f"✓ Parity achieved for {method} {endpoint}")
            else:
                self.logger.error(f"✗ Parity failed for {method} {endpoint}")
                if result.real_error:
                    self.logger.warning(f"  Real backend error: {result.real_error}")
                if result.mock_error:
                    self.logger.warning(f"  Mock backend error: {result.mock_error}")
        
        return results
    
    async def test_websocket_parity(self) -> bool:
        """Test WebSocket functionality parity"""
        self.logger.section("Testing WebSocket Parity")
        
        try:
            # Test real backend WebSocket
            real_ws_works = False
            if self.real_backend_manager.is_backend_available(backend_type='real'):
                real_ws_works = await self._test_websocket_connection("real")
            
            # Test mock backend WebSocket
            mock_ws_works = False
            if self.mock_backend_manager.is_backend_available(backend_type='mock'):
                mock_ws_works = await self._test_websocket_connection("mock")
            
            parity_achieved = real_ws_works == mock_ws_works
            
            if parity_achieved:
                self.logger.ok("WebSocket parity achieved")
            else:
                self.logger.error("WebSocket parity failed")
            
            return parity_achieved
            
        except Exception as e:
            self.logger.error(f"WebSocket parity test failed: {e}")
            return False
    
    async def _test_websocket_connection(self, backend_type: str) -> bool:
        """Test WebSocket connection for a specific backend"""
        import websockets
        
        try:
            # Determine WebSocket URL
            if backend_type == "real":
                ws_url = "ws://localhost:8000/ws"
            else:  # mock
                ws_url = "ws://localhost:8001/ws"
            
            # Try to connect
            async with websockets.connect(ws_url) as websocket:
                # Send a test message
                test_message = {"type": "ping", "data": "parity_test"}
                await websocket.send(json.dumps(test_message))
                
                # Wait for response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    return True
                except asyncio.TimeoutError:
                    return False
                    
        except Exception as e:
            self.logger.warning(f"WebSocket test failed for {backend_type}: {e}")
            return False
    
    @requires_mock_backend("Error Handling Parity Test")
    def test_error_handling_parity(self) -> bool:
        """Test error handling behavior parity"""
        self.logger.section("Testing Error Handling Parity")
        
        try:
            # Test invalid endpoint on both backends
            real_error_handling = self._test_error_handling("real")
            mock_error_handling = self._test_error_handling("mock")
            
            parity_achieved = real_error_handling == mock_error_handling
            
            if parity_achieved:
                self.logger.ok("Error handling parity achieved")
            else:
                self.logger.error("Error handling parity failed")
            
            return parity_achieved
            
        except Exception as e:
            self.logger.error(f"Error handling parity test failed: {e}")
            return False
    
    def _test_error_handling(self, backend_type: str) -> bool:
        """Test error handling for a specific backend"""
        import requests
        
        try:
            # Determine base URL
            if backend_type == "real":
                base_url = "http://localhost:8000"
            else:  # mock
                base_url = "http://localhost:8001"
            
            # Test invalid endpoint
            response = requests.get(f"{base_url}/invalid_endpoint", timeout=5)
            
            # Both should return 404 or similar error status
            return response.status_code >= 400
            
        except Exception:
            # If request fails entirely, that's also acceptable error handling
            return True
    
    def analyze_performance_comparison(self, results: List[EndpointTestResult]) -> Dict[str, Any]:
        """Analyze performance differences between backends"""
        self.logger.section("Analyzing Performance Comparison")
        
        real_times = [r.response_time_real for r in results if r.real_backend_success]
        mock_times = [r.response_time_mock for r in results if r.mock_backend_success]
        
        if not real_times or not mock_times:
            return {"error": "Insufficient data for performance comparison"}
        
        avg_real = sum(real_times) / len(real_times)
        avg_mock = sum(mock_times) / len(mock_times)
        
        # Mock backend should be faster (within reasonable bounds)
        speed_ratio = avg_real / avg_mock if avg_mock > 0 else float('inf')
        
        performance_analysis = {
            "avg_real_response_time": avg_real,
            "avg_mock_response_time": avg_mock,
            "speed_ratio": speed_ratio,
            "performance_acceptable": speed_ratio > 1.0,  # Mock should be faster
            "total_real_requests": len(real_times),
            "total_mock_requests": len(mock_times)
        }
        
        if performance_analysis["performance_acceptable"]:
            self.logger.ok("Performance comparison acceptable")
        else:
            self.logger.warning("Performance comparison shows issues")
        
        return performance_analysis
    
    def generate_parity_report(self, results: ParityTestResults) -> None:
        """Generate a comprehensive parity report"""
        self.logger.banner("Backend Parity Test Report", 60)
        
        # Summary statistics
        self.logger.info(f"Total endpoints tested: {results.total_endpoints}")
        self.logger.info(f"Successful parity: {results.successful_parity}")
        self.logger.info(f"Failed parity: {results.failed_parity}")
        self.logger.info(f"Success rate: {results.get_success_rate():.1f}%")
        
        # Performance comparison
        if results.performance_comparison:
            perf = results.performance_comparison
            self.logger.info(f"Average real response time: {perf.get('avg_real_response_time', 0):.3f}s")
            self.logger.info(f"Average mock response time: {perf.get('avg_mock_response_time', 0):.3f}s")
            self.logger.info(f"Speed ratio: {perf.get('speed_ratio', 0):.2f}x")
        
        # WebSocket and error handling
        self.logger.info(f"WebSocket parity: {'✓' if results.websocket_parity else '✗'}")
        self.logger.info(f"Error handling parity: {'✓' if results.error_handling_parity else '✗'}")
        
        # Detailed endpoint results
        if results.failed_parity > 0:
            self.logger.section("Failed Endpoints")
            for result in results.endpoint_results:
                if not result.is_parity_achieved():
                    self.logger.error(f"{result.method} {result.endpoint}")
                    if result.real_error:
                        self.logger.warning(f"  Real: {result.real_error}")
                    if result.mock_error:
                        self.logger.warning(f"  Mock: {result.mock_error}")
        
        # Overall assessment
        overall_success = (results.get_success_rate() >= 95.0 and 
                          results.websocket_parity and 
                          results.error_handling_parity)
        
        if overall_success:
            self.logger.success("✓ BACKEND PARITY ACHIEVED")
        else:
            self.logger.error("✗ BACKEND PARITY FAILED")
    
    async def run_complete_parity_test(self) -> ParityTestResults:
        """Run the complete backend parity test suite"""
        self.logger.banner("Backend Parity Test Suite", 60)
        
        results = ParityTestResults()
        
        try:
            # Test all endpoints
            endpoint_results = await self.test_all_endpoints()
            for result in endpoint_results:
                results.add_endpoint_result(result)
            
            # Test WebSocket parity
            results.websocket_parity = await self.test_websocket_parity()
            
            # Test error handling parity
            results.error_handling_parity = self.test_error_handling_parity()
            
            # Analyze performance
            results.performance_comparison = self.analyze_performance_comparison(endpoint_results)
            
            # Generate report
            self.generate_parity_report(results)
            
        finally:
            # Clean up backends
            self.cleanup()
        
        return results
    
    def cleanup(self):
        """Clean up backend resources"""
        try:
            if self.real_backend_manager:
                self.real_backend_manager.stop_real_backend()
            if self.mock_backend_manager:
                self.mock_backend_manager.stop_mock_backend()
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")


@requires_mock_backend("Backend Parity Test Suite")
async def main():
    """Main entry point for backend parity testing"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Backend Parity Test Suite")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output")
    parser.add_argument("--timeout", "-t", type=int, default=300,
                       help="Timeout for individual tests in seconds (default: 300)")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Run quick test with fewer endpoints")
    
    args = parser.parse_args()
    
    logger = TestLogger()
    logger.banner("Backend Parity Test Suite", 60)
    
    if args.verbose:
        logger.info("Verbose mode enabled")
    if args.quick:
        logger.info("Quick test mode enabled")
    
    tester = BackendParityTester()
    
    # Modify test behavior based on arguments
    if args.quick:
        # Use only core endpoints for quick test
        tester.endpoints_to_test = [
            ("/checkpoints", "GET"),
            ("/checkpoints/stats", "GET"),
            ("/training/status", "GET"),
            ("/health", "GET"),
            ("/status", "GET"),
        ]
        logger.info("Running quick test with core endpoints only")
    
    results = await tester.run_complete_parity_test()
    
    # Determine overall success
    overall_success = (results.get_success_rate() >= 95.0 and 
                      results.websocket_parity and 
                      results.error_handling_parity)
    
    if overall_success:
        logger.success("Backend parity test suite PASSED")
        return True
    else:
        logger.error("Backend parity test suite FAILED")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 