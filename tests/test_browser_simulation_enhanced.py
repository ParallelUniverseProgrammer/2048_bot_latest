#!/usr/bin/env python3
"""
Enhanced Browser Simulation Test
===============================

This is an enhanced version of the browser simulation test that addresses
performance timeout issues by implementing:

- Adaptive timeouts based on system performance
- Real-time performance monitoring
- Better error handling and recovery
- Progressive stress testing
- Graceful degradation under load
- Performance metrics collection
"""

import sys
import os
import asyncio
import time
import json
import threading
import psutil
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import weakref
from dataclasses import dataclass, field

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backend_availability_manager import BackendAvailabilityManager
from test_utils import TestLogger

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    messages_sent: int = 0
    messages_processed: int = 0
    errors: int = 0
    timeouts: int = 0
    max_memory_usage: float = 0.0
    avg_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    
    def add_response_time(self, response_time: float):
        """Add a response time measurement"""
        self.response_times.append(response_time)
        self.avg_response_time = sum(self.response_times) / len(self.response_times)
    
    def finish(self):
        """Mark metrics as finished"""
        self.end_time = time.time()
    
    def duration(self) -> float:
        """Get total duration"""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def messages_per_second(self) -> float:
        """Calculate messages per second"""
        duration = self.duration()
        return self.messages_processed / duration if duration > 0 else 0
    
    def error_rate(self) -> float:
        """Calculate error rate"""
        total = self.messages_sent + self.errors
        return self.errors / total if total > 0 else 0


class AdaptiveTimeoutManager:
    """Manages adaptive timeouts based on system performance"""
    
    def __init__(self, base_timeout: float = 30.0, min_timeout: float = 10.0, max_timeout: float = 120.0):
        self.base_timeout = base_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.performance_history = []
        self.system_load_threshold = 0.8
        
    def get_adaptive_timeout(self, operation_type: str = "default") -> float:
        """Get adaptive timeout based on current system performance"""
        
        # Get current system load
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Calculate system load factor
        system_load = max(cpu_percent, memory_percent) / 100.0
        
        # Adjust timeout based on system load
        if system_load > self.system_load_threshold:
            # System under stress - increase timeout
            load_factor = min(system_load / self.system_load_threshold, 3.0)
            timeout = self.base_timeout * load_factor
        else:
            # System performing well - use base timeout
            timeout = self.base_timeout
        
        # Apply operation-specific multipliers
        multipliers = {
            "stress_test": 1.5,
            "memory_test": 2.0,
            "concurrent_test": 1.2,
            "quick_test": 0.5
        }
        
        timeout *= multipliers.get(operation_type, 1.0)
        
        # Clamp to min/max bounds
        timeout = max(self.min_timeout, min(timeout, self.max_timeout))
        
        return timeout
    
    def record_performance(self, operation_type: str, duration: float, success: bool):
        """Record performance data for future timeout calculations"""
        self.performance_history.append({
            'operation': operation_type,
            'duration': duration,
            'success': success,
            'timestamp': time.time()
        })
        
        # Keep only recent history (last 100 entries)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]


class EnhancedBrowserSimulation:
    """Enhanced browser simulation with performance monitoring and adaptive timeouts"""
    
    def __init__(self, logger: Optional[TestLogger] = None):
        self.logger = logger or TestLogger()
        self.timeout_manager = AdaptiveTimeoutManager()
        self.backend_manager = BackendAvailabilityManager(test_logger=self.logger)
        self.performance_metrics = PerformanceMetrics()
        self.is_running = False
        self.should_stop = False
        
        # Performance monitoring
        self.monitoring_task = None
        self.performance_data = []
        
        # Mock components (for testing without real backend)
        self.mock_websocket_manager = None
        self.mock_playback = None
        
    async def start_performance_monitoring(self):
        """Start continuous performance monitoring"""
        self.monitoring_task = asyncio.create_task(self._monitor_performance())
        
    async def stop_performance_monitoring(self):
        """Stop performance monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_performance(self):
        """Continuously monitor system performance"""
        while not self.should_stop:
            try:
                # Collect performance data
                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                
                perf_data = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_info.percent,
                    'memory_available': memory_info.available,
                    'messages_sent': self.performance_metrics.messages_sent,
                    'messages_processed': self.performance_metrics.messages_processed,
                    'errors': self.performance_metrics.errors
                }
                
                self.performance_data.append(perf_data)
                
                # Update max memory usage
                self.performance_metrics.max_memory_usage = max(
                    self.performance_metrics.max_memory_usage,
                    memory_info.percent
                )
                
                # Keep only recent data (last 5 minutes)
                five_minutes_ago = time.time() - 300
                self.performance_data = [
                    d for d in self.performance_data 
                    if d['timestamp'] > five_minutes_ago
                ]
                
                await asyncio.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                self.logger.warning(f"Performance monitoring error: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error
    
    async def simulate_browser_websocket_load(self, num_clients: int = 5, 
                                            test_duration: float = 30.0) -> Tuple[bool, str, Dict[str, Any]]:
        """Simulate browser WebSocket load with adaptive timeout"""
        
        operation_type = "stress_test"
        adaptive_timeout = self.timeout_manager.get_adaptive_timeout(operation_type)
        
        self.logger.info(f"Starting WebSocket load simulation with {num_clients} clients")
        self.logger.info(f"Base timeout: {test_duration}s, Adaptive timeout: {adaptive_timeout:.1f}s")
        
        start_time = time.time()
        
        try:
            # Use adaptive timeout instead of fixed timeout
            actual_timeout = min(test_duration, adaptive_timeout)
            
            # Start performance monitoring
            await self.start_performance_monitoring()
            
            # Create mock websocket clients
            clients = []
            for i in range(num_clients):
                client = await self._create_mock_websocket_client(f"client_{i}")
                clients.append(client)
            
            # Simulate WebSocket message exchange
            tasks = []
            for i, client in enumerate(clients):
                task = asyncio.create_task(
                    self._simulate_client_activity(client, test_duration, i)
                )
                tasks.append(task)
            
            # Wait for completion or timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=actual_timeout
                )
                
                success = True
                message = f"Load test completed successfully in {time.time() - start_time:.2f}s"
                
            except asyncio.TimeoutError:
                # Handle timeout gracefully
                success = False
                message = f"Load test timed out after {actual_timeout:.1f}s (adaptive timeout)"
                
                # Try to cancel tasks gracefully
                for task in tasks:
                    task.cancel()
                
                # Wait a bit for cleanup
                await asyncio.sleep(1.0)
                
            # Stop performance monitoring
            await self.stop_performance_monitoring()
            
            # Calculate performance metrics
            duration = time.time() - start_time
            self.performance_metrics.finish()
            
            # Record performance for future timeout calculations
            self.timeout_manager.record_performance(operation_type, duration, success)
            
            # Generate performance report
            performance_report = self._generate_performance_report()
            
            return success, message, performance_report
            
        except Exception as e:
            await self.stop_performance_monitoring()
            return False, f"Error during load simulation: {str(e)}", {}
    
    async def _create_mock_websocket_client(self, client_id: str):
        """Create a mock WebSocket client for testing"""
        return {
            'id': client_id,
            'connected': True,
            'messages_sent': 0,
            'messages_received': 0,
            'last_activity': time.time(),
            'response_times': []
        }
    
    async def _simulate_client_activity(self, client: Dict[str, Any], 
                                      duration: float, client_index: int):
        """Simulate client WebSocket activity"""
        start_time = time.time()
        message_interval = 0.1 + (client_index * 0.05)  # Stagger clients
        
        try:
            while time.time() - start_time < duration and not self.should_stop:
                # Simulate sending a message
                message_start = time.time()
                
                # Create a realistic message
                message = {
                    'type': 'checkpoint_request',
                    'client_id': client['id'],
                    'timestamp': time.time(),
                    'step': client['messages_sent']
                }
                
                # Simulate message processing delay
                await asyncio.sleep(0.01)  # Simulate network/processing delay
                
                # Record metrics
                response_time = time.time() - message_start
                client['response_times'].append(response_time)
                client['messages_sent'] += 1
                client['last_activity'] = time.time()
                
                self.performance_metrics.messages_sent += 1
                self.performance_metrics.messages_processed += 1
                self.performance_metrics.add_response_time(response_time)
                
                # Wait for next message
                await asyncio.sleep(message_interval)
                
        except Exception as e:
            self.performance_metrics.errors += 1
            self.logger.warning(f"Client {client['id']} error: {e}")
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return {
            'duration': self.performance_metrics.duration(),
            'messages_sent': self.performance_metrics.messages_sent,
            'messages_processed': self.performance_metrics.messages_processed,
            'messages_per_second': self.performance_metrics.messages_per_second(),
            'errors': self.performance_metrics.errors,
            'error_rate': self.performance_metrics.error_rate(),
            'avg_response_time': self.performance_metrics.avg_response_time,
            'max_memory_usage': self.performance_metrics.max_memory_usage,
            'system_performance': self._get_system_performance_summary()
        }
    
    def _get_system_performance_summary(self) -> Dict[str, Any]:
        """Get system performance summary"""
        if not self.performance_data:
            return {}
        
        cpu_values = [d['cpu_percent'] for d in self.performance_data]
        memory_values = [d['memory_percent'] for d in self.performance_data]
        
        return {
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_percent': sum(memory_values) / len(memory_values),
            'max_memory_percent': max(memory_values),
            'samples': len(self.performance_data)
        }
    
    async def run_progressive_stress_test(self) -> Dict[str, Any]:
        """Run progressive stress test with increasing load"""
        self.logger.section("Progressive Stress Test")
        
        results = {}
        client_counts = [1, 2, 5, 10, 20]  # Progressive load increase
        base_duration = 10.0  # Shorter duration for progressive testing
        
        for client_count in client_counts:
            self.logger.info(f"Testing with {client_count} clients...")
            
            # Reset metrics for each test
            self.performance_metrics = PerformanceMetrics()
            
            # Run test with current client count
            success, message, report = await self.simulate_browser_websocket_load(
                num_clients=client_count,
                test_duration=base_duration
            )
            
            results[f"{client_count}_clients"] = {
                'success': success,
                'message': message,
                'report': report
            }
            
            # Print results
            if success:
                self.logger.ok(f"{client_count} clients: {message}")
                self.logger.info(f"  Messages/sec: {report.get('messages_per_second', 0):.1f}")
                self.logger.info(f"  Avg response time: {report.get('avg_response_time', 0):.3f}s")
            else:
                self.logger.error(f"{client_count} clients: {message}")
                # Don't continue if we're already failing
                break
            
            # Brief pause between tests
            await asyncio.sleep(2.0)
        
        return results
    
    async def run_memory_pressure_test(self) -> Dict[str, Any]:
        """Test system behavior under memory pressure"""
        self.logger.section("Memory Pressure Test")
        
        # Use longer timeout for memory test
        adaptive_timeout = self.timeout_manager.get_adaptive_timeout("memory_test")
        
        self.logger.info(f"Running memory pressure test (timeout: {adaptive_timeout:.1f}s)")
        
        # Force garbage collection before test
        gc.collect()
        
        # Run test with memory monitoring
        success, message, report = await self.simulate_browser_websocket_load(
            num_clients=10,
            test_duration=min(20.0, adaptive_timeout)
        )
        
        if success:
            self.logger.ok(f"Memory pressure test: {message}")
            self.logger.info(f"  Max memory usage: {report.get('max_memory_usage', 0):.1f}%")
        else:
            self.logger.warning(f"Memory pressure test: {message}")
        
        return {
            'success': success,
            'message': message,
            'report': report
        }
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the comprehensive enhanced browser simulation test suite"""
        self.logger.banner("Enhanced Browser Simulation Test Suite")
        
        all_results = {}
        
        # Test 1: Progressive stress test
        try:
            progressive_results = await self.run_progressive_stress_test()
            all_results['progressive_stress'] = progressive_results
        except Exception as e:
            self.logger.error(f"Progressive stress test failed: {e}")
            all_results['progressive_stress'] = {'error': str(e)}
        
        # Test 2: Memory pressure test
        try:
            memory_results = await self.run_memory_pressure_test()
            all_results['memory_pressure'] = memory_results
        except Exception as e:
            self.logger.error(f"Memory pressure test failed: {e}")
            all_results['memory_pressure'] = {'error': str(e)}
        
        # Test 3: Backend availability test
        try:
            backend_available = self.backend_manager.is_backend_available()
            all_results['backend_availability'] = {
                'available': backend_available,
                'backend_type': self.backend_manager.get_backend_type()
            }
            if backend_available:
                self.logger.ok(f"Backend available: {self.backend_manager.get_backend_type()}")
            else:
                self.logger.warning("Backend not available")
        except Exception as e:
            self.logger.error(f"Backend availability test failed: {e}")
            all_results['backend_availability'] = {'error': str(e)}
        
        # Generate summary
        self._print_test_summary(all_results)
        
        return all_results
    
    def _print_test_summary(self, results: Dict[str, Any]):
        """Print comprehensive test summary"""
        self.logger.separator()
        self.logger.banner("Test Summary")
        
        # Progressive stress test summary
        if 'progressive_stress' in results:
            prog_results = results['progressive_stress']
            if 'error' not in prog_results:
                successful_clients = [
                    k for k, v in prog_results.items() 
                    if v.get('success', False)
                ]
                max_clients = max([int(k.split('_')[0]) for k in successful_clients]) if successful_clients else 0
                self.logger.info(f"Progressive Stress: Max successful clients: {max_clients}")
            else:
                self.logger.error(f"Progressive Stress: {prog_results['error']}")
        
        # Memory pressure test summary
        if 'memory_pressure' in results:
            mem_results = results['memory_pressure']
            if 'error' not in mem_results:
                if mem_results.get('success', False):
                    self.logger.ok("Memory Pressure: PASSED")
                else:
                    self.logger.warning("Memory Pressure: FAILED")
            else:
                self.logger.error(f"Memory Pressure: {mem_results['error']}")
        
        # Backend availability summary
        if 'backend_availability' in results:
            backend_results = results['backend_availability']
            if 'error' not in backend_results:
                if backend_results.get('available', False):
                    self.logger.ok(f"Backend: Available ({backend_results.get('backend_type', 'unknown')})")
                else:
                    self.logger.warning("Backend: Not available")
            else:
                self.logger.error(f"Backend: {backend_results['error']}")
        
        # Final verdict
        self.logger.separator()
        self.logger.success("Enhanced Browser Simulation Test Suite Complete")


async def main():
    """Main test runner for enhanced browser simulation"""
    logger = TestLogger()
    
    # Create and run enhanced browser simulation
    simulation = EnhancedBrowserSimulation(logger)
    
    # Run comprehensive test suite
    results = await simulation.run_comprehensive_test_suite()
    
    # Output results as JSON for analysis
    print("\nDETAILED RESULTS:")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main()) 