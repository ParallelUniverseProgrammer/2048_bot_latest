import sys, os
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Add backend to path for internal imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import asyncio
import time
import threading
import psutil
import gc
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime

from backend.app.models.checkpoint_playback import CheckpointPlayback
from backend.app.models.checkpoint_metadata import CheckpointManager
from backend.app.api.websocket_manager import WebSocketManager


class SystemMonitor:
    """Monitor system resources during playback"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.stats = []
        self.monitor_task = None
    
    async def start_monitoring(self, interval=0.5):
        """Start monitoring system resources"""
        self.monitoring = True
        self.stats = []
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
    
    async def stop_monitoring(self):
        """Stop monitoring and return collected stats"""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        return self.stats
    
    async def _monitor_loop(self, interval):
        """Monitor system resources in a loop"""
        while self.monitoring:
            try:
                # Get memory info
                memory_info = self.process.memory_info()
                
                # Get CPU usage
                cpu_percent = self.process.cpu_percent()
                
                # Get thread count
                thread_count = self.process.num_threads()
                
                # Get file descriptors (Unix-like systems)
                try:
                    fd_count = self.process.num_fds()
                except (AttributeError, psutil.AccessDenied):
                    fd_count = 0
                
                # Get asyncio event loop info
                loop = asyncio.get_event_loop()
                pending_tasks = len([t for t in asyncio.all_tasks() if not t.done()])
                
                stat = {
                    'timestamp': time.time(),
                    'memory_rss': memory_info.rss,
                    'memory_vms': memory_info.vms,
                    'cpu_percent': cpu_percent,
                    'thread_count': thread_count,
                    'fd_count': fd_count,
                    'pending_tasks': pending_tasks,
                    'gc_objects': len(gc.get_objects())
                }
                
                self.stats.append(stat)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"Error in monitoring: {e}")
                await asyncio.sleep(interval)
    
    def get_summary(self):
        """Get summary of monitoring data"""
        if not self.stats:
            return {}
        
        memory_rss = [s['memory_rss'] for s in self.stats]
        cpu_usage = [s['cpu_percent'] for s in self.stats]
        thread_counts = [s['thread_count'] for s in self.stats]
        
        return {
            'duration': self.stats[-1]['timestamp'] - self.stats[0]['timestamp'],
            'memory_peak': max(memory_rss) / 1024 / 1024,  # MB
            'memory_avg': sum(memory_rss) / len(memory_rss) / 1024 / 1024,  # MB
            'cpu_peak': max(cpu_usage),
            'cpu_avg': sum(cpu_usage) / len(cpu_usage),
            'thread_peak': max(thread_counts),
            'thread_avg': sum(thread_counts) / len(thread_counts),
            'sample_count': len(self.stats)
        }


class DetailedPlaybackMonitor:
    """Detailed monitoring of playback operations"""
    
    def __init__(self):
        self.events = []
        self.step_timings = []
        self.broadcast_timings = []
        self.error_events = []
        self.start_time = None
        
    def log_event(self, event_type: str, details: Dict[str, Any] = None):
        """Log an event with timestamp"""
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'details': details or {},
            'relative_time': time.time() - self.start_time if self.start_time else 0
        }
        self.events.append(event)
        
    def log_step_timing(self, step_num: int, duration: float, action: int = None):
        """Log timing for a game step"""
        self.step_timings.append({
            'step': step_num,
            'duration': duration,
            'action': action,
            'timestamp': time.time()
        })
    
    def log_broadcast_timing(self, message_type: str, duration: float, success: bool):
        """Log timing for a broadcast"""
        self.broadcast_timings.append({
            'message_type': message_type,
            'duration': duration,
            'success': success,
            'timestamp': time.time()
        })
    
    def log_error(self, error: str, context: str = ""):
        """Log an error event"""
        self.error_events.append({
            'error': error,
            'context': context,
            'timestamp': time.time()
        })
    
    def start_monitoring(self):
        """Start the monitoring session"""
        self.start_time = time.time()
        self.log_event('monitoring_started')
    
    def get_summary(self):
        """Get a summary of the monitoring session"""
        if not self.events:
            return {}
        
        total_duration = self.events[-1]['timestamp'] - self.events[0]['timestamp']
        
        summary = {
            'total_duration': total_duration,
            'total_events': len(self.events),
            'total_errors': len(self.error_events),
            'step_count': len(self.step_timings),
            'broadcast_count': len(self.broadcast_timings)
        }
        
        # Step timing analysis
        if self.step_timings:
            durations = [s['duration'] for s in self.step_timings]
            summary['step_timing'] = {
                'avg_duration': sum(durations) / len(durations),
                'max_duration': max(durations),
                'min_duration': min(durations),
                'slow_steps': [s for s in self.step_timings if s['duration'] > 1.0]
            }
        
        # Broadcast timing analysis
        if self.broadcast_timings:
            durations = [b['duration'] for b in self.broadcast_timings]
            success_count = sum(1 for b in self.broadcast_timings if b['success'])
            summary['broadcast_timing'] = {
                'avg_duration': sum(durations) / len(durations),
                'max_duration': max(durations),
                'success_rate': success_count / len(self.broadcast_timings),
                'slow_broadcasts': [b for b in self.broadcast_timings if b['duration'] > 2.0]
            }
        
        return summary


class InstrumentedCheckpointPlayback(CheckpointPlayback):
    """Instrumented version of CheckpointPlayback with detailed monitoring"""
    
    def __init__(self, checkpoint_manager, monitor: DetailedPlaybackMonitor):
        super().__init__(checkpoint_manager)
        self.monitor = monitor
        
    def load_checkpoint(self, checkpoint_id: str) -> bool:
        """Instrumented checkpoint loading"""
        start_time = time.time()
        self.monitor.log_event('checkpoint_load_start', {'checkpoint_id': checkpoint_id})
        
        try:
            result = super().load_checkpoint(checkpoint_id)
            duration = time.time() - start_time
            
            self.monitor.log_event('checkpoint_load_complete', {
                'checkpoint_id': checkpoint_id,
                'success': result,
                'duration': duration
            })
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.monitor.log_error(f"Checkpoint loading failed: {str(e)}", f"load_checkpoint({checkpoint_id})")
            self.monitor.log_event('checkpoint_load_failed', {
                'checkpoint_id': checkpoint_id,
                'error': str(e),
                'duration': duration
            })
            raise
    
    def select_action(self, state, legal_actions, env_game):
        """Instrumented action selection"""
        start_time = time.time()
        
        try:
            action, action_probs, attention_weights = super().select_action(state, legal_actions, env_game)
            duration = time.time() - start_time
            
            # Log slow action selections
            if duration > 0.5:
                self.monitor.log_event('slow_action_selection', {
                    'duration': duration,
                    'action': action,
                    'legal_actions': legal_actions
                })
            
            return action, action_probs, attention_weights
            
        except Exception as e:
            duration = time.time() - start_time
            self.monitor.log_error(f"Action selection failed: {str(e)}", "select_action")
            self.monitor.log_event('action_selection_failed', {
                'error': str(e),
                'duration': duration
            })
            raise
    
    def play_single_game(self) -> Dict[str, Any]:
        """Instrumented single game playing"""
        start_time = time.time()
        self.monitor.log_event('game_start')
        
        try:
            result = super().play_single_game()
            duration = time.time() - start_time
            
            self.monitor.log_event('game_complete', {
                'duration': duration,
                'steps': result.get('steps', 0),
                'score': result.get('final_score', 0),
                'success': 'error' not in result
            })
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.monitor.log_error(f"Game playing failed: {str(e)}", "play_single_game")
            self.monitor.log_event('game_failed', {
                'error': str(e),
                'duration': duration
            })
            raise


class InstrumentedWebSocketManager:
    """Instrumented WebSocket manager for monitoring broadcasts"""
    
    def __init__(self, real_manager, monitor: DetailedPlaybackMonitor):
        self.real_manager = real_manager
        self.monitor = monitor
        self.broadcast_count = 0
        
    def get_connection_count(self):
        """Get connection count from real manager"""
        return self.real_manager.get_connection_count()
    
    async def broadcast(self, message: Dict[str, Any]):
        """Instrumented broadcast method"""
        start_time = time.time()
        self.broadcast_count += 1
        message_type = message.get('type', 'unknown')
        
        self.monitor.log_event('broadcast_start', {
            'message_type': message_type,
            'broadcast_num': self.broadcast_count
        })
        
        try:
            await self.real_manager.broadcast(message)
            duration = time.time() - start_time
            
            self.monitor.log_broadcast_timing(message_type, duration, True)
            
            # Log slow broadcasts
            if duration > 2.0:
                self.monitor.log_event('slow_broadcast', {
                    'message_type': message_type,
                    'duration': duration,
                    'broadcast_num': self.broadcast_count
                })
            
        except Exception as e:
            duration = time.time() - start_time
            self.monitor.log_broadcast_timing(message_type, duration, False)
            self.monitor.log_error(f"Broadcast failed: {str(e)}", f"broadcast({message_type})")
            self.monitor.log_event('broadcast_failed', {
                'message_type': message_type,
                'error': str(e),
                'duration': duration
            })
            raise


class FreezeDiagnostics:
    """Main diagnostic tool for checkpoint playback freezing"""
    
    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.playback_monitor = DetailedPlaybackMonitor()
        self.checkpoint_manager = None
        self.websocket_manager = None
        self.playback = None
        
    def setup_for_real_system(self):
        """Setup diagnostics for real system (not test mocks)"""
        # Use real checkpoint manager
        checkpoint_dir = os.getenv('CHECKPOINTS_DIR', os.path.join(os.path.dirname(__file__), '..', 'backend', 'checkpoints'))
        print(f"[test_freeze_diagnostics] Using checkpoint_dir: {checkpoint_dir}")
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Use real WebSocket manager
        from backend.app.api.websocket_manager import WebSocketManager
        real_ws_manager = WebSocketManager()
        self.websocket_manager = InstrumentedWebSocketManager(real_ws_manager, self.playback_monitor)
        
        # Create instrumented playback
        self.playback = InstrumentedCheckpointPlayback(self.checkpoint_manager, self.playback_monitor)
    
    async def diagnose_checkpoint_loading(self, checkpoint_id: str):
        """Diagnose checkpoint loading performance"""
        print(f"\n=== Diagnosing Checkpoint Loading: {checkpoint_id} ===")
        
        self.playback_monitor.start_monitoring()
        await self.system_monitor.start_monitoring()
        
        try:
            start_time = time.time()
            success = self.playback.load_checkpoint(checkpoint_id)
            duration = time.time() - start_time
            
            print(f"Checkpoint loading: {'SUCCESS' if success else 'FAILED'} ({duration:.2f}s)")
            
            if not success:
                print("ERROR: Checkpoint loading failed - check error logs")
                return False
            
            return True
            
        except Exception as e:
            print(f"ERROR: Checkpoint loading exception: {e}")
            return False
        
        finally:
            system_stats = await self.system_monitor.stop_monitoring()
            system_summary = self.system_monitor.get_summary()
            
            print(f"System Impact: Memory peak: {system_summary.get('memory_peak', 0):.1f}MB, "
                  f"CPU peak: {system_summary.get('cpu_peak', 0):.1f}%")
    
    async def diagnose_single_game(self, checkpoint_id: str):
        """Diagnose single game playing performance"""
        print(f"\n=== Diagnosing Single Game: {checkpoint_id} ===")
        
        # Load checkpoint first
        if not await self.diagnose_checkpoint_loading(checkpoint_id):
            return False
        
        self.playback_monitor.start_monitoring()
        await self.system_monitor.start_monitoring()
        
        try:
            start_time = time.time()
            result = self.playback.play_single_game()
            duration = time.time() - start_time
            
            if 'error' in result:
                print(f"ERROR: Single game failed: {result['error']}")
                return False
            
            print(f"Single game: SUCCESS ({duration:.2f}s)")
            print(f"  Steps: {result.get('steps', 0)}")
            print(f"  Score: {result.get('final_score', 0)}")
            print(f"  Max tile: {result.get('max_tile', 0)}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Single game exception: {e}")
            return False
        
        finally:
            system_stats = await self.system_monitor.stop_monitoring()
            system_summary = self.system_monitor.get_summary()
            playback_summary = self.playback_monitor.get_summary()
            
            print(f"System Impact: Memory peak: {system_summary.get('memory_peak', 0):.1f}MB, "
                  f"CPU peak: {system_summary.get('cpu_peak', 0):.1f}%")
            
            if 'step_timing' in playback_summary:
                step_timing = playback_summary['step_timing']
                print(f"Step Performance: Avg: {step_timing['avg_duration']:.3f}s, "
                      f"Max: {step_timing['max_duration']:.3f}s")
    
    async def diagnose_live_playback(self, checkpoint_id: str, duration_seconds: int = 30):
        """Diagnose live playback performance"""
        print(f"\n=== Diagnosing Live Playback: {checkpoint_id} ===")
        
        # Load checkpoint first
        if not await self.diagnose_checkpoint_loading(checkpoint_id):
            return False
        
        self.playback_monitor.start_monitoring()
        await self.system_monitor.start_monitoring()
        
        try:
            # Start live playback
            playback_task = asyncio.create_task(
                self.playback.start_live_playback(self.websocket_manager)
            )
            
            # Monitor for specified duration
            start_time = time.time()
            monitoring_task = asyncio.create_task(self._monitor_playback_health(duration_seconds))
            
            # Wait for monitoring to complete
            await monitoring_task
            
            # Stop playback
            self.playback.stop_playback()
            
            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(playback_task, timeout=10.0)
                print("OK: Live playback completed successfully")
                return True
            except asyncio.TimeoutError:
                print("ERROR: Live playback hung during shutdown!")
                return False
            
        except Exception as e:
            print(f"ERROR: Live playback exception: {e}")
            return False
        
        finally:
            system_stats = await self.system_monitor.stop_monitoring()
            system_summary = self.system_monitor.get_summary()
            playback_summary = self.playback_monitor.get_summary()
            
            print(f"\nSystem Impact: Memory peak: {system_summary.get('memory_peak', 0):.1f}MB, "
                  f"CPU peak: {system_summary.get('cpu_peak', 0):.1f}%")
            
            if 'broadcast_timing' in playback_summary:
                broadcast_timing = playback_summary['broadcast_timing']
                print(f"Broadcast Performance: Avg: {broadcast_timing['avg_duration']:.3f}s, "
                      f"Max: {broadcast_timing['max_duration']:.3f}s, "
                      f"Success: {broadcast_timing['success_rate']:.1%}")
            
            print(f"Total Events: {playback_summary.get('total_events', 0)}")
            print(f"Total Errors: {playback_summary.get('total_errors', 0)}")
            
            # Print recent errors
            if self.playback_monitor.error_events:
                print("\nRecent Errors:")
                for error in self.playback_monitor.error_events[-5:]:
                    print(f"  - {error['context']}: {error['error']}")
    
    async def _monitor_playback_health(self, duration_seconds: int):
        """Monitor playback health during live playback"""
        end_time = time.time() + duration_seconds
        last_broadcast_count = 0
        stuck_count = 0
        
        while time.time() < end_time:
            await asyncio.sleep(2.0)
            
            # Check if broadcasts are progressing
            current_broadcasts = len(self.playback_monitor.broadcast_timings)
            if current_broadcasts == last_broadcast_count:
                stuck_count += 1
                if stuck_count >= 3:
                    self.playback_monitor.log_event('playback_stuck', {
                        'stuck_duration': stuck_count * 2,
                        'last_broadcast_count': last_broadcast_count
                    })
                    print(f"WARNING:  Playback appears stuck (no broadcasts for {stuck_count * 2}s)")
            else:
                stuck_count = 0
            
            last_broadcast_count = current_broadcasts
            
            # Check playback health
            if hasattr(self.playback, '_is_healthy') and not self.playback._is_healthy():
                self.playback_monitor.log_event('playback_unhealthy', {
                    'consecutive_failures': self.playback.consecutive_failures
                })
                print(f"WARNING:  Playback unhealthy (failures: {self.playback.consecutive_failures})")
    
    def save_diagnostic_report(self, filename: str):
        """Save diagnostic report to file"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_summary': self.system_monitor.get_summary(),
            'playback_summary': self.playback_monitor.get_summary(),
            'events': self.playback_monitor.events,
            'errors': self.playback_monitor.error_events,
            'step_timings': self.playback_monitor.step_timings,
            'broadcast_timings': self.playback_monitor.broadcast_timings
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Diagnostic report saved to: {filename}")


async def main():
    """Main diagnostic function"""
    print("=== Checkpoint Playback Freeze Diagnostics ===\n")
    
    # Create diagnostics instance
    diagnostics = FreezeDiagnostics()
    
    # Setup for real system
    try:
        diagnostics.setup_for_real_system()
        print("OK: Connected to real system")
    except Exception as e:
        print(f"ERROR: Failed to connect to real system: {e}")
        print("Make sure the backend is running and accessible")
        return 1
    
    # Get list of available checkpoints
    try:
        checkpoints = diagnostics.checkpoint_manager.list_checkpoints()
        if not checkpoints:
            print("ERROR: No checkpoints found")
            return 1
        
        print(f"Found {len(checkpoints)} checkpoints")
        
        # Use the first checkpoint for testing
        checkpoint_id = checkpoints[0]['id']
        print(f"Using checkpoint: {checkpoint_id}")
        
    except Exception as e:
        print(f"ERROR: Failed to list checkpoints: {e}")
        return 1
    
    # Run diagnostics
    try:
        # Test checkpoint loading
        success = await diagnostics.diagnose_checkpoint_loading(checkpoint_id)
        if not success:
            print("ERROR: Checkpoint loading failed - cannot continue")
            return 1
        
        # Test single game
        success = await diagnostics.diagnose_single_game(checkpoint_id)
        if not success:
            print("ERROR: Single game failed - but continuing with live playback test")
        
        # Test live playback (this is where freezing likely occurs)
        success = await diagnostics.diagnose_live_playback(checkpoint_id, duration_seconds=20)
        
        # Save diagnostic report
        report_filename = f"freeze_diagnostics_{int(time.time())}.json"
        diagnostics.save_diagnostic_report(report_filename)
        
        if success:
            print("\nOK: All diagnostics completed successfully!")
            print("If freezing occurs, check the diagnostic report for details.")
        else:
            print("\nERROR: Diagnostics detected issues!")
            print("Check the diagnostic report for detailed analysis.")
            return 1
    
    except Exception as e:
        print(f"ERROR: Diagnostic error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 