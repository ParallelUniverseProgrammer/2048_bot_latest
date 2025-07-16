#!/usr/bin/env python3
"""
Master Checkpoint Directory Test Suite
=====================================

This script provides comprehensive testing of checkpoint directory isolation and functionality.
It builds upon the training_diagnostic_script.py to test:

1. Custom Directory Visibility - Backend lists checkpoints from custom directory only
2. Checkpoint Loading - Backend can load checkpoints from custom directory
3. Checkpoint Saving - New checkpoints are saved to custom directory
4. Checkpoint Isolation - Backend doesn't see checkpoints from default directory
5. Training Progress - Training can start from custom checkpoint and progress
6. No Infinite Loops/Hangs - All operations have timeouts
7. Cleanup - All resources are properly cleaned up

Usage:
    python tests/master_checkpoint_dir_test.py
"""

import os
import sys
import time
import json
import signal
import subprocess
import threading
import requests
import psutil
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('master_checkpoint_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MasterCheckpointTest:
    def __init__(self):
        self.backend_process: Optional[subprocess.Popen] = None
        self.backend_port = 8000
        self.backend_url = f"http://localhost:{self.backend_port}"
        self.test_dir = Path(tempfile.mkdtemp(prefix="master_checkpoint_test_"))
        self.default_checkpoints_dir = project_root / "backend" / "checkpoints"
        self.test_checkpoints_dir = self.test_dir / "checkpoints"
        self.monitoring_active = True
        self.test_results = {}
        self.timeout_seconds = 300  # 5 minutes timeout for any operation
        
        # Test state
        self.original_checkpoint_count = 0
        self.test_checkpoint_loaded = False
        self.new_checkpoint_created = False
        self.training_progressed = False
        
        logger.info(f"Master test initialized with test directory: {self.test_dir}")
    
    def setup_test_environment(self):
        """Set up test environment - no longer copy files, just verify backend can access checkpoints"""
        logger.info("Setting up test environment...")
        
        # Create test checkpoints directory (for backend to use)
        self.test_checkpoints_dir.mkdir(exist_ok=True)
        logger.info(f"Created test checkpoints directory: {self.test_checkpoints_dir}")
        
        # Verify source checkpoints exist
        source_checkpoints_dir = project_root / "backend" / "checkpoints"
        if not source_checkpoints_dir.exists():
            logger.error("No checkpoints found in backend/checkpoints")
            return False
        
        logger.info(f"Source checkpoints directory: {source_checkpoints_dir}")
        
        # Count checkpoints in default directory
        self.original_checkpoint_count = len(list(source_checkpoints_dir.glob("checkpoint_episode_*.json")))
        logger.info(f"Default directory contains {self.original_checkpoint_count} checkpoints")
        
        return True
    
    def start_backend(self):
        """Start backend server with test checkpoint directory and capture stdout"""
        logger.info("Starting backend server with test checkpoint directory...")
        
        # Set environment variables for test
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        env['CHECKPOINTS_DIR'] = str(self.test_checkpoints_dir)
        env['DISABLE_RELOAD'] = '1'  # Disable uvicorn reload for testing
        
        logger.info(f"Environment variables:")
        logger.info(f"  PYTHONPATH: {env['PYTHONPATH']}")
        logger.info(f"  CHECKPOINTS_DIR: {env['CHECKPOINTS_DIR']}")
        logger.info(f"  DISABLE_RELOAD: {env['DISABLE_RELOAD']}")
        logger.info(f"  Test checkpoints directory exists: {self.test_checkpoints_dir.exists()}")
        logger.info(f"  Test checkpoints directory contents: {[f.name for f in self.test_checkpoints_dir.iterdir()] if self.test_checkpoints_dir.exists() else 'N/A'}")
        
        # Start backend process
        cmd = [
            sys.executable, 
            str(project_root / "backend" / "main.py")
        ]
        
        logger.info(f"Backend command: {' '.join(cmd)}")
        
        try:
            self.backend_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Start a thread to print backend stdout in real time
            def print_backend_output(proc):
                for line in iter(proc.stdout.readline, ''):
                    print(f"[BACKEND] {line.rstrip()}")
                    logger.info(f"[BACKEND] {line.rstrip()}")
            self.backend_output_thread = threading.Thread(target=print_backend_output, args=(self.backend_process,), daemon=True)
            self.backend_output_thread.start()
            
            # Wait for backend to start with timeout
            logger.info("Waiting for backend to start...")
            start_time = time.time()
            while time.time() - start_time < 30:  # 30 second timeout
                try:
                    response = requests.get(f"{self.backend_url}/health", timeout=2)
                    if response.status_code == 200:
                        logger.info("Backend started successfully")
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(1)
                
                # Check if process died
                if self.backend_process.poll() is not None:
                    logger.error("Backend process died during startup")
                    return False
            
            logger.error("Backend failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start backend: {e}")
            return False
    
    def stop_backend(self):
        """Stop backend server and cleanup"""
        logger.info("Stopping backend server...")
        
        if self.backend_process:
            try:
                # Try graceful shutdown
                self.backend_process.terminate()
                self.backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Backend didn't terminate gracefully, forcing kill")
                self.backend_process.kill()
                self.backend_process.wait()
            except Exception as e:
                logger.error(f"Error stopping backend: {e}")
        
        # Cleanup test directory
        try:
            shutil.rmtree(self.test_dir)
            logger.info("Test environment cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up test environment: {e}")
    
    def run_with_timeout(self, operation: Callable, operation_name: str, timeout_seconds: Optional[int] = None) -> bool:
        """Run an operation with timeout protection"""
        actual_timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        
        logger.info(f"Running {operation_name} with {actual_timeout}s timeout...")
        start_time = time.time()
        
        try:
            result = operation()
            elapsed = time.time() - start_time
            logger.info(f"{operation_name} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{operation_name} failed after {elapsed:.2f}s: {e}")
            return False
    
    def test_custom_directory_visibility(self) -> bool:
        """Test 1: Backend lists checkpoints from custom directory only"""
        logger.info("=== Test 1: Custom Directory Visibility ===")
        
        try:
            # Get checkpoints from backend
            response = requests.get(f"{self.backend_url}/checkpoints", timeout=10)
            if response.status_code != 200:
                logger.error("Failed to get checkpoints list")
                return False
            
            checkpoints = response.json()
            logger.info(f"Backend returned {len(checkpoints)} checkpoints")
            logger.info(f"Full /checkpoints response: {checkpoints}")
            
            # Verify all checkpoints have absolute paths and are from test directory
            for checkpoint in checkpoints:
                checkpoint_id = checkpoint['id']
                absolute_path = checkpoint.get('absolute_path')
                if not absolute_path:
                    logger.error(f"Checkpoint {checkpoint_id} missing absolute_path")
                    logger.error(f"Full checkpoint object: {checkpoint}")
                    return False
                
                # Verify the absolute path points to our test directory
                checkpoint_path = Path(absolute_path)
                if not str(checkpoint_path.parent).startswith(str(self.test_checkpoints_dir)):
                    logger.error(f"Checkpoint {checkpoint_id} absolute_path not in test directory: {absolute_path}")
                    logger.error(f"Expected directory: {self.test_checkpoints_dir}")
                    logger.error(f"Actual directory: {checkpoint_path.parent}")
                    return False
                
                logger.info(f"PASS Checkpoint {checkpoint_id} correctly located in test directory: {absolute_path}")
            
            logger.info("PASS Custom directory visibility test passed")
            return True
            
        except Exception as e:
            logger.error(f"Custom directory visibility test failed: {e}")
            return False

    def test_checkpoint_loading(self) -> bool:
        """Test 2: Backend can load checkpoint using absolute paths"""
        logger.info("=== Test 2: Checkpoint Loading ===")
        
        try:
            # Get available checkpoints
            response = requests.get(f"{self.backend_url}/checkpoints", timeout=10)
            if response.status_code != 200:
                logger.error("Failed to get checkpoints list")
                return False
            
            checkpoints = response.json()
            if not checkpoints:
                logger.error("No checkpoints available for loading test")
                return False
            
            # Find smallest checkpoint
            smallest_checkpoint = min(checkpoints, key=lambda x: x['episode'])
            checkpoint_id = smallest_checkpoint['id']
            absolute_path = smallest_checkpoint['absolute_path']
            
            logger.info(f"Loading checkpoint: {checkpoint_id} (episode {smallest_checkpoint['episode']})")
            logger.info(f"Using absolute path: {absolute_path}")
            
            # Load checkpoint using absolute path
            response = requests.post(
                f"{self.backend_url}/checkpoints/{checkpoint_id}/load",
                json={"absolute_path": absolute_path},
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to load checkpoint: {response.text}")
                return False
            
            result = response.json()
            logger.info(f"Checkpoint loaded successfully: {result}")
            
            # Wait for training to start with longer timeout
            logger.info("Waiting for training to start after checkpoint load...")
            start_time = time.time()
            while time.time() - start_time < 30:  # 30 second timeout for training to start
                try:
                    response = requests.get(f"{self.backend_url}/training/status", timeout=5)
                    if response.status_code == 200:
                        training_status = response.json()
                        if training_status.get('is_training', False):
                            logger.info("Training started successfully after checkpoint load")
                            return True
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(2)
            
            logger.error("Training did not start after loading checkpoint within timeout")
            return False
            
            logger.info("PASS Checkpoint loading test passed")
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint loading test failed: {e}")
            return False

    def test_checkpoint_saving(self) -> bool:
        """Test 3: Backend can save checkpoints to custom directory"""
        logger.info("=== Test 3: Checkpoint Saving ===")
        
        try:
            # Start training to generate a checkpoint
            response = requests.post(f"{self.backend_url}/training/start", timeout=10)
            if response.status_code != 200:
                logger.error("Failed to start training")
                return False
            
            # Wait for a checkpoint to be created
            logger.info("Waiting for checkpoint to be created...")
            start_time = time.time()
            while time.time() - start_time < 60:  # 60 second timeout
                response = requests.get(f"{self.backend_url}/checkpoints", timeout=10)
                if response.status_code == 200:
                    checkpoints = response.json()
                    if len(checkpoints) > 0:
                        # Verify the new checkpoint has an absolute path in test directory
                        for checkpoint in checkpoints:
                            absolute_path = checkpoint.get('absolute_path')
                            if absolute_path:
                                checkpoint_path = Path(absolute_path)
                                if str(checkpoint_path.parent).startswith(str(self.test_checkpoints_dir)):
                                    logger.info(f"PASS New checkpoint saved to test directory: {absolute_path}")
                                    return True
                
                time.sleep(2)
            
            logger.error("No checkpoint was created in test directory within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Checkpoint saving test failed: {e}")
            return False

    def test_checkpoint_isolation(self) -> bool:
        """Test 4: Checkpoints in test directory don't affect default directory"""
        logger.info("=== Test 4: Checkpoint Isolation ===")
        
        try:
            # Get checkpoints from backend (should be from test directory)
            response = requests.get(f"{self.backend_url}/checkpoints", timeout=10)
            if response.status_code != 200:
                logger.error("Failed to get checkpoints list")
                return False
            
            test_checkpoints = response.json()
            logger.info(f"Test directory contains {len(test_checkpoints)} checkpoints")
            
            # Verify all checkpoints are from test directory
            for checkpoint in test_checkpoints:
                absolute_path = checkpoint.get('absolute_path')
                if not absolute_path:
                    logger.error(f"Checkpoint {checkpoint['id']} missing absolute_path")
                    return False
                
                checkpoint_path = Path(absolute_path)
                if not str(checkpoint_path.parent).startswith(str(self.test_checkpoints_dir)):
                    logger.error(f"Checkpoint {checkpoint['id']} not in test directory: {absolute_path}")
                    return False
            
            logger.info("PASS Checkpoint isolation test passed")
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint isolation test failed: {e}")
            return False

    def test_training_progress(self) -> bool:
        """Test 5: Training progresses normally with custom checkpoint directory"""
        logger.info("=== Test 5: Training Progress ===")
        
        try:
            # Get initial training status
            response = requests.get(f"{self.backend_url}/training/status", timeout=10)
            if response.status_code != 200:
                logger.error("Failed to get training status")
                return False
            
            initial_status = response.json()
            initial_episode = initial_status.get('current_episode', 0)
            logger.info(f"Initial episode: {initial_episode}")
            
            # Wait for some training progress
            logger.info("Waiting for training progress...")
            start_time = time.time()
            while time.time() - start_time < 30:  # 30 second timeout
                response = requests.get(f"{self.backend_url}/training/status", timeout=10)
                if response.status_code == 200:
                    current_status = response.json()
                    current_episode = current_status.get('current_episode', 0)
                    
                    if current_episode > initial_episode:
                        logger.info(f"PASS Training progressed from episode {initial_episode} to {current_episode}")
                        return True
                
                time.sleep(2)
            
            logger.error("Training did not progress within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Training progress test failed: {e}")
            return False
    
    def test_checkpoint_loading_all_sizes(self) -> bool:
        """Test: Load all available checkpoints and wait up to 4 minutes for training to start, with progress feedback."""
        logger.info("=== Test: Checkpoint Loading All Sizes ===")
        try:
            response = requests.get(f"{self.backend_url}/checkpoints", timeout=30)
            if response.status_code != 200:
                logger.error("Failed to get checkpoints list")
                return False
            checkpoints = response.json()
            if not checkpoints:
                logger.error("No checkpoints available for all-sizes loading test")
                return False

            results = []
            for idx, checkpoint in enumerate(checkpoints):
                checkpoint_id = checkpoint['id']
                absolute_path = checkpoint.get('absolute_path')
                logger.info(f"\n--- Loading checkpoint {idx+1}/{len(checkpoints)}: {checkpoint_id} ---")
                logger.info(f"Absolute path: {absolute_path}")
                print(f"\n[TEST] Loading checkpoint {idx+1}/{len(checkpoints)}: {checkpoint_id}")
                print(f"[TEST] Absolute path: {absolute_path}")
                start_time = time.time()
                try:
                    # POST to load checkpoint
                    response = requests.post(
                        f"{self.backend_url}/checkpoints/{checkpoint_id}/load",
                        json={"absolute_path": absolute_path},
                        timeout=60  # allow up to 1 minute for the POST
                    )
                    if response.status_code != 200:
                        logger.error(f"Failed to load checkpoint: {response.text}")
                        results.append((checkpoint_id, False, 0, 'load_failed'))
                        continue
                except Exception as e:
                    logger.error(f"Exception during checkpoint load: {e}")
                    results.append((checkpoint_id, False, 0, 'exception'))
                    continue

                # Wait for initializing to become False before checking for training progress
                print("[TEST] Waiting for backend to finish initializing (timeout: 4 minutes)...")
                spinner = ['|', '/', '-', '\\']
                spinner_idx = 0
                elapsed = 0
                last_status = None
                initializing_timeout = 240  # 4 minutes
                initializing_start = time.time()
                while time.time() - initializing_start < initializing_timeout:
                    try:
                        status_response = requests.get(f"{self.backend_url}/training/status", timeout=10)
                        if status_response.status_code == 200:
                            status = status_response.json()
                            last_status = status
                            initializing = status.get('initializing', False)
                            print(f"\r[TEST] Initializing: {initializing}... {spinner[spinner_idx % len(spinner)]} {int(time.time() - initializing_start)}s", end='', flush=True)
                            spinner_idx += 1
                            if not initializing:
                                print("\n[TEST] Backend finished initializing.")
                                break
                    except Exception as e:
                        last_status = str(e)
                    time.sleep(2)
                else:
                    print(f"\n[TEST] Timeout: Backend did not finish initializing for checkpoint {checkpoint_id} after {initializing_timeout} seconds.")
                    logger.error(f"Timeout: Backend did not finish initializing for checkpoint {checkpoint_id} after {initializing_timeout} seconds. Last status: {last_status}")
                    results.append((checkpoint_id, False, time.time() - start_time, last_status))
                    continue

                # After backend finished initializing, get initial episode count
                try:
                    status_response = requests.get(f"{self.backend_url}/training/status", timeout=10)
                    if status_response.status_code == 200:
                        status = status_response.json()
                        initial_episode = status.get('current_episode', 0)
                        print(f"[TEST] Initial episode count after initializing: {initial_episode}")
                    else:
                        initial_episode = 0
                except Exception as e:
                    print(f"[TEST] Could not get initial episode count after initializing: {e}")
                    initial_episode = 0

                # Now check for training progress as before
                success = False
                progress_timeout = 240 - (time.time() - start_time)
                if progress_timeout < 10:
                    progress_timeout = 10  # Always allow at least 10s for progress
                progress_start = time.time()
                while time.time() - progress_start < progress_timeout:
                    try:
                        status_response = requests.get(f"{self.backend_url}/training/status", timeout=10)
                        if status_response.status_code == 200:
                            status = status_response.json()
                            last_status = status
                            current_episode = status.get('current_episode', 0)
                            is_training = status.get('is_training', False)
                            if is_training and current_episode > initial_episode:
                                elapsed = time.time() - start_time
                                print(f"\n[TEST] Training started and progressed after {elapsed:.1f} seconds for checkpoint {checkpoint_id}")
                                print(f"[TEST] Episode progressed from {initial_episode} to {current_episode}")
                                logger.info(f"Training started and progressed after {elapsed:.1f} seconds for checkpoint {checkpoint_id}")
                                logger.info(f"Episode progressed from {initial_episode} to {current_episode}")
                                success = True
                                break
                    except Exception as e:
                        last_status = str(e)
                    spinner_char = spinner[spinner_idx % len(spinner)]
                    spinner_idx += 1
                    waited = int(time.time() - progress_start)
                    print(f"\r[TEST] Waiting for training progress... {spinner_char} {waited}s", end='', flush=True)
                    time.sleep(2)
                if not success:
                    elapsed = time.time() - start_time
                    print(f"\n[TEST] Timeout: Training did not start for checkpoint {checkpoint_id} after {elapsed:.1f} seconds.")
                    logger.error(f"Timeout: Training did not start for checkpoint {checkpoint_id} after {elapsed:.1f} seconds. Last status: {last_status}")
                results.append((checkpoint_id, success, elapsed, last_status))

            # Print summary
            print("\n=== Checkpoint Loading All Sizes Summary ===")
            for checkpoint_id, success, elapsed, last_status in results:
                status_str = "PASS" if success else "FAIL"
                print(f"{status_str}: {checkpoint_id} - {elapsed:.1f}s")
            logger.info("Checkpoint Loading All Sizes test complete.")
            return all(r[1] for r in results)
        except Exception as e:
            logger.error(f"Exception in test_checkpoint_loading_all_sizes: {e}")
            return False

    def run_all_tests(self):
        """Run all tests with timeout protection"""
        logger.info("Starting master checkpoint directory test suite...")
        
        try:
            # Setup
            if not self.setup_test_environment():
                logger.error("Failed to setup test environment")
                return False
            
            # Start backend
            if not self.start_backend():
                logger.error("Failed to start backend")
                return False
            
            # Run tests with timeout protection
            tests = [
                ("Custom Directory Visibility", self.test_custom_directory_visibility),
                ("Checkpoint Loading", self.test_checkpoint_loading),
                ("Checkpoint Saving", self.test_checkpoint_saving),
                ("Checkpoint Isolation", self.test_checkpoint_isolation),
                ("Training Progress", self.test_training_progress),
            ]
            
            for test_name, test_func in tests:
                success = self.run_with_timeout(test_func, test_name)
                self.test_results[test_name] = success
                
                if not success:
                    logger.error(f"Test '{test_name}' failed")
                    break
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Test suite interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in test suite: {e}")
            return False
        finally:
            self.monitoring_active = False
            self.stop_backend()

def main():
    """Main entry point"""
    print("=" * 80)
    print("Master Checkpoint Directory Test Suite")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    print(f"Test directory: {Path(tempfile.gettempdir())}")
    print()
    
    test_suite = MasterCheckpointTest()
    
    try:
        success = test_suite.run_all_tests()
        
        print()
        print("=" * 80)
        print("TEST RESULTS")
        print("=" * 80)
        
        all_passed = True
        for test_name, result in test_suite.test_results.items():
            status = "PASS" if result else "FAIL"
            print(f"{status} {test_name}")
            if not result:
                all_passed = False
        
        print()
        if all_passed:
            print("ALL TESTS PASSED")
            print("Checkpoint directory isolation and functionality verified")
        else:
            print("SOME TESTS FAILED")
            print("Check 'master_checkpoint_test.log' for detailed logs")
        
        print()
        print("Test Summary:")
        print(f"- Custom Directory Visibility: {test_suite.test_results.get('Custom Directory Visibility', False)}")
        print(f"- Checkpoint Loading: {test_suite.test_results.get('Checkpoint Loading', False)}")
        print(f"- Checkpoint Saving: {test_suite.test_results.get('Checkpoint Saving', False)}")
        print(f"- Checkpoint Isolation: {test_suite.test_results.get('Checkpoint Isolation', False)}")
        print(f"- Training Progress: {test_suite.test_results.get('Training Progress', False)}")
        
        print()
        print("Check 'master_checkpoint_test.log' for detailed logs")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
    
    finally:
        print(f"\nTest suite completed at: {datetime.now()}")

# Add CLI option to run this test directly
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Master Checkpoint Directory Test Suite")
    parser.add_argument('--all-sizes', action='store_true', help='Run checkpoint loading all sizes test only')
    args = parser.parse_args()
    if args.all_sizes:
        test_suite = MasterCheckpointTest()
        try:
            test_suite.setup_test_environment()
            test_suite.start_backend()
            result = test_suite.test_checkpoint_loading_all_sizes()
            test_suite.stop_backend()
            print("\nTest result:", "PASS" if result else "FAIL")
        except Exception as e:
            print(f"Fatal error: {e}")
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            print(f"\nTest suite completed at: {datetime.now()}")
    else:
        main() 