#!/usr/bin/env python3
"""
Training Diagnostic Script
==========================

This script runs a self-contained backend instance and monitors the training process
to identify where the system hangs or crashes. It performs the following:

1. Starts a real backend server in a separate process
2. Loads the smallest available checkpoint (episode 50)
3. Begins training and monitors progress
4. Tracks until episode 100 when first checkpoint should be created
5. Performs additional diagnostics if issues arise
6. Cleans up the backend process

Usage:
    python tests/training_diagnostic_script.py
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
from typing import Optional, Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.utilities.test_utils import TestLogger

class TrainingDiagnostic:
    def __init__(self):
        self.logger = TestLogger()
        self.backend_process: Optional[subprocess.Popen] = None
        self.backend_port = 8000
        self.backend_url = f"http://localhost:{self.backend_port}"
        self.test_dir = Path(tempfile.mkdtemp(prefix="training_diagnostic_"))
        self.backend_log_file = self.test_dir / "backend.log"
        self.monitoring_active = True
        self.training_started = False
        self.current_episode = 0
        self.last_progress_time = time.time()
        self.progress_timeout = 300  # 5 minutes timeout
        self.checkpoint_created = False
        
        # Copy smallest checkpoint to test directory
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Set up test environment with smallest checkpoint"""
        self.logger.info("Setting up test environment...")
        
        # Create checkpoints directory in test environment
        test_checkpoints_dir = self.test_dir / "checkpoints"
        test_checkpoints_dir.mkdir(exist_ok=True)
        
        # Find and copy smallest checkpoint
        source_checkpoints_dir = project_root / "backend" / "checkpoints"
        if not source_checkpoints_dir.exists():
            self.logger.error("No checkpoints found in backend/checkpoints")
            return False
        
        # Find smallest checkpoint (episode 50)
        smallest_checkpoint = None
        for checkpoint_file in source_checkpoints_dir.glob("checkpoint_episode_*.json"):
            episode_num = int(checkpoint_file.stem.split("_")[-1])
            if smallest_checkpoint is None or episode_num < smallest_checkpoint[1]:
                smallest_checkpoint = (checkpoint_file, episode_num)
        
        if not smallest_checkpoint:
            self.logger.error("No valid checkpoints found")
            return False
        
        checkpoint_file, episode_num = smallest_checkpoint
        self.logger.info(f"Using checkpoint from episode {episode_num}")
        
        # Copy checkpoint files
        checkpoint_base = checkpoint_file.stem
        for ext in ['.json', '.pt']:
            source_file = source_checkpoints_dir / f"{checkpoint_base}{ext}"
            dest_file = test_checkpoints_dir / f"{checkpoint_base}{ext}"
            if source_file.exists():
                shutil.copy2(source_file, dest_file)
                self.logger.info(f"Copied {source_file.name} to test environment")
        
        return True
    
    def start_backend(self):
        """Start backend server in separate process"""
        self.logger.info("Starting backend server...")
        
        # Set environment variables for test
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        env['CHECKPOINTS_DIR'] = str(self.test_dir / "checkpoints")
        
        # Start backend process
        cmd = [
            sys.executable, 
            str(project_root / "backend" / "main.py")
        ]
        
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
            
            # Wait for backend to start
            self.logger.info("Waiting for backend to start...")
            for attempt in range(30):  # 30 second timeout
                try:
                    response = requests.get(f"{self.backend_url}/health", timeout=2)
                    if response.status_code == 200:
                        self.logger.ok("Backend started successfully")
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(1)
                
                # Check if process died
                if self.backend_process.poll() is not None:
                    self.logger.error("Backend process died during startup")
                    return False
            
            self.logger.error("Backend failed to start within timeout")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to start backend: {e}")
            return False
    
    def stop_backend(self):
        """Stop backend server and cleanup"""
        self.logger.info("Stopping backend server...")
        
        if self.backend_process:
            try:
                # Try graceful shutdown
                self.backend_process.terminate()
                self.backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning("Backend didn't terminate gracefully, forcing kill")
                self.backend_process.kill()
                self.backend_process.wait()
            except Exception as e:
                self.logger.error(f"Error stopping backend: {e}")
        
        # Cleanup test directory
        try:
            shutil.rmtree(self.test_dir)
            self.logger.info("Test environment cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up test environment: {e}")
    
    def load_checkpoint(self):
        """Load the smallest checkpoint for training"""
        self.logger.info("Loading checkpoint for training...")
        
        try:
            # Debug: Check what checkpoints are actually in the test directory
            test_checkpoints_dir = self.test_dir / "checkpoints"
            self.logger.info(f"Test checkpoints directory: {test_checkpoints_dir}")
            if test_checkpoints_dir.exists():
                files = list(test_checkpoints_dir.glob("*"))
                self.logger.info(f"Files in test checkpoints directory: {[f.name for f in files]}")
            else:
                self.logger.error("Test checkpoints directory does not exist!")
            
            # Get available checkpoints
            response = requests.get(f"{self.backend_url}/checkpoints")
            if response.status_code != 200:
                self.logger.error("Failed to get checkpoints list")
                return False
            
            checkpoints = response.json()
            self.logger.info(f"Backend returned {len(checkpoints)} checkpoints")
            if not checkpoints:
                self.logger.error("No checkpoints available")
                return False
            
            # Find smallest checkpoint
            smallest_checkpoint = min(checkpoints, key=lambda x: x['episode'])
            checkpoint_id = smallest_checkpoint['id']
            
            self.logger.info(f"Loading checkpoint: {checkpoint_id} (episode {smallest_checkpoint['episode']})")
            
            # Load checkpoint
            response = requests.post(f"{self.backend_url}/checkpoints/{checkpoint_id}/load")
            if response.status_code != 200:
                self.logger.error(f"Failed to load checkpoint: {response.text}")
                return False
            
            self.logger.ok("Checkpoint loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def start_training(self):
        """Start training process"""
        self.logger.info("Starting training...")
        
        try:
            response = requests.post(f"{self.backend_url}/training/start")
            if response.status_code != 200:
                self.logger.error(f"Failed to start training: {response.text}")
                return False
            
            self.training_started = True
            self.last_progress_time = time.time()
            self.logger.ok("Training started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting training: {e}")
            return False
    
    def monitor_training(self):
        """Monitor training progress until episode 100"""
        self.logger.info("Starting training monitoring...")
        
        while self.monitoring_active:
            try:
                # Get training status
                response = requests.get(f"{self.backend_url}/training/status")
                if response.status_code != 200:
                    self.logger.error("Failed to get training status")
                    break
                
                status = response.json()
                current_episode = status.get('current_episode', 0)
                is_training = status.get('is_training', False)
                
                # Update progress tracking
                if current_episode != self.current_episode:
                    self.current_episode = current_episode
                    self.last_progress_time = time.time()
                    self.logger.info(f"Training progress: Episode {current_episode}")
                
                # Check for timeout
                if time.time() - self.last_progress_time > self.progress_timeout:
                    self.logger.error(f"Training stalled at episode {current_episode} for {self.progress_timeout} seconds")
                    self.perform_stall_diagnostics()
                    break
                
                # Check if training stopped unexpectedly
                if not is_training and self.training_started:
                    self.logger.error("Training stopped unexpectedly")
                    self.perform_stall_diagnostics()
                    break
                
                # Check for checkpoint creation (should happen at episode 100)
                if current_episode >= 100 and not self.checkpoint_created:
                    response = requests.get(f"{self.backend_url}/checkpoints")
                    if response.status_code == 200:
                        checkpoints = response.json()
                        new_checkpoints = [c for c in checkpoints if c['episode'] > 50]
                        if new_checkpoints:
                            self.logger.info(f"Checkpoint created at episode {current_episode}")
                            self.checkpoint_created = True
                            # Continue monitoring for a bit more to ensure stability
                            time.sleep(30)
                            break
                
                # Check if we've reached episode 100
                if current_episode >= 100:
                    self.logger.info("Reached episode 100 - training diagnostic complete")
                    break
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring training: {e}")
                break
    
    def perform_stall_diagnostics(self):
        """Perform additional diagnostics when training stalls"""
        self.logger.info("Performing stall diagnostics...")
        
        try:
            # Check backend health
            response = requests.get(f"{self.backend_url}/health")
            self.logger.info(f"Backend health: {response.status_code}")
            
            # Check model config
            response = requests.get(f"{self.backend_url}/model/config")
            if response.status_code == 200:
                config = response.json()
                self.logger.info(f"Model config: {config}")
            
            # Check websocket stats
            response = requests.get(f"{self.backend_url}/ws/stats")
            if response.status_code == 200:
                stats = response.json()
                self.logger.info(f"WebSocket stats: {stats}")
            
            # Check system resources
            if self.backend_process and self.backend_process.pid:
                process = psutil.Process(self.backend_process.pid)
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()
                self.logger.info(f"Backend process - Memory: {memory_info.rss / 1024 / 1024:.1f}MB, CPU: {cpu_percent:.1f}%")
            else:
                self.logger.warning("Backend process not available for resource monitoring")
            
            # Check for GPU memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024
                    self.logger.info(f"GPU Memory - Allocated: {gpu_memory:.1f}MB, Reserved: {gpu_memory_reserved:.1f}MB")
            except ImportError:
                self.logger.info("PyTorch not available for GPU memory check")
            
            # Get backend logs
            if self.backend_process and self.backend_process.stdout:
                self.logger.info("Recent backend output:")
                # This is a simplified approach - in practice you'd want to capture logs continuously
                self.logger.info("Backend process is still running")
            
        except Exception as e:
            self.logger.error(f"Error during stall diagnostics: {e}")
    
    def run_diagnostic(self):
        """Run the complete diagnostic process"""
        self.logger.info("Starting training diagnostic...")
        
        try:
            # Step 1: Start backend
            if not self.start_backend():
                self.logger.error("Failed to start backend")
                return False
            
            # Step 2: Load checkpoint
            if not self.load_checkpoint():
                self.logger.error("Failed to load checkpoint")
                return False
            
            # Step 3: Start training
            if not self.start_training():
                self.logger.error("Failed to start training")
                return False
            
            # Step 4: Monitor training
            self.monitor_training()
            
            # Step 5: Final status check
            response = requests.get(f"{self.backend_url}/training/status")
            if response.status_code == 200:
                final_status = response.json()
                self.logger.info(f"Final training status: {final_status}")
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("Diagnostic interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during diagnostic: {e}")
            return False
        finally:
            self.monitoring_active = False
            self.stop_backend()

def main():
    """Main entry point"""
    logger = TestLogger()
    logger.banner("2048 Bot Training Diagnostic Script", 60)
    logger.info(f"Started at: {datetime.now()}")
    logger.info(f"Test directory: {Path(tempfile.gettempdir())}")
    
    diagnostic = TrainingDiagnostic()
    
    try:
        success = diagnostic.run_diagnostic()
        
        logger.separator()
        logger.info("DIAGNOSTIC RESULTS")
        logger.separator()
        
        if success:
            logger.success("Training diagnostic completed successfully")
            logger.info(f"Reached episode {diagnostic.current_episode}")
            if diagnostic.checkpoint_created:
                logger.ok("Checkpoint creation verified")
            else:
                logger.warning("No new checkpoint created (may be normal)")
        else:
            logger.error("Training diagnostic failed")
            logger.error(f"Stopped at episode {diagnostic.current_episode}")
        
        return success
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return False
    
    finally:
        logger.info(f"Diagnostic completed at: {datetime.now()}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 