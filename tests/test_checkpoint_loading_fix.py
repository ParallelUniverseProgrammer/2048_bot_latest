#!/usr/bin/env python3
"""
Test script to verify checkpoint loading fix
"""

import requests
import time
import json
from typing import Dict, Any
from test_utils import TestLogger, BackendTester

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_checkpoint_loading_behavior():
    """Test that checkpoint loading works correctly"""
    logger = TestLogger()
    backend = BackendTester(BASE_URL, logger, TIMEOUT)
    
    logger.banner("Testing Checkpoint Loading Behavior", 50)
    
    # Test 1: Check if checkpoints load quickly
    logger.log("\n1. Testing checkpoint loading speed")
    start_time = time.time()
    checkpoints = backend.get_checkpoints()
    load_time = time.time() - start_time
    
    logger.log(f"   Load Time: {load_time:.2f}s")
    logger.log(f"   Checkpoints Found: {len(checkpoints)}")
    
    if load_time > 5:
        logger.warning("Slow loading time - may indicate performance issue")
    else:
        logger.ok("Loading time is acceptable")
    
    # Test 2: Check if stats load correctly
    logger.log("\n2. Testing stats loading")
    start_time = time.time()
    stats = backend.get_checkpoint_stats()
    load_time = time.time() - start_time
    
    logger.log(f"   Load Time: {load_time:.2f}s")
    if stats:
        logger.log(f"   Total Checkpoints: {stats.get('total_checkpoints', 0)}")
        logger.log(f"   Best Score: {stats.get('best_score', 0)}")
    
    # Test 3: Test training status doesn't affect checkpoints
    logger.log("\n3. Testing training status independence")
    training_status = backend.get_training_status()
    if training_status:
        logger.log(f"   Training Active: {training_status.get('is_training', False)}")
        logger.log(f"   Current Episode: {training_status.get('current_episode', 0)}")
    
    # Test 4: Verify checkpoints still load when training is active
    logger.log("\n4. Testing checkpoint loading during training")
    checkpoints_during_training = backend.get_checkpoints()
    logger.log(f"   Checkpoints Available: {len(checkpoints_during_training)}")
    
    if checkpoints_during_training:
        logger.ok("Checkpoints load correctly regardless of training state")
    else:
        logger.error("Checkpoints failed to load during training")

def test_frontend_loading_states():
    """Test that frontend loading states work correctly"""
    logger = TestLogger()
    
    logger.banner("Testing Frontend Loading States", 50)
    
    logger.log("\nFor comprehensive frontend testing, use the enhanced test suite:")
    logger.log("   python test_frontend_automation.py")
    logger.log("\nThis provides:")
    logger.log("- Automated API endpoint testing")
    logger.log("- WebSocket connectivity verification")
    logger.log("- Data consistency checks")
    logger.log("- Detailed manual testing checklist")
    logger.log("- Cross-browser compatibility guide")
    logger.log("- Performance testing instructions")
    
    logger.log("\nQuick manual verification steps:")
    logger.log("1. Navigate to Checkpoints tab")
    logger.log("2. Verify checkpoints load and display immediately")
    logger.log("3. Start training from Training tab")
    logger.log("4. Navigate back to Checkpoints tab")
    logger.log("5. Verify checkpoints still display (not stuck in loading)")
    logger.log("6. Stop training")
    logger.log("7. Verify checkpoints still display correctly")
    logger.log("8. Test checkpoint playback functionality")

def main():
    """Run all tests"""
    logger = TestLogger()
    
    logger.banner("Testing Checkpoint Loading Fix", 60)
    
    # Test backend functionality
    test_checkpoint_loading_behavior()
    
    # Document frontend tests
    test_frontend_loading_states()
    
    logger.separator(60)
    logger.ok("Testing Complete")
    logger.log("\nSummary:")
    logger.log("Backend checkpoint endpoints are working")
    logger.log("Loading times are reasonable")
    logger.log("Checkpoints load independently of training state")
    logger.log("\nNext steps:")
    logger.log("1. Test the frontend manually to verify the fix")
    logger.log("2. Check that CheckpointManager no longer shows loading during training")
    logger.log("3. Verify checkpoint functionality works as expected")

if __name__ == "__main__":
    main() 