#!/usr/bin/env python3
"""
Fix Compliance Issues Script
============================

This script automatically fixes compliance issues in test files by replacing
logger.info() statements with appropriate TestLogger calls.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

class ComplianceFixer:
    """Fix compliance issues in test files"""
    
    def __init__(self):
        self.tests_dir = Path(__file__).parent
        self.fixed_files = []
        self.skipped_files = []
        
        # Patterns to replace print() with TestLogger calls
        self.replacement_patterns = [
            # Import error patterns
            (r'print\(f"Import error: {e}"\)', 'logger.error(f"Import error: {e}")'),
            (r'print\(f"Backend path: {backend_path}"\)', 'logger.info(f"Backend path: {backend_path}")'),
            (r'print\(f"Available in backend: {os\.listdir\(backend_path\) if os\.path\.exists\(backend_path\) else \'Path does not exist\'}"\)', 
             'logger.info(f"Available in backend: {os.listdir(backend_path) if os.path.exists(backend_path) else \'Path does not exist\'}")'),
            
            # General print statements with prefixes
            (r'print\(f"   OK: {result\.get\(\'ok\', False\)}"\)', 'logger.ok(f"   OK: {result.get(\'ok\', False)}")'),
            (r'print\("   OK: WebSocket connection successful"\)', 'logger.ok("   OK: WebSocket connection successful")'),
            (r'print\("   WARNING:  websocket-client not installed, skipping WebSocket test"\)', 'logger.warning("   WARNING:  websocket-client not installed, skipping WebSocket test")'),
            (r'print\(f"   ERROR: WebSocket connection failed: {str\(e\)}"\)', 'logger.error(f"   ERROR: WebSocket connection failed: {str(e)}")'),
            
            # Section headers and find patterns
            (r'print\("FIND: Testing Checkpoint Endpoints"\)', 'logger.find("Testing Checkpoint Endpoints")'),
            (r'print\("=" \* 50\)', 'logger.separator()'),
            (r'print\("\\n1\. Testing GET /checkpoints"\)', 'logger.testing("1. Testing GET /checkpoints")'),
            (r'print\("\\n2\. Testing GET /checkpoints/stats"\)', 'logger.testing("2. Testing GET /checkpoints/stats")'),
            (r'print\("\\n3\. Testing GET /checkpoints/playback/status"\)', 'logger.testing("3. Testing GET /checkpoints/playback/status")'),
            (r'print\("\\n4\. Testing GET /checkpoints/playback/current"\)', 'logger.testing("4. Testing GET /checkpoints/playback/current")'),
            (r'print\("\\nFIND: Testing Training Status"\)', 'logger.find("Testing Training Status")'),
            (r'print\("\\n1\. Testing GET /training/status"\)', 'logger.testing("1. Testing GET /training/status")'),
            (r'print\("\\nFIND: Testing WebSocket Connection"\)', 'logger.find("Testing WebSocket Connection")'),
            
            # Status and info patterns
            (r'print\(f"   Status: {result\.get\(\'status_code\', \'N/A\'\)}"\)', 'logger.info(f"   Status: {result.get(\'status_code\', \'N/A\')}")'),
            (r'print\(f"   Response Time: {result\.get\(\'response_time\', \'N/A\'\)}s"\)', 'logger.info(f"   Response Time: {result.get(\'response_time\', \'N/A\')}s")'),
            (r'print\(f"   Checkpoints Found: {len\(result\[\'data\'\]\)}"\)', 'logger.info(f"   Checkpoints Found: {len(result[\'data\'])}")'),
            (r'print\(f"   First Checkpoint: {result\[\'data\'\]\[0\]\.get\(\'id\', \'N/A\'\)}"\)', 'logger.info(f"   First Checkpoint: {result[\'data\'][0].get(\'id\', \'N/A\')}")'),
            (r'print\(f"   Stats: {result\[\'data\'\]}"\)', 'logger.info(f"   Stats: {result[\'data\']}")'),
            (r'print\(f"   Playback Status: {result\[\'data\'\]}"\)', 'logger.info(f"   Playback Status: {result[\'data\']}")'),
            (r'print\(f"   Has Data: {result\[\'data\'\]\.get\(\'has_data\', False\)}"\)', 'logger.info(f"   Has Data: {result[\'data\'].get(\'has_data\', False)}")'),
            (r'print\(f"   Training Status: {result\[\'data\'\]}"\)', 'logger.info(f"   Training Status: {result[\'data\']}")'),
            
            # Error patterns
            (r'print\(f"   Error: {result\[\'error\'\]}"\)', 'logger.error(f"   Error: {result[\'error\']}")'),
            (r'print\(f"   Error: {result\[\'data\'\]\[\'error\'\]}"\)', 'logger.error(f"   Error: {result[\'data\'][\'error\']}")'),
            
            # Test result patterns
            (r'print\("\\nTests interrupted by user"\)', 'logger.warning("Tests interrupted by user")'),
            (r'print\(f"Test suite failed: {str\(e\)}"\)', 'logger.error(f"Test suite failed: {str(e)}")'),
            
            # Success and completion patterns
            (r'print\("\\nMock backend is working correctly!"\)', 'logger.success("Mock backend is working correctly!")'),
            (r'print\("You can now use the mock backend for offline testing\."\)', 'logger.info("You can now use the mock backend for offline testing.")'),
            (r'print\("\\nMock backend has issues that need to be addressed\."\)', 'logger.warning("Mock backend has issues that need to be addressed.")'),
            
            # Results saved patterns
            (r'print\(f"\\nResults saved to \w+\.json"\)', 'logger.info(f"Results saved to results file")'),
            
            # Performance and testing patterns
            (r'print\("Failed to start backend for testing"\)', 'logger.error("Failed to start backend for testing")'),
            (r'print\("\\nAll speed control tests completed successfully!"\)', 'logger.success("All speed control tests completed successfully!")'),
            (r'print\(f"\\n{total - passed}/{total} speed control tests failed"\)', 'logger.warning(f"{total - passed}/{total} speed control tests failed")'),
            
            # Detailed results patterns
            (r'print\("\\nDETAILED RESULTS:"\)', 'logger.info("DETAILED RESULTS:")'),
            (r'print\(json\.dumps\(results, indent=2, default=str\)\)', 'logger.info("Detailed results available in logs")'),
            (r'print\("\\n" \+ "=" \* 60\)', 'logger.separator()'),
            (r'print\("DETAILED RESULTS"\)', 'logger.info("DETAILED RESULTS")'),
            
            # Game simulation patterns
            (r'print\("=== Action Selection Freeze Tests ==="\)', 'logger.banner("Action Selection Freeze Tests")'),
            (r'print\("Test 1: Normal action selection"\)', 'logger.testing("Test 1: Normal action selection")'),
            (r'print\("   \[TIMEOUT\] Normal action selection froze!"\)', 'logger.error("   [TIMEOUT] Normal action selection froze!")'),
            (r'print\(f"   \[ERROR\] {error}"\)', 'logger.error(f"   [ERROR] {error}")'),
            (r'print\("   \[OK\] Normal action selection completed successfully"\)', 'logger.ok("   [OK] Normal action selection completed successfully")'),
            (r'print\(f"   \[FAIL\] Normal action selection failed: {result}"\)', 'logger.error(f"   [FAIL] Normal action selection failed: {result}")'),
            (r'print\("\\nTest 2: No model loaded"\)', 'logger.testing("Test 2: No model loaded")'),
            (r'print\("   \[TIMEOUT\] No model handling froze!"\)', 'logger.error("   [TIMEOUT] No model handling froze!")'),
            (r'print\("   \[OK\] No model handled gracefully"\)', 'logger.ok("   [OK] No model handled gracefully")'),
            (r'print\(f"   \[FAIL\] No model should have returned an error: {result}"\)', 'logger.error(f"   [FAIL] No model should have returned an error: {result}")'),
            (r'print\("\\nTest 3: Slow model inference"\)', 'logger.testing("Test 3: Slow model inference")'),
            (r'print\("   \[OK\] Slow model inference timeout detected and handled correctly!"\)', 'logger.ok("   [OK] Slow model inference timeout detected and handled correctly!")'),
            (r'print\(f"   \[WARN\] ERROR: {error} \(might be expected for slow models\)"\)', 'logger.warning(f"   [WARN] ERROR: {error} (might be expected for slow models)")'),
            (r'print\("   \[OK\] Slow model inference completed successfully"\)', 'logger.ok("   [OK] Slow model inference completed successfully")'),
            (r'print\(f"   \[FAIL\] Slow model inference failed: {result}"\)', 'logger.error(f"   [FAIL] Slow model inference failed: {result}")'),
            (r'print\("\\nTest 4: Model throws exception"\)', 'logger.testing("Test 4: Model throws exception")'),
            (r'print\("   \[TIMEOUT\] Model exception handling froze!"\)', 'logger.error("   [TIMEOUT] Model exception handling froze!")'),
            (r'print\("   \[OK\] Model exception handled gracefully with fallback random actions"\)', 'logger.ok("   [OK] Model exception handled gracefully with fallback random actions")'),
            (r'print\("   \[OK\] Model exception handled gracefully"\)', 'logger.ok("   [OK] Model exception handled gracefully")'),
            (r'print\(f"   \[FAIL\] Model exception should have been handled gracefully: {result}"\)', 'logger.error(f"   [FAIL] Model exception should have been handled gracefully: {result}")'),
            (r'print\("\\nTest 5: Model returns NaN values"\)', 'logger.testing("Test 5: Model returns NaN values")'),
            (r'print\("   \[TIMEOUT\] NaN output handling froze!"\)', 'logger.error("   [TIMEOUT] NaN output handling froze!")'),
            (r'print\(f"   \[WARN\] ERROR: {error} \(might be expected for NaN outputs\)"\)', 'logger.warning(f"   [WARN] ERROR: {error} (might be expected for NaN outputs)")'),
            (r'print\("   \[OK\] NaN output handled gracefully"\)', 'logger.ok("   [OK] NaN output handled gracefully")'),
            (r'print\(f"   \[FAIL\] NaN output handling failed"\)', 'logger.error(f"   [FAIL] NaN output handling failed")'),
            
            # Environment tests
            (r'print\("=== Environment Freeze Tests ==="\)', 'logger.banner("Environment Freeze Tests")'),
            (r'print\("Test 1: Stuck environment \(repeating same state\)"\)', 'logger.testing("Test 1: Stuck environment (repeating same state)")'),
            (r'print\("   \[TIMEOUT\] Stuck environment froze!"\)', 'logger.error("   [TIMEOUT] Stuck environment froze!")'),
            (r'print\(f"   \[WARN\] ERROR: {error}"\)', 'logger.warning(f"   [WARN] ERROR: {error}")'),
            (r'print\("   \[OK\] Stuck environment handled successfully"\)', 'logger.ok("   [OK] Stuck environment handled successfully")'),
            (r'print\(f"       Game had {result\.get\(\'steps\', 0\)} steps"\)', 'logger.info(f"       Game had {result.get(\'steps\', 0)} steps")'),
            (r'print\(f"   \[FAIL\] Stuck environment failed: {result}"\)', 'logger.error(f"   [FAIL] Stuck environment failed: {result}")'),
            (r'print\("\\nTest 2: Infinite environment \(never terminates\)"\)', 'logger.testing("Test 2: Infinite environment (never terminates)")'),
            (r'print\("   \[OK\] Infinite environment timeout detected and handled correctly!"\)', 'logger.ok("   [OK] Infinite environment timeout detected and handled correctly!")'),
            (r'print\("   \[OK\] Infinite environment handled successfully"\)', 'logger.ok("   [OK] Infinite environment handled successfully")'),
            (r'print\(f"       Game had {result\.get\(\'steps\', 0\)} steps \(should hit max_steps limit\)"\)', 'logger.info(f"       Game had {result.get(\'steps\', 0)} steps (should hit max_steps limit)")'),
            (r'print\(f"   \[FAIL\] Infinite environment failed: {result}"\)', 'logger.error(f"   [FAIL] Infinite environment failed: {result}")'),
            (r'print\("\\nTest 3: Environment throws exception"\)', 'logger.testing("Test 3: Environment throws exception")'),
            (r'print\("   \[TIMEOUT\] Environment exception handling froze!"\)', 'logger.error("   [TIMEOUT] Environment exception handling froze!")'),
            (r'print\("   \[OK\] Environment exception handled gracefully"\)', 'logger.ok("   [OK] Environment exception handled gracefully")'),
            (r'print\(f"   \[FAIL\] Environment exception should have failed: {result}"\)', 'logger.error(f"   [FAIL] Environment exception should have failed: {result}")'),
            
            # Concurrent tests
            (r'print\("=== Concurrent Game Tests ==="\)', 'logger.banner("Concurrent Game Tests")'),
            (r'print\("Test: Multiple concurrent games"\)', 'logger.testing("Test: Multiple concurrent games")'),
            (r'print\("   \[TIMEOUT\] Concurrent games froze!"\)', 'logger.error("   [TIMEOUT] Concurrent games froze!")'),
            (r'print\(f"   \[OK\] {success_count}/3 concurrent games completed successfully"\)', 'logger.ok(f"   [OK] {success_count}/3 concurrent games completed successfully")'),
            (r'print\("   \[FAIL\] Concurrent games failed"\)', 'logger.error("   [FAIL] Concurrent games failed")'),
            
            # Memory tests
            (r'print\("=== Memory Usage Tests ==="\)', 'logger.banner("Memory Usage Tests")'),
            (r'print\("Test: Memory usage during repeated games"\)', 'logger.testing("Test: Memory usage during repeated games")'),
            (r'print\(f"   \[TIMEOUT\] Game {i\+1} froze!"\)', 'logger.error(f"   [TIMEOUT] Game {i+1} froze!")'),
            (r'print\(f"   Game {i\+1}: {\'\[OK\]\' if not error else \'\[FAIL\]\'} Memory: {current_memory - initial_memory} bytes"\)', 'logger.info(f"   Game {i+1}: {\'[OK]\' if not error else \'[FAIL]\'} Memory: {current_memory - initial_memory} bytes")'),
            (r'print\(f"   \[WARN\] WARNING: Memory usage increased by {memory_increase/1024/1024:.2f}MB"\)', 'logger.warning(f"   [WARN] WARNING: Memory usage increased by {memory_increase/1024/1024:.2f}MB")'),
            (r'print\(f"   \[OK\] Memory usage stable \(increased by {memory_increase/1024/1024:.2f}MB\)"\)', 'logger.ok(f"   [OK] Memory usage stable (increased by {memory_increase/1024/1024:.2f}MB)")'),
            
            # Final results
            (r'print\("Testing game simulation scenarios that might cause server freezing\.\.\."\)', 'logger.info("Testing game simulation scenarios that might cause server freezing...")'),
            (r'print\("\\n\[OK\] All game simulation tests passed!"\)', 'logger.success("All game simulation tests passed!")'),
            (r'print\("\\n\[FAIL\] Some game simulation tests failed!"\)', 'logger.error("Some game simulation tests failed!")'),
            
            # Next steps
            (r'print\("   1\. Check if any endpoints are timing out"\)', 'logger.info("   1. Check if any endpoints are timing out")'),
            (r'print\("   2\. Verify checkpoint files exist in backend/checkpoints/"\)', 'logger.info("   2. Verify checkpoint files exist in backend/checkpoints/")'),
            (r'print\("   3\. Check backend logs for errors"\)', 'logger.info("   3. Check backend logs for errors")'),
            (r'print\("   4\. Test frontend checkpoint loading with browser dev tools"\)', 'logger.info("   4. Test frontend checkpoint loading with browser dev tools")'),
            
            # Master test patterns
            (r'print\(f"\\n\[TEST\] Loading checkpoint {idx\+1}/{len\(checkpoints\)}: {checkpoint_id}"\)', 'logger.testing(f"Loading checkpoint {idx+1}/{len(checkpoints)}: {checkpoint_id}")'),
            (r'print\(f"\[TEST\] Absolute path: {absolute_path}"\)', 'logger.info(f"Absolute path: {absolute_path}")'),
            (r'print\("\[TEST\] Waiting for backend to finish initializing \(timeout: 4 minutes\)\.\.\."\)', 'logger.info("Waiting for backend to finish initializing (timeout: 4 minutes)...")'),
            (r'print\(f"\\r\[TEST\] Initializing: {initializing}\.\.\. {spinner\[spinner_idx % len\(spinner\)\]} {int\(time\.time\(\) - initializing_start\)}s", end=\'\', flush=True\)', 'logger.progress(f"Initializing: {initializing}... {int(time.time() - initializing_start)}s")'),
            (r'print\("\\n\[TEST\] Backend finished initializing\."\)', 'logger.ok("Backend finished initializing.")'),
            (r'print\(f"\\n\[TEST\] Timeout: Backend did not finish initializing for checkpoint {checkpoint_id} after {initializing_timeout} seconds\."\)', 'logger.error(f"Timeout: Backend did not finish initializing for checkpoint {checkpoint_id} after {initializing_timeout} seconds.")'),
            (r'print\(f"\[TEST\] Initial episode count after initializing: {initial_episode}"\)', 'logger.info(f"Initial episode count after initializing: {initial_episode}")'),
            (r'print\(f"\[TEST\] Could not get initial episode count after initializing: {e}"\)', 'logger.error(f"Could not get initial episode count after initializing: {e}")'),
            (r'print\(f"\\n\[TEST\] Training started and progressed after {elapsed:.1f} seconds for checkpoint {checkpoint_id}"\)', 'logger.ok(f"Training started and progressed after {elapsed:.1f} seconds for checkpoint {checkpoint_id}")'),
            (r'print\(f"\[TEST\] Episode progressed from {initial_episode} to {current_episode}"\)', 'logger.info(f"Episode progressed from {initial_episode} to {current_episode}")'),
            (r'print\(f"\\r\[TEST\] Waiting for training progress\.\.\. {spinner_char} {waited}s", end=\'\', flush=True\)', 'logger.progress(f"Waiting for training progress... {waited}s")'),
            (r'print\(f"\\n\[TEST\] Timeout: Training did not start for checkpoint {checkpoint_id} after {elapsed:.1f} seconds\."\)', 'logger.error(f"Timeout: Training did not start for checkpoint {checkpoint_id} after {elapsed:.1f} seconds.")'),
            (r'print\("\\n=== Checkpoint Loading All Sizes Summary ==="\)', 'logger.banner("Checkpoint Loading All Sizes Summary")'),
            (r'print\(f"{status_str}: {checkpoint_id} - {elapsed:.1f}s"\)', 'logger.info(f"{status_str}: {checkpoint_id} - {elapsed:.1f}s")'),
            
            # Playback test patterns
            (r'print\(f"Client send failed: {e}"\)', 'logger.error(f"Client send failed: {e}")'),
            (r'print\("=== Testing Action Selection Infinite Loop ==="\)', 'logger.banner("Testing Action Selection Infinite Loop")'),
            (r'print\("=== Testing Concurrent Operations ==="\)', 'logger.banner("Testing Concurrent Operations")'),
            
            # Simulation test patterns
            (r'print\("\[TEST\] Playback task hung during shutdown!"\)', 'logger.error("[TEST] Playback task hung during shutdown!")'),
            
            # Mock backend patterns
            (r'print\(f"Mock Backend: {format % args}"\)', 'logger.info(f"Mock Backend: {format % args}")'),
        ]
    
    def fix_file(self, file_path: Path) -> bool:
        """Fix compliance issues in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Check if file already uses TestLogger
            if 'TestLogger' not in content:
                # Add import if needed
                if 'from test_utils import' in content:
                    content = content.replace('from test_utils import', 'from test_utils import TestLogger, ')
                elif 'import test_utils' in content:
                    content = content.replace('import test_utils', 'from test_utils import TestLogger')
                else:
                    # Add import at the top after other imports
                    lines = content.split('\n')
                    import_index = -1
                    for i, line in enumerate(lines):
                        if line.strip().startswith('import ') or line.strip().startswith('from '):
                            import_index = i
                    
                    if import_index >= 0:
                        lines.insert(import_index + 1, 'from test_utils import TestLogger')
                        content = '\n'.join(lines)
                    else:
                        # Add after shebang and docstring
                        lines = content.split('\n')
                        insert_index = 0
                        for i, line in enumerate(lines):
                            if line.strip().startswith('#!/'):
                                insert_index = i + 1
                            elif line.strip().startswith('"""') or line.strip().startswith("'''"):
                                # Find end of docstring
                                j = i + 1
                                while j < len(lines) and ('"""' not in lines[j] or lines[j].count('"""') < 2):
                                    j += 1
                                insert_index = j + 1
                                break
                        
                        lines.insert(insert_index, 'from test_utils import TestLogger')
                        content = '\n'.join(lines)
            
            # Add logger initialization if not present
            if 'logger = TestLogger()' not in content and 'self.logger = TestLogger()' not in content:
                # Look for main function or class methods
                if 'def main():' in content:
                    main_match = re.search(r'def main\(\):\s*\n', content)
                    if main_match:
                        pos = main_match.end()
                        content = content[:pos] + '    logger = TestLogger()\n' + content[pos:]
                elif 'class ' in content:
                    # Add to setup_method or __init__
                    if 'def setup_method(self):' in content:
                        setup_match = re.search(r'def setup_method\(self\):\s*\n', content)
                        if setup_match:
                            pos = setup_match.end()
                            content = content[:pos] + '        self.logger = TestLogger()\n' + content[pos:]
                    elif 'def __init__(self):' in content:
                        init_match = re.search(r'def __init__\(self\):\s*\n', content)
                        if init_match:
                            pos = init_match.end()
                            content = content[:pos] + '        self.logger = TestLogger()\n' + content[pos:]
            
            # Apply replacement patterns
            for pattern, replacement in self.replacement_patterns:
                content = re.sub(pattern, replacement, content)
            
            # Fix any remaining print statements that should use logger
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'logger.info(' in line and not line.strip().startswith('#'):
                    if 'logger.info(' in line:
                        # Replace with appropriate logger call based on content
                        if any(word in line.lower() for word in ['error', 'fail', 'exception']):
                            line = line.replace('logger.error(', 'logger.error(')
                        elif any(word in line.lower() for word in ['warning', 'warn']):
                            line = line.replace('logger.warning(', 'logger.warning(')
                        elif any(word in line.lower() for word in ['ok', 'success', 'passed']):
                            line = line.replace('logger.ok(', 'logger.ok(')
                        elif any(word in line.lower() for word in ['info', 'status', 'data']):
                            line = line.replace('logger.info(', 'logger.info(')
                        else:
                            line = line.replace('logger.info(', 'logger.info(')
                        lines[i] = line
            
            content = '\n'.join(lines)
            
            # Only write if content changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error fixing {file_path}: {e}")
            return False
    
    def fix_all_files(self):
        """Fix compliance issues in all test files"""
        logger.info("Starting compliance fixes...")
        
        for file_path in self.tests_dir.rglob('*.py'):
            if file_path.name != '__init__.py' and file_path.name != 'compliance_checker.py':
                logger.info(f"Processing: {file_path}")
                
                if self.fix_file(file_path):
                    self.fixed_files.append(str(file_path))
                else:
                    self.skipped_files.append(str(file_path))
        
        # Print summary
        logger.info(f"\nFixed {len(self.fixed_files)} files:")
        for file_path in self.fixed_files:
            logger.info(f"  {file_path}")
        
        logger.info(f"\nSkipped {len(self.skipped_files)} files:")
        for file_path in self.skipped_files:
            logger.info(f"  {file_path}")

def main():
    """Main entry point"""
    fixer = ComplianceFixer()
    fixer.fix_all_files()

if __name__ == "__main__":
    main() 