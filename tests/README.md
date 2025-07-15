# Checkpoint System Test Suite

This directory contains a comprehensive test suite for the 2048 AI checkpoint system. The test suite has been refactored to provide consistent, maintainable, and thorough testing of all system components.

## Test Architecture

### Core Components

#### `test_utils.py`
The foundation of the test suite, providing:
- **TestLogger**: Standardized logging with colors, formatting, and structured output
- **BackendTester**: Common backend API testing functionality
- **GameTester**: Game playback testing utilities
- **PlaybackTester**: Live playback and control testing

#### Message Prefixes
All test output uses standardized prefixes for easy parsing and visual scanning:
- `OK:` - Successful operations (green)
- `ERROR:` - Failed operations (red)
- `WARNING:` - Potential issues (yellow)
- `INFO:` - Informational messages (blue)
- `GAME:` - Game-related operations (magenta)
- `STARTING:` - Test suite initialization (cyan)
- `RUNNING:` - Script execution (blue)
- `TESTING:` - Test operations (blue)
- `CONTROLS:` - Playback controls (magenta)
- `PLAYBACK:` - Live playback operations (cyan)
- `STEP:` - Multi-step process indicators (cyan)
- `PROGRESS:` - Progress indicators (cyan)
- `SUCCESS:` - Major success messages (green)

## Test Categories

### 1. Core Functionality Tests

#### `test_checkpoint_loading_fix.py`
Tests the core checkpoint loading functionality:
- Checkpoint loading speed and reliability
- Stats endpoint functionality
- Training independence (checkpoints load during training)
- Basic frontend integration points

#### `test_checkpoint_complete_games.py`
Comprehensive game playback testing:
- Complete game execution from checkpoints
- Game data validation and structure
- Performance metrics and thresholds
- Live playback initialization
- Playback controls (pause/resume/stop)

#### `run_checkpoint_tests.py`
Main test runner with multiple test levels:
- **Basic**: Quick connectivity and API tests
- **Core**: Checkpoint loading and game playback
- **Full**: All core functionality plus live playback
- **Comprehensive**: All tests including performance and edge cases

### 2. Frontend Integration Tests

#### `test_frontend_automation.py`
Enhanced frontend testing with:
- Automated API endpoint verification
- WebSocket connectivity testing
- Data consistency validation
- Comprehensive manual testing checklist
- Cross-browser compatibility guide
- Performance testing instructions

### 3. Edge Case and Robustness Tests

#### `test_edge_cases.py`
Comprehensive edge case testing:
- Invalid checkpoint IDs (including security tests)
- Malformed HTTP requests and payloads
- Concurrent operations and race conditions
- Resource limits and rapid requests
- Error recovery scenarios
- Boundary conditions and limits

### 4. Specialized Tests

#### Performance Tests
- `test_performance_improvements.py` - Performance optimization validation
- `test_gpu_usage.py` - GPU utilization testing

#### Freeze and Stability Tests
- `test_real_playback_freeze.py` - Real-world freeze detection
- `test_freeze_diagnostics.py` - Freeze diagnostic tools
- `test_freeze_reproduction.py` - Freeze reproduction scenarios

#### Device Compatibility Tests
- `test_device_compatibility.py` - Multi-device testing
- `test_device_fix.py` - Device-specific fixes
- `test_comprehensive_device_fix.py` - Comprehensive device testing

#### Live System Tests
- `test_live_playback.py` - Live playback functionality
- `test_game_simulation.py` - Game simulation testing
- `test_playback_sim.py` - Playback simulation

## Usage Guide

### Running Tests

#### Quick Start
```bash
# Run core tests (recommended for regular testing)
python run_checkpoint_tests.py --level core

# Run all tests
python run_checkpoint_tests.py --level comprehensive
```

#### Individual Test Scripts
```bash
# Test checkpoint loading
python test_checkpoint_loading_fix.py

# Test complete game playback
python test_checkpoint_complete_games.py

# Test frontend integration
python test_frontend_automation.py

# Test edge cases
python test_edge_cases.py
```

#### Test Levels
- **basic**: Backend connectivity and API endpoints (~1 minute)
- **core**: Checkpoint loading and game playback (~5 minutes)
- **full**: Core tests plus live playback (~10 minutes)
- **comprehensive**: All tests including edge cases (~15 minutes)

### Prerequisites

1. **Backend Server**: Must be running on `http://localhost:8000`
   ```bash
   cd backend
   python main.py
   ```

2. **Checkpoints**: At least one checkpoint must be available for testing

3. **Python Dependencies**: All required packages must be installed
   ```bash
   pip install requests
   ```

### Test Output Format

The test suite uses structured, colorized output for easy reading:

```
[10:30:45] STARTING: Complete Game Playback Test Suite
============================================================
[10:30:45] STEP: Step 1/7: Testing backend connectivity
  [10:30:45] OK: Backend is accessible
[10:30:47] STEP: Step 2/7: Getting available checkpoints
  [10:30:47] OK: Found 5 checkpoints
...
============================================================
[10:32:15] SUCCESS: COMPLETE GAME PLAYBACK TEST SUITE PASSED!

Test Component                | Status
------------------------------|--------------------
Backend Connectivity         | PASS
Checkpoint Availability      | PASS
Checkpoint Loading           | PASS
Single Game Playback         | PASS
Live Playback Start          | PASS
Playback Controls            | PASS
Performance                  | PASS
```

## Development Guidelines

### Adding New Tests

1. **Use Shared Utilities**: Import from `test_utils.py` for consistent logging and common functionality
2. **Follow Naming Conventions**: Use descriptive test names with standardized prefixes
3. **Include Documentation**: Add comprehensive docstrings and comments
4. **Handle Errors Gracefully**: Provide clear error messages and recovery instructions
5. **Test Edge Cases**: Include boundary conditions and error scenarios

### Test Structure Template

```python
#!/usr/bin/env python3
"""
Test Description
===============

Brief description of what this test covers and why it's important.
"""

from test_utils import TestLogger, BackendTester

class YourTester:
    """Test class description"""
    
    def __init__(self):
        self.logger = TestLogger()
        self.backend = BackendTester()
    
    def test_specific_functionality(self) -> bool:
        """Test a specific piece of functionality"""
        self.logger.banner("Testing Specific Functionality", 60)
        
        # Test implementation
        result = self.backend.test_connectivity()
        
        if result:
            self.logger.ok("Test passed")
            return True
        else:
            self.logger.error("Test failed")
            return False

def main():
    """Main entry point"""
    logger = TestLogger()
    logger.banner("Your Test Suite", 60)
    
    tester = YourTester()
    success = tester.test_specific_functionality()
    
    if success:
        logger.success("All tests passed!")
    else:
        logger.error("Some tests failed!")

if __name__ == "__main__":
    main()
```

### Best Practices

1. **Consistent Logging**: Use the TestLogger class for all output
2. **Error Handling**: Catch and report exceptions gracefully
3. **Timeouts**: Use appropriate timeouts for network operations
4. **Cleanup**: Clean up any temporary files or resources
5. **Documentation**: Document test purpose, expected behavior, and failure conditions

## Troubleshooting

### Common Issues

#### Backend Not Running
```
ERROR: Backend server is not running!
Please start the backend server first:
   cd backend
   python main.py
```

#### No Checkpoints Available
```
ERROR: No checkpoints available for testing
```
- Train the model to generate checkpoints
- Check the checkpoints directory

#### Test Timeouts
- Increase timeout values in test configuration
- Check system performance and resource usage
- Verify network connectivity

#### Permission Errors
- Ensure proper file permissions
- Run tests from the correct directory
- Check for file locks or concurrent access

### Getting Help

1. **Check Logs**: Review test output for specific error messages
2. **Run Individual Tests**: Isolate issues by running specific test scripts
3. **Verify Prerequisites**: Ensure backend is running and checkpoints exist
4. **Check Documentation**: Review this README and inline documentation

## Contributing

When contributing to the test suite:

1. Follow the established patterns and conventions
2. Add comprehensive tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass before submitting
5. Include edge cases and error scenarios

## Test Results Interpretation

### Success Indicators
- All tests show "PASS" status
- No ERROR messages in output
- Performance metrics within acceptable ranges
- Clean test completion without exceptions

### Failure Indicators
- Any test shows "FAIL" status
- ERROR messages in output
- Timeouts or crashes
- Performance below thresholds

### Performance Benchmarks
- Checkpoint loading: < 5 seconds
- Game playback: > 0.5 steps/second
- API response times: < 2 seconds
- Memory usage: Stable over time

The test suite is designed to be comprehensive, maintainable, and developer-friendly. It provides clear feedback on system health and helps ensure the reliability of the checkpoint system. 