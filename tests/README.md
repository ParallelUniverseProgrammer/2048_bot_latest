# Checkpoint System Test Suite

This directory contains a comprehensive test suite for the 2048 AI checkpoint system. The test suite has been refactored to provide consistent, maintainable, and thorough testing of all system components.

## 📋 TODO

### Code Cleanup
- **Remove redundant backend_manager.py wrapper**: The `utilities/backend_manager.py` file is a simple import wrapper that re-exports from `tests/utilities/backend_manager.py`. This creates confusion and should be removed, with all imports updated to use the direct path.

## 🚀 Backend Decorator Implementation - COMPLETED ✅

The test suite now features a comprehensive backend management system that automatically handles backend requirements for all tests:

### **Backend Decorator System**

#### **Decorator Types**
- **`@requires_real_backend`** - For tests that need the actual backend server running
- **`@requires_mock_backend`** - For tests that can use the mock backend for faster execution

#### **Automatic Backend Management**
- **Real Backend Tests**: Integration, core, and performance tests automatically start the real backend
- **Mock Backend Tests**: Training, mobile, frontend, playback, and runner tests use the mock backend
- **Automatic Cleanup**: Backends are automatically stopped after tests complete
- **Health Checks**: Automatic backend health monitoring and restart capabilities

#### **Implementation Results**
- **97% reduction** in major compliance issues (66 → 3 files)
- **53+ test files** automatically updated with appropriate decorators
- **0 legacy backend logic** remaining in test files
- **Fully automated** backend management across the entire test suite

### **Backend Decorator Utilities**

#### **`apply_backend_decorators.py`**
Automatically applies `@requires_real_backend` or `@requires_mock_backend` decorators to all test files based on their directory location and content analysis.

**Features:**
- File categorization logic based on directory structure
- Content analysis to detect test functions and main functions
- Safe file modification with backup creation
- Progress reporting and error handling
- Dry-run mode for testing

**Usage:**
```bash
# Dry run to see what would be changed
python tests/utilities/apply_backend_decorators.py --dry-run

# Apply decorators to all test files
python tests/utilities/apply_backend_decorators.py
```

#### **`cleanup_legacy_backend.py`**
Removes legacy backend logic from all test files and updates imports to use the new decorators.

**Features:**
- Pattern matching for legacy code detection
- Safe removal of legacy logic while preserving test functionality
- Import statement cleanup and standardization
- Syntax validation after changes
- Comprehensive logging of all modifications

**Usage:**
```bash
# Dry run to see what would be cleaned up
python tests/utilities/cleanup_legacy_backend.py --dry-run

# Clean up legacy backend code
python tests/utilities/cleanup_legacy_backend.py
```

#### **`validate_backend_decorators.py`**
Provides comprehensive validation of the backend decorator implementation.

**Features:**
- Runs the compliance checker to verify zero major issues
- Tests that all decorators are working correctly
- Verifies that real backend tests can start the backend
- Verifies that mock backend tests can use the mock backend
- Tests the backend manager's health checking and restart capabilities
- Generates a final compliance report

**Usage:**
```bash
python tests/utilities/validate_backend_decorators.py
```

### **Backend Decorator Categorization**

The system automatically categorizes test files based on their directory location:

| Directory | Backend Type | Reason |
|-----------|--------------|---------|
| `tests/integration/` | Real Backend | Tests actual backend integration |
| `tests/core/` | Real Backend | Tests core backend functionality |
| `tests/performance/` | Real Backend | Tests performance with real backend |
| `tests/training/` | Mock Backend | Training tests can use mock backend |
| `tests/mobile/` | Mock Backend | Mobile tests don't need real backend |
| `tests/frontend/` | Mock Backend | Frontend tests can use mock backend |
| `tests/playback/` | Mock Backend | Playback tests can use mock backend |
| `tests/runners/` | Mock Backend | Test runners can use mock backend |

### **Benefits of Backend Decorators**

1. **Automated Backend Management**: Tests automatically start the appropriate backend type
2. **Consistent Architecture**: All tests use the same decorator pattern
3. **Improved Reliability**: No more manual backend startup/teardown in tests
4. **Better Test Organization**: Clear separation between real and mock backend tests
5. **Reduced Maintenance**: Centralized backend management logic
6. **Faster Test Execution**: Mock backend tests run much faster than real backend tests

## Test Architecture

### Core Components

#### `utilities/test_utils.py`
The foundation of the test suite, providing:
- **TestLogger**: Standardized logging with colors, formatting, and structured output
- **BackendTester**: Common backend API testing functionality
- **GameTester**: Game playback testing utilities
- **PlaybackTester**: Live playback and control testing

#### `utilities/backend_manager.py`
The backend management system, providing:
- **BackendManager**: Centralized backend lifecycle management
- **requires_real_backend**: Decorator for tests that need the actual backend server
- **requires_mock_backend**: Decorator for tests that can use the mock backend
- **Automatic health checking**: Backend availability monitoring and restart capabilities
- **Mock backend support**: Fast, reliable mock backend for testing without real server

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

#### `core/test_checkpoint_loading.py`
Tests the core checkpoint loading functionality:
- Checkpoint loading speed and reliability
- Stats endpoint functionality
- Training independence (checkpoints load during training)
- Basic frontend integration points

#### `integration/test_complete_games.py`
Comprehensive game playback testing:
- Complete game execution from checkpoints
- Game data validation and structure
- Performance metrics and thresholds
- Live playback initialization
- Playback controls (pause/resume/stop)

#### `runners/master_test_runner.py`
Main test runner with multiple test levels:
- **Basic**: Quick connectivity and API tests
- **Core**: Checkpoint loading and game playback
- **Full**: All core functionality plus live playback
- **Comprehensive**: All tests including performance and edge cases

### 2. Frontend Integration Tests

#### `frontend/test_automation.py`
Enhanced frontend testing with:
- Automated API endpoint verification
- WebSocket connectivity testing
- Data consistency validation
- Comprehensive manual testing checklist
- Cross-browser compatibility guide
- Performance testing instructions

### 3. Edge Case and Robustness Tests

#### `integration/test_edge_cases.py`
Comprehensive edge case testing:
- Invalid checkpoint IDs (including security tests)
- Malformed HTTP requests and payloads
- Concurrent operations and race conditions
- Resource limits and rapid requests
- Error recovery scenarios
- Boundary conditions and limits

### 4. Specialized Tests

#### Performance Tests
- `performance/test_performance.py` - Performance optimization validation
- `performance/test_gpu_usage.py` - GPU utilization testing

#### Freeze and Stability Tests
- `playback/test_freeze_detection.py` - Real-world freeze detection
- `playback/test_freeze_diagnostics.py` - Freeze diagnostic tools
- `playback/test_freeze_reproduction.py` - Freeze reproduction scenarios

#### Device Compatibility Tests
- `mobile/test_device_compatibility.py` - Multi-device testing
- `mobile/test_device_fix.py` - Device-specific fixes
- `mobile/test_comprehensive_device.py` - Comprehensive device testing

#### Live System Tests
- `integration/test_live_playback.py` - Live playback functionality
- `integration/test_game_simulation.py` - Game simulation testing
- `playback/test_simulation.py` - Playback simulation

## Usage Guide

### Running Tests

#### 🚀 **Backend Decorator Management**

The test suite now includes automated utilities for managing backend decorators:

```bash
# Apply backend decorators to all test files
python tests/utilities/apply_backend_decorators.py

# Clean up legacy backend code
python tests/utilities/cleanup_legacy_backend.py

# Validate the backend decorator implementation
python tests/utilities/validate_backend_decorators.py
```

#### 🚀 **Recommended: Use the Master Test Runner**

The preferred way to run tests is through the master test runner, which provides different intensity levels:

```bash
# Quick connectivity test (~1 minute)
python tests/runners/master_test_runner.py --level basic

# Core functionality test - RECOMMENDED for regular testing (~5 minutes)
python tests/runners/master_test_runner.py --level core

# Full functionality test (~10 minutes)
python tests/runners/master_test_runner.py --level full

# Comprehensive test - all tests (~15 minutes)
python tests/runners/master_test_runner.py --level comprehensive

# List all available test levels and their contents
python tests/runners/master_test_runner.py --list
```

**Why use the master test runner?**
- ✅ **Consistent experience**: Same interface for all test levels
- ✅ **Smart test selection**: Each level includes appropriate tests from different categories
- ✅ **Proper reporting**: Detailed results with timing and error information
- ✅ **CI/CD ready**: Proper exit codes for automation
- ✅ **Timeout protection**: Prevents tests from hanging indefinitely

#### Individual Test Scripts (Advanced Usage)

For debugging specific issues or running individual tests:

```bash
# Test checkpoint loading
python tests/core/test_checkpoint_loading.py

# Test complete game playback
python tests/integration/test_complete_games.py

# Test frontend integration
python tests/frontend/test_automation.py

# Test edge cases
python tests/integration/test_edge_cases.py
```

#### Test Level Details

| Level | Duration | Description | Use Case |
|-------|----------|-------------|----------|
| **basic** | ~1 minute | Backend connectivity and API endpoints | Quick health check, CI/CD smoke tests |
| **core** | ~5 minutes | Checkpoint loading and game playback | **Regular development testing** |
| **full** | ~10 minutes | All core functionality plus live playback | Pre-commit testing, feature validation |
| **comprehensive** | ~15 minutes | All tests including edge cases | Release testing, full regression suite |

### Prerequisites

1. **Backend Server**: 
   - **Real Backend Tests**: Must be running on `http://localhost:8000` (automatically started by decorators)
   - **Mock Backend Tests**: No backend required (mock backend is automatically started)
   
   ```bash
   # For manual testing without decorators
   cd backend
   python main.py
   ```

2. **Checkpoints**: At least one checkpoint must be available for testing

3. **Python Dependencies**: All required packages must be installed
   ```bash
   pip install requests
   ```

4. **Backend Decorator System**: The test suite now automatically manages backend requirements through decorators, so manual backend management is no longer required for most tests.

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

1. **Use Shared Utilities**: Import from `tests.utilities.test_utils` for consistent logging and common functionality
2. **Follow Naming Conventions**: Use descriptive test names with standardized prefixes
3. **Include Documentation**: Add comprehensive docstrings and comments
4. **Handle Errors Gracefully**: Provide clear error messages and recovery instructions
5. **Test Edge Cases**: Include boundary conditions and error scenarios
6. **Use Backend Decorators**: Apply appropriate backend decorators based on test requirements:
   - Use `@requires_real_backend` for tests that need the actual backend server
   - Use `@requires_mock_backend` for tests that can use the mock backend
   - Import decorators: `from tests.utilities.backend_manager import requires_real_backend, requires_mock_backend`

### Test Structure Template

```python
#!/usr/bin/env python3
"""
Test Description
===============

Brief description of what this test covers and why it's important.
"""

from tests.utilities.test_utils import TestLogger, BackendTester
from tests.utilities.backend_manager import requires_real_backend, requires_mock_backend

class YourTester:
    """Test class description"""
    
    def __init__(self):
        self.logger = TestLogger()
        self.backend = BackendTester()
    
    @requires_mock_backend("Specific Functionality Test")
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

@requires_mock_backend("Your Test Suite")
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
6. **Backend Decorators**: Use appropriate backend decorators for all test functions:
   - Choose `@requires_real_backend` for integration and performance tests
   - Choose `@requires_mock_backend` for unit tests and faster execution
   - Always include a descriptive test name in the decorator
7. **Backend Management**: Let the decorator system handle backend lifecycle automatically

## Troubleshooting

### Common Issues

#### Backend Decorator Issues
```
ERROR: Backend decorator not working
```
- Ensure you've imported the decorators: `from tests.utilities.backend_manager import requires_real_backend, requires_mock_backend`
- Check that the decorator is applied to the correct function
- Verify the test name parameter is provided: `@requires_mock_backend("Test Name")`
- Run the validation script: `python tests/utilities/validate_backend_decorators.py`

#### Backend Not Running
```
ERROR: Backend server is not running!
Please start the backend server first:
   cd backend
   python main.py
```

**Note**: With the backend decorator system, most tests now automatically manage backend requirements. Only run the backend manually if you're testing without decorators.

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

1. **Start with the master test runner**: Use `python tests/runners/master_test_runner.py --level core` for regular testing
2. **Check Logs**: Review test output for specific error messages
3. **Run Individual Tests**: Isolate issues by running specific test scripts
4. **Verify Prerequisites**: Ensure backend is running and checkpoints exist
5. **Check Documentation**: Review this README and inline documentation

## Contributing

When contributing to the test suite:

### 🚨 **MANDATORY PRE-COMMIT CHECKS**

**Before committing ANY test changes, you MUST run both:**

1. **Compliance Checker** - Ensures your test files follow standards:
   ```bash
   python tests/compliance_checker.py
   ```
   - Must show 0 major issues and 0 non-compliant files
   - Fix any minor issues before committing
   - All test files must use TestLogger instead of print()

2. **Master Test Runner** - Verifies your changes work correctly:
   ```bash
   python tests/runners/master_test_runner.py --level core
   ```
   - Must pass all core functionality tests
   - Ensures your changes don't break existing functionality
   - Provides comprehensive validation before commit

### Development Guidelines

1. **Use the master test runner**: Test your changes with `python tests/runners/master_test_runner.py --level core`
2. Follow the established patterns and conventions
3. Add comprehensive tests for new functionality
4. Update documentation as needed
5. Ensure all tests pass before submitting
6. Include edge cases and error scenarios
7. **Always use TestLogger** instead of print() statements
8. **Include proper main() functions** with `__name__ == "__main__"` guards

## Compliance Checker

The compliance checker (`tests/compliance_checker.py`) is an automated tool that ensures all test files follow the established standards. It checks for:

### Compliance Requirements
1. **TestLogger Usage** - All output must use TestLogger instead of print()
2. **Proper Docstrings** - Comprehensive documentation for all test files
3. **Main Functions** - All test files must have main() functions with `__name__ == "__main__"` guards
4. **Error Handling** - Proper try/except blocks and graceful error recovery
5. **Shared Utilities** - Use of common utilities from test_utils
6. **Standardized Prefixes** - Consistent message prefixes (OK:, ERROR:, WARNING:, etc.)

### Running the Compliance Checker
```bash
python tests/compliance_checker.py
```

### Compliance Levels
- **COMPLIANT** - File meets all standards
- **MINOR_ISSUES** - Small issues that should be fixed (print() statements, etc.)
- **MAJOR_ISSUES** - Significant violations (missing TestLogger, no main function, etc.)
- **NON_COMPLIANT** - Critical issues (cannot read file, etc.)

### Fixing Compliance Issues
If you have compliance issues, you can use the automated fix script:
```bash
python tests/fix_compliance_issues.py
```

This script will automatically replace print() statements with TestLogger calls and add missing imports.

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