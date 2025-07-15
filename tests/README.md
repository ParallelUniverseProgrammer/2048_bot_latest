# Tests Directory

This directory contains all test files for the 2048 AI Bot project.

## Test Files

### Original Tests
- **test_playback_sim.py** - Tests checkpoint playback simulation functionality
- **mobile_test.py** - Tests mobile connectivity to frontend and backend services
- **test_gpu_usage.py** - Tests GPU usage during model training

### New Freeze Diagnostic Tests
- **test_checkpoint_loading.py** - Tests checkpoint loading scenarios that might cause freezing
- **test_game_simulation.py** - Tests game playing and action selection logic for freezing issues
- **test_live_playback.py** - Tests live playback WebSocket broadcasting functionality
- **test_freeze_diagnostics.py** - Comprehensive diagnostic tool for identifying freeze locations

## Running Tests

### Quick Start - Run All Tests
```bash
# Run all tests (recommended)
python tests/run_all_tests.py

# Run all tests including real system diagnostics (requires backend running)
python tests/run_all_tests.py --real-system

# Run quick tests only (faster feedback)
python tests/run_all_tests.py --quick
```

### Individual Tests
To run individual tests, execute them from the project root directory:

```bash
# Original tests
python tests/test_playback_sim.py
python tests/mobile_test.py
python tests/test_gpu_usage.py

# New freeze diagnostic tests
python tests/test_checkpoint_loading.py
python tests/test_game_simulation.py
python tests/test_live_playback.py
python tests/test_freeze_diagnostics.py
```

### Test Categories

#### Checkpoint Loading Tests (`test_checkpoint_loading.py`)
- Tests various checkpoint loading scenarios (missing files, corrupted data, large models)
- Detects freezing during model loading
- Tests concurrent checkpoint loading

#### Game Simulation Tests (`test_game_simulation.py`)
- Tests action selection with problematic models
- Tests stuck game environments
- Tests memory usage during gameplay
- Tests concurrent game playing

#### Live Playback Tests (`test_live_playback.py`)
- Tests WebSocket broadcasting scenarios
- Tests recovery mechanisms
- Tests health monitoring
- Tests performance at different speeds

#### Freeze Diagnostics (`test_freeze_diagnostics.py`)
- **Real-time diagnostic tool** that connects to the actual system
- Monitors system resources (CPU, memory, threads)
- Detailed event logging and timing analysis
- Generates comprehensive diagnostic reports
- **Use this when experiencing actual freezing issues**

## Troubleshooting Freezing Issues

If you're experiencing server freezing during checkpoint playback:

1. **Run the diagnostic tool first:**
   ```bash
   python tests/test_freeze_diagnostics.py
   ```

2. **Check the generated diagnostic report** (JSON file with timestamp)

3. **Run specific tests** based on suspected issue:
   - If freezing during loading: `python tests/test_checkpoint_loading.py`
   - If freezing during gameplay: `python tests/test_game_simulation.py`
   - If freezing during broadcasting: `python tests/test_live_playback.py`

All tests are configured to work with the project's import paths and should be run from the root directory. 