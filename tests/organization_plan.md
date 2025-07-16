# Tests Directory Organization Plan

## Directory Structure

```
tests/
├── core/                    # Core functionality tests
├── integration/             # Backend integration and API tests
├── performance/             # Performance and optimization tests
├── mobile/                  # Mobile and device compatibility tests
├── playback/                # Game playback and simulation tests
├── training/                # Training-related functionality tests
├── frontend/                # Frontend integration and UI tests
├── utilities/               # Test utilities and mock services
├── runners/                 # Test runners and orchestration scripts
└── checkpoints/             # Test checkpoint data (existing)
```

## File Categorization

### Core Tests (`core/`)
Basic functionality tests that don't require backend:
- `test_checkpoint_loading_fix.py` → `core/test_checkpoint_loading.py`
- `test_checkpoint_loading_fix_verification.py` → `core/test_checkpoint_loading_verification.py`
- `test_checkpoint_loading_fixes.py` → `core/test_checkpoint_loading_fixes.py`
- `test_checkpoint_json_serialization.py` → `core/test_json_serialization.py`
- `test_tiny_model.py` → `core/test_tiny_model.py`
- `test_training_manager.py` → `core/test_training_manager.py`
- `test_training_fix.py` → `core/test_training_fix.py`
- `test_training_issue.py` → `core/test_training_issue.py`

### Integration Tests (`integration/`)
Backend integration and API tests:
- `test_checkpoint_loading.py` → `integration/test_checkpoint_loading.py`
- `test_checkpoint_complete_games.py` → `integration/test_complete_games.py`
- `test_live_playback.py` → `integration/test_live_playback.py`
- `test_game_simulation.py` → `integration/test_game_simulation.py`
- `test_mock_backend_integration.py` → `integration/test_mock_backend.py`
- `test_mock_backend.py` → `integration/test_mock_backend_basic.py`
- `test_edge_cases.py` → `integration/test_edge_cases.py`
- `test_websocket_broadcast_fix.py` → `integration/test_websocket_broadcast.py`

### Performance Tests (`performance/`)
Performance and optimization tests:
- `test_performance_improvements.py` → `performance/test_performance.py`
- `test_gpu_usage.py` → `performance/test_gpu_usage.py`
- `test_speed_control.py` → `performance/test_speed_control.py`
- `test_training_speed.py` → `performance/test_training_speed.py`
- `websocket_performance_optimizer.py` → `performance/websocket_optimizer.py`
- `test_load_balancing_rewards.py` → `performance/test_load_balancing.py`

### Mobile Tests (`mobile/`)
Mobile and device compatibility tests:
- `test_mobile_connection_issues.py` → `mobile/test_connection_issues.py`
- `test_mobile_training_disconnection.py` → `mobile/test_training_disconnection.py`
- `test_device_compatibility.py` → `mobile/test_device_compatibility.py`
- `test_device_fix.py` → `mobile/test_device_fix.py`
- `test_comprehensive_device_fix.py` → `mobile/test_comprehensive_device.py`
- `test_device_error.py` → `mobile/test_device_error.py`
- `mobile_test.py` → `mobile/test_mobile_basic.py`

### Playback Tests (`playback/`)
Game playback and simulation tests:
- `test_real_playback_freeze.py` → `playback/test_freeze_detection.py`
- `test_freeze_diagnostics.py` → `playback/test_freeze_diagnostics.py`
- `test_freeze_reproduction.py` → `playback/test_freeze_reproduction.py`
- `test_playback_controls_fix.py` → `playback/test_controls.py`
- `test_playback_sim.py` → `playback/test_simulation.py`
- `test_real_checkpoint_failure.py` → `playback/test_checkpoint_failure.py`
- `test_checkpoint_playback_failure.py` → `playback/test_playback_failure.py`
- `test_checkpoint_failure_comprehensive.py` → `playback/test_failure_comprehensive.py`
- `test_checkpoint_failure_final.py` → `playback/test_failure_final.py`
- `test_checkpoint_failure_simple.py` → `playback/test_failure_simple.py`

### Training Tests (`training/`)
Training-related functionality tests:
- `test_training_status_sync.py` → `training/test_status_sync.py`
- `test_training_status_sync_simple.py` → `training/test_status_sync_simple.py`
- `test_training_reconnection_failure.py` → `training/test_reconnection_failure.py`
- `test_real_training_reconnection.py` → `training/test_reconnection.py`
- `training_diagnostic_script.py` → `training/diagnostic_script.py`

### Frontend Tests (`frontend/`)
Frontend integration and UI tests:
- `test_frontend_automation.py` → `frontend/test_automation.py`
- `test_browser_simulation.py` → `frontend/test_browser_simulation.py`
- `test_browser_simulation_enhanced.py` → `frontend/test_browser_enhanced.py`
- `test_pwa_install.py` → `frontend/test_pwa_install.py`
- `test_pwa_redirection.py` → `frontend/test_pwa_redirection.py`

### Utilities (`utilities/`)
Test utilities and mock services:
- `test_utils.py` → `utilities/test_utils.py`
- `mock_backend.py` → `utilities/mock_backend.py`
- `backend_availability_manager.py` → `utilities/backend_manager.py`
- `create_test_checkpoint.py` → `utilities/create_checkpoint.py`
- `fix_emojis.py` → `utilities/fix_emojis.py`

### Runners (`runners/`)
Test runners and orchestration scripts:
- `run_all_tests.py` → `runners/run_all.py`
- `run_checkpoint_tests.py` → `runners/run_checkpoint.py`
- `comprehensive_checkpoint_test_plan.py` → `runners/comprehensive_plan.py`
- `master_checkpoint_dir_test.py` → `runners/master_test.py`

### Documentation and Results
Keep in root directory:
- `README.md` → Updated with new organization
- `CHECKPOINT_TESTING_PLAN.md` → Updated with new structure
- All `*_test_results.json` files → Keep in root for easy access

## Naming Conventions

### File Names
- Use descriptive, consistent naming
- Prefix with category when helpful
- Use underscores for spaces
- Keep names concise but clear

### Import Updates
- Update all import statements to reflect new paths
- Use relative imports within categories
- Update `sys.path` modifications in runners

### Backend Dependencies
- Tests using `check_backend_or_start_mock()` → Can run without manual backend
- Tests requiring manual backend → Document clearly in README
- Mock backend integration → Available for all integration tests

## Implementation Steps

1. Create directory structure ✅
2. Move files to appropriate directories
3. Rename files according to plan
4. Update import statements
5. Update documentation
6. Test that everything still works
7. Update README with new organization 