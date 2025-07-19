from tests.utilities.backend_manager import requires_real_backend
#!/usr/bin/env python3
"""
Model Studio Integration Test
============================

This test verifies the Model Studio API endpoints functionality:
- Design creation and management
- Architecture validation
- Model compilation
- API response consistency

This test is part of the integration test suite and requires a running backend.
"""

from tests.utilities.test_utils import TestLogger, BackendTester
import requests
import json
import time
from typing import Dict, Any, Optional

class ModelStudioTester:
    """Test class for Model Studio API functionality"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.logger = TestLogger()
        self.backend = BackendTester(base_url, self.logger)
        self.base_url = base_url
@requires_real_backend
    
    def test_design_creation(self) -> Optional[str]:
        """Test creating a new design"""
        self.logger.section("Testing Design Creation")
        
        try:
            response = requests.post(f"{self.base_url}/api/designs", timeout=30)
            if response.status_code == 201:
                design_data = response.json()
                design_id = design_data["id"]
                self.logger.ok(f"Design created successfully: {design_id}")
                return design_id
            else:
                self.logger.error(f"Failed to create design: {response.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"Design creation error: {e}")
            return None
@requires_real_backend
    
    def test_design_update(self, design_id: str) -> bool:
        """Test updating a design with valid architecture"""
        self.logger.section("Testing Design Update")
        
        valid_design = {
            "id": design_id,
            "name": "Test Design",
            "components": [
                {
                    "id": "input_1",
                    "type": "BOARD_INPUT",
                    "props": {"d_model": 128}
                },
                {
                    "id": "transformer_1",
                    "type": "TRANSFORMER_LAYER",
                    "props": {"d_model": 128, "n_heads": 4}
                },
                {
                    "id": "output_1",
                    "type": "ACTION_OUTPUT",
                    "props": {"d_model": 128}
                }
            ],
            "edges": [
                ["input_1", "transformer_1"],
                ["transformer_1", "output_1"]
            ],
            "meta": {
                "d_model": 128,
                "n_heads": 4,
                "n_experts": 4
            }
        }
        
        try:
            response = requests.put(f"{self.base_url}/api/designs/{design_id}", 
                                  json=valid_design, timeout=30)
            if response.status_code == 200:
                self.logger.ok("Design updated successfully")
                return True
            else:
                self.logger.error(f"Failed to update design: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Design update error: {e}")
            return False
@requires_real_backend
    
    def test_design_validation(self, design_id: str) -> Optional[Dict[str, Any]]:
        """Test design validation"""
        self.logger.section("Testing Design Validation")
        
        valid_design = {
            "id": design_id,
            "name": "Test Design",
            "components": [
                {
                    "id": "input_1",
                    "type": "BOARD_INPUT",
                    "props": {"d_model": 128}
                },
                {
                    "id": "transformer_1",
                    "type": "TRANSFORMER_LAYER",
                    "props": {"d_model": 128, "n_heads": 4}
                },
                {
                    "id": "output_1",
                    "type": "ACTION_OUTPUT",
                    "props": {"d_model": 128}
                }
            ],
            "edges": [
                ["input_1", "transformer_1"],
                ["transformer_1", "output_1"]
            ],
            "meta": {
                "d_model": 128,
                "n_heads": 4,
                "n_experts": 4
            }
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/designs/{design_id}/validate", 
                                   json=valid_design, timeout=30)
            if response.status_code == 200:
                validation = response.json()
                self.logger.ok(f"Validation successful: {validation['valid']}")
                self.logger.info(f"Parameters: {validation['paramCount']}M")
                self.logger.info(f"Memory: {validation['estimatedMemory']}MB")
                
                if not validation['valid']:
                    self.logger.warning(f"Validation errors: {validation['errors']}")
                
                return validation
            else:
                self.logger.error(f"Validation failed: {response.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return None
@requires_real_backend
    
    def test_design_compilation(self, design_id: str) -> bool:
        """Test design compilation"""
        self.logger.section("Testing Design Compilation")
        
        try:
            response = requests.post(f"{self.base_url}/api/designs/{design_id}/compile", 
                                   timeout=60)
            if response.status_code == 200:
                compile_result = response.json()
                self.logger.ok("Compilation successful")
                self.logger.info(f"Import path: {compile_result['import_path']}")
                self.logger.info(f"Parameters: {compile_result['paramCount']}M")
                return True
            else:
                self.logger.error(f"Compilation failed: {response.status_code}")
                self.logger.error(f"Response: {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Compilation error: {e}")
            return False
    
    def run_full_test_suite(self) -> bool:
        """Run the complete Model Studio test suite"""
        self.logger.banner("Model Studio Integration Test Suite", 60)
        
        # Test backend connectivity first
        if not self.backend.test_connectivity():
            self.logger.error("Backend not available, skipping Model Studio tests")
            return False
        
        # Step 1: Create design
        self.logger.step(1, 4, "Creating new design")
        design_id = self.test_design_creation()
        if not design_id:
            return False
        
        # Step 2: Update design
        self.logger.step(2, 4, "Updating design architecture")
        if not self.test_design_update(design_id):
            return False
        
        # Step 3: Validate design
        self.logger.step(3, 4, "Validating design")
        validation = self.test_design_validation(design_id)
        if not validation:
            return False
        
        # Step 4: Compile design (only if valid)
        self.logger.step(4, 4, "Compiling design")
        if validation.get('valid', False):
            if not self.test_design_compilation(design_id):
                return False
        else:
            self.logger.warning("Skipping compilation due to validation errors")
        
        self.logger.success("Model Studio test suite completed successfully!")
        return True
@requires_real_backend

def main():
    """Main entry point for Model Studio integration test"""
    logger = TestLogger()
    logger.banner("Model Studio Integration Test", 60)
    
    tester = ModelStudioTester()
    success = tester.run_full_test_suite()
    
    if success:
        logger.success("🎉 Model Studio API test completed successfully!")
        return 0
    else:
        logger.error("❌ Model Studio API test failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 