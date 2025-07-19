#!/usr/bin/env python3
"""
Test script for Model Studio API endpoints
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_model_studio_api():
    """Test the Model Studio API endpoints"""
    print("🧪 Testing Model Studio API...")
    
    # Test 1: Create a new design
    print("\n1. Creating new design...")
    response = requests.post(f"{BASE_URL}/api/designs")
    if response.status_code == 201:
        design_data = response.json()
        design_id = design_data["id"]
        print(f"✅ Design created: {design_id}")
    else:
        print(f"❌ Failed to create design: {response.status_code}")
        return
    
    # Test 2: Create a valid design architecture
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
    
    # Test 3: Update the design
    print("\n2. Updating design...")
    response = requests.put(f"{BASE_URL}/api/designs/{design_id}", json=valid_design)
    if response.status_code == 200:
        print("✅ Design updated successfully")
    else:
        print(f"❌ Failed to update design: {response.status_code}")
        return
    
    # Test 4: Validate the design
    print("\n3. Validating design...")
    response = requests.post(f"{BASE_URL}/api/designs/{design_id}/validate", json=valid_design)
    if response.status_code == 200:
        validation = response.json()
        print(f"✅ Validation result: {validation['valid']}")
        print(f"   Parameters: {validation['paramCount']}M")
        print(f"   Memory: {validation['estimatedMemory']}MB")
        if not validation['valid']:
            print(f"   Errors: {validation['errors']}")
    else:
        print(f"❌ Validation failed: {response.status_code}")
        return
    
    # Test 5: Compile the design (only if valid)
    if validation['valid']:
        print("\n4. Compiling design...")
        response = requests.post(f"{BASE_URL}/api/designs/{design_id}/compile")
        if response.status_code == 200:
            compile_result = response.json()
            print(f"✅ Compilation successful")
            print(f"   Import path: {compile_result['import_path']}")
            print(f"   Parameters: {compile_result['paramCount']}M")
        else:
            print(f"❌ Compilation failed: {response.status_code}")
            print(f"   Response: {response.text}")
    
    print("\n🎉 Model Studio API test completed!")

if __name__ == "__main__":
    try:
        test_model_studio_api()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the backend is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed with error: {e}") 