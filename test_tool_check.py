#!/usr/bin/env python3
import subprocess
import sys

def check_tool(tool_name: str) -> bool:
    """Check if a command-line tool is available"""
    try:
        print(f"Testing {tool_name}...")
        result = subprocess.run([tool_name, "--version"], 
                              capture_output=True, timeout=10)
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout.decode()}")
        print(f"Stderr: {result.stderr.decode()}")
        return result.returncode == 0
    except Exception as e:
        print(f"Exception: {e}")
        return False

if __name__ == "__main__":
    tools = ["npm", "poetry", "node"]
    for tool in tools:
        print(f"\n{'='*50}")
        success = check_tool(tool)
        print(f"Result: {'✓' if success else '✗'}") 