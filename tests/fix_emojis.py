#!/usr/bin/env python3
"""
Fix emoji encoding issues in test files
"""

import re
import os

def fix_emojis_in_file(filepath):
    """Replace emojis with text equivalents"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace emojis with text equivalents
    replacements = {
        'OK:': 'OK:',
        'ERROR:': 'ERROR:',
        'WARNING:': 'WARNING:',
        'STATUS:': 'STARTING:',
        'SUCCESS:': 'SUCCESS:',
        'GAME:': 'GAME:',
        'STATUS:': 'PLAYBACK:',
        'STATUS:': 'CONTROLS:',
        'FIND:': 'TESTING:',
        'STATUS:': 'COMPLETE:',
        'ALARM:': 'TIMEOUT:',
        'CRASH:': 'CRASH:',
        'RUNNING:': 'RUNNING:'
    }
    
    for emoji, text in replacements.items():
        content = content.replace(emoji, text)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed emojis in {filepath}")

def main():
    """Fix emojis in all test files"""
    test_files = [
        'tests/test_checkpoint_loading_fix.py',
        'tests/test_checkpoint_complete_games.py',
        'tests/run_checkpoint_tests.py'
    ]
    
    for filepath in test_files:
        if os.path.exists(filepath):
            fix_emojis_in_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main() 