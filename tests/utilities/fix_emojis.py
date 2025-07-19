#!/usr/bin/env python3
"""
Fix Emojis Utility
=================

This utility fixes emoji encoding issues in test files by replacing emojis
with standardized text equivalents. It ensures consistent output formatting
across different environments and prevents encoding issues.

This utility is critical for maintaining consistent test output formatting.
"""

import re
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from tests.utilities.test_utils import TestLogger
from tests.utilities.backend_manager import requires_mock_backend

class EmojiFixer:
    """Utility class for fixing emoji encoding issues"""
    
    def __init__(self, create_backups: bool = True):
        self.logger = TestLogger()
        self.create_backups = create_backups
        
        # Define comprehensive emoji to text replacements
        self.replacements = {
            # Success/Status emojis
            '✅': 'OK:',
            '✓': 'OK:',
            '✔': 'OK:',
            '🎉': 'SUCCESS:',
            '🎊': 'SUCCESS:',
            '🏆': 'SUCCESS:',
            
            # Error/Failure emojis
            '❌': 'ERROR:',
            '✗': 'ERROR:',
            '✘': 'ERROR:',
            '💥': 'CRASH:',
            '💣': 'CRASH:',
            '🔥': 'ERROR:',
            
            # Warning emojis
            '⚠️': 'WARNING:',
            '⚠': 'WARNING:',
            '🚨': 'WARNING:',
            '⚡': 'WARNING:',
            
            # Process/Status emojis
            '🔄': 'STARTING:',
            '🔄': 'RUNNING:',
            '⏳': 'WAITING:',
            '⏰': 'TIMEOUT:',
            '⏱️': 'TIMEOUT:',
            '🏃': 'RUNNING:',
            '🚀': 'STARTING:',
            '⏹️': 'STOPPED:',
            '⏸️': 'PAUSED:',
            '▶️': 'PLAYBACK:',
            '⏯️': 'PLAYBACK:',
            
            # Game/Testing emojis
            '🎮': 'GAME:',
            '🎯': 'TARGET:',
            '🎲': 'GAME:',
            '🧪': 'TESTING:',
            '🔬': 'TESTING:',
            '🔍': 'TESTING:',
            '🔎': 'TESTING:',
            '📊': 'STATUS:',
            '📈': 'PROGRESS:',
            '📉': 'STATUS:',
            
            # Control/Interface emojis
            '🔧': 'FIX:',
            '🔨': 'FIX:',
            '⚙️': 'CONFIG:',
            '🎛️': 'CONTROLS:',
            '🎚️': 'CONTROLS:',
            
            # Completion/Finish emojis
            '🏁': 'COMPLETE:',
            '🎪': 'COMPLETE:',
            '🎭': 'COMPLETE:',
            '🎨': 'COMPLETE:',
            
            # Information/Status emojis
            'ℹ️': 'INFO:',
            '💡': 'INFO:',
            '📝': 'INFO:',
            '📋': 'INFO:',
            '📌': 'INFO:',
            '📍': 'INFO:',
            
            # Network/Connection emojis
            '🌐': 'NETWORK:',
            '📡': 'NETWORK:',
            '🔗': 'CONNECTION:',
            '🔌': 'CONNECTION:',
            '🔋': 'STATUS:',
            
            # File/Data emojis
            '📁': 'FILE:',
            '📂': 'FILE:',
            '📄': 'FILE:',
            '📃': 'FILE:',
            '📑': 'FILE:',
            '💾': 'SAVE:',
            '💿': 'DATA:',
            '📀': 'DATA:',
            
            # Time/Date emojis
            '🕐': 'TIME:',
            '🕑': 'TIME:',
            '🕒': 'TIME:',
            '🕓': 'TIME:',
            '🕔': 'TIME:',
            '🕕': 'TIME:',
            '🕖': 'TIME:',
            '🕗': 'TIME:',
            '🕘': 'TIME:',
            '🕙': 'TIME:',
            '🕚': 'TIME:',
            '🕛': 'TIME:',
            '📅': 'DATE:',
            '📆': 'DATE:',
            '🗓️': 'DATE:',
            
            # Direction/Navigation emojis
            '⬆️': 'UP:',
            '⬇️': 'DOWN:',
            '⬅️': 'LEFT:',
            '➡️': 'RIGHT:',
            '↩️': 'BACK:',
            '↪️': 'FORWARD:',
            '🔄': 'REFRESH:',
            '🔀': 'SHUFFLE:',
            '🔁': 'REPEAT:',
            '🔂': 'REPEAT_ONE:',
            
            # Common test prefixes that might be used
            'FIND:': 'TESTING:',
            'ALARM:': 'TIMEOUT:',
            'CRASH:': 'ERROR:',
            'REFRESH:': 'STARTING:',
            'TARGET:': 'SUCCESS:',
            'COMPLETE:': 'SUCCESS:',
        }
    
    def create_backup(self, filepath: str) -> Optional[str]:
        """Create a backup of the file before modifying it"""
        if not self.create_backups:
            return None
            
        try:
            backup_path = f"{filepath}.backup"
            shutil.copy2(filepath, backup_path)
            self.logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.warning(f"Failed to create backup for {filepath}: {e}")
            return None
    
    def restore_backup(self, filepath: str, backup_path: str) -> bool:
        """Restore file from backup"""
        try:
            shutil.copy2(backup_path, filepath)
            self.logger.info(f"Restored {filepath} from backup")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore {filepath} from backup: {e}")
            return False
    
    def validate_file_content(self, content: str, filepath: str) -> Tuple[bool, str]:
        """Validate that the content is safe to write back"""
        try:
            # Check for basic Python syntax indicators
            if not content.strip():
                return False, "File would be empty after processing"
            
            # Check for basic Python structure
            if 'def ' in content or 'class ' in content or 'import ' in content:
                # Basic Python file structure preserved
                pass
            else:
                # Not a Python file, be more careful
                self.logger.warning(f"File {filepath} doesn't appear to be a Python file")
            
            # Check for potential corruption (excessive replacements)
            original_length = len(content)
            if original_length < 100:  # Very small files
                return True, "Small file, proceeding with caution"
            
            return True, "Content validation passed"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def fix_emojis_in_file(self, filepath: str) -> bool:
        """Replace emojis with text equivalents in a single file"""
        backup_path = None
        
        try:
            self.logger.info(f"Fixing emojis in {filepath}")
            
            # Check if file exists
            if not os.path.exists(filepath):
                self.logger.error(f"File does not exist: {filepath}")
                return False
            
            # Check if file is readable
            if not os.access(filepath, os.R_OK):
                self.logger.error(f"File is not readable: {filepath}")
                return False
            
            # Create backup
            backup_path = self.create_backup(filepath)
            
            # Read file content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            changes_made = 0
            
            # Replace emojis with text equivalents
            for emoji, text in self.replacements.items():
                if emoji in content:
                    content = content.replace(emoji, text)
                    changes_made += content.count(text) - original_content.count(text)
            
            # Check if any changes were made
            if content == original_content:
                self.logger.info(f"No emojis found in {filepath}")
                return True
            
            # Validate content before writing
            is_valid, validation_msg = self.validate_file_content(content, filepath)
            if not is_valid:
                self.logger.error(f"Content validation failed for {filepath}: {validation_msg}")
                if backup_path:
                    self.restore_backup(filepath, backup_path)
                return False
            
            # Check if file is writable
            if not os.access(filepath, os.W_OK):
                self.logger.error(f"File is not writable: {filepath}")
                if backup_path:
                    self.restore_backup(filepath, backup_path)
                return False
            
            # Write the fixed content back
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.ok(f"Fixed {changes_made} emoji(s) in {filepath}")
            return True
            
        except UnicodeDecodeError as e:
            self.logger.error(f"Unicode decode error in {filepath}: {e}")
            if backup_path:
                self.restore_backup(filepath, backup_path)
            return False
        except UnicodeEncodeError as e:
            self.logger.error(f"Unicode encode error in {filepath}: {e}")
            if backup_path:
                self.restore_backup(filepath, backup_path)
            return False
        except PermissionError as e:
            self.logger.error(f"Permission error with {filepath}: {e}")
            if backup_path:
                self.restore_backup(filepath, backup_path)
            return False
        except Exception as e:
            self.logger.error(f"Failed to fix emojis in {filepath}: {e}")
            if backup_path:
                self.restore_backup(filepath, backup_path)
            return False
    
    def fix_emojis_in_directory(self, directory: str, file_pattern: str = "*.py") -> dict:
        """Fix emojis in all files matching the pattern in a directory"""
        try:
            self.logger.info(f"Scanning directory {directory} for files matching {file_pattern}")
            
            results = {
                'total_files': 0,
                'fixed_files': 0,
                'failed_files': 0,
                'skipped_files': 0,
                'errors': []
            }
            
            directory_path = Path(directory)
            if not directory_path.exists():
                self.logger.error(f"Directory {directory} does not exist")
                results['errors'].append(f"Directory {directory} does not exist")
                return results
            
            # Find all matching files
            files = list(directory_path.rglob(file_pattern))
            
            self.logger.info(f"Found {len(files)} files to process")
            
            for filepath in files:
                results['total_files'] += 1
                
                # Skip backup files
                if filepath.name.endswith('.backup'):
                    results['skipped_files'] += 1
                    continue
                
                if self.fix_emojis_in_file(str(filepath)):
                    results['fixed_files'] += 1
                else:
                    results['failed_files'] += 1
            
            self.logger.ok(f"Processed {results['total_files']} files")
            self.logger.info(f"Fixed: {results['fixed_files']}, Failed: {results['failed_files']}, Skipped: {results['skipped_files']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process directory {directory}: {e}")
            return {
                'total_files': 0,
                'fixed_files': 0,
                'failed_files': 0,
                'skipped_files': 0,
                'errors': [str(e)]
            }
    
    def fix_specific_test_files(self, file_list: List[str]) -> dict:
        """Fix emojis in a specific list of test files"""
        try:
            self.logger.info(f"Fixing emojis in {len(file_list)} specific files")
            
            results = {
                'total_files': len(file_list),
                'fixed_files': 0,
                'failed_files': 0,
                'not_found': 0,
                'skipped_files': 0,
                'errors': []
            }
            
            for filepath in file_list:
                if os.path.exists(filepath):
                    # Skip backup files
                    if filepath.endswith('.backup'):
                        results['skipped_files'] += 1
                        continue
                        
                    if self.fix_emojis_in_file(filepath):
                        results['fixed_files'] += 1
                    else:
                        results['failed_files'] += 1
                else:
                    self.logger.warning(f"File not found: {filepath}")
                    results['not_found'] += 1
            
            self.logger.ok(f"Processed {results['total_files']} files")
            self.logger.info(f"Fixed: {results['fixed_files']}, Failed: {results['failed_files']}, Not found: {results['not_found']}, Skipped: {results['skipped_files']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to process specific files: {e}")
            return {
                'total_files': len(file_list),
                'fixed_files': 0,
                'failed_files': 0,
                'not_found': 0,
                'skipped_files': 0,
                'errors': [str(e)]
            }
    
    def cleanup_backups(self, directory: str = "tests") -> int:
        """Clean up backup files created during processing"""
        try:
            self.logger.info(f"Cleaning up backup files in {directory}")
            
            backup_count = 0
            directory_path = Path(directory)
            
            if not directory_path.exists():
                self.logger.warning(f"Directory {directory} does not exist")
                return 0
            
            # Find all backup files
            backup_files = list(directory_path.rglob("*.backup"))
            
            for backup_file in backup_files:
                try:
                    backup_file.unlink()
                    backup_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to remove backup {backup_file}: {e}")
            
            self.logger.ok(f"Cleaned up {backup_count} backup files")
            return backup_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup backups: {e}")
            return 0

@requires_mock_backend("Fix Emojis Utility")
def main():
    """Main entry point for emoji fixing utility"""
    logger = TestLogger()
    logger.banner("Emoji Fixing Utility", 60)
    
    try:
        # Create fixer with backup enabled
        fixer = EmojiFixer(create_backups=True)
        
        # Fix emojis in specific test files
        test_files = [
            'tests/core/test_checkpoint_loading_fixes.py',
            'tests/integration/test_complete_games.py',
            'tests/runners/run_checkpoint.py'
        ]
        
        logger.info("Fixing emojis in specific test files...")
        results = fixer.fix_specific_test_files(test_files)
        
        # Fix emojis in all test files in the tests directory
        logger.info("Fixing emojis in all test files...")
        directory_results = fixer.fix_emojis_in_directory('tests', "*.py")
        
        # Summary
        logger.banner("Emoji Fixing Summary", 60)
        logger.info(f"Specific files - Fixed: {results['fixed_files']}, Failed: {results['failed_files']}")
        logger.info(f"All test files - Fixed: {directory_results['fixed_files']}, Failed: {directory_results['failed_files']}")
        
        total_fixed = results['fixed_files'] + directory_results['fixed_files']
        total_failed = results['failed_files'] + directory_results['failed_files']
        
        if total_failed == 0:
            logger.success("EMOJI FIXING COMPLETED SUCCESSFULLY!")
            
            # Optionally cleanup backups
            cleanup_backups = input("Clean up backup files? (y/N): ").lower().strip() == 'y'
            if cleanup_backups:
                fixer.cleanup_backups()
        else:
            logger.warning(f"Emoji fixing completed with {total_failed} failures")
            logger.info("Backup files have been preserved for manual review")
        
    except Exception as e:
        logger.error(f"Emoji fixing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 