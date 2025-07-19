#!/usr/bin/env python3
"""
Legacy Backend Code Cleanup Script
==================================

This script removes legacy backend logic from all test files and updates
imports to use the new backend decorators.

Legacy patterns to remove:
- requires_backend imports and usage
- Manual backend startup logic in test functions

Features:
- Pattern matching for legacy code detection
- Safe removal of legacy logic while preserving test functionality
- Import statement cleanup and standardization
- Syntax validation after changes
- Comprehensive logging of all modifications
"""

import os
import sys
import shutil
import re
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tests.utilities.test_utils import TestLogger
from tests.utilities.backend_manager import requires_mock_backend


class LegacyBackendCleaner:
    """Legacy backend code cleanup system"""
    
    def __init__(self, dry_run: bool = False):
        self.logger = TestLogger()
        self.dry_run = dry_run
        self.project_root = Path(__file__).parent.parent.parent
        self.tests_dir = self.project_root / "tests"
        
        # Legacy patterns to detect and remove
        self.legacy_patterns = {
            'check_backend_or_start_mock': {
                'pattern': r'check_backend_or_start_mock\s*\([^)]*\)',
                'description': 'check_backend_or_start_mock calls'
            },
            'requires_backend_import': {
                'pattern': r'from\s+[^.]*\.test_utils\s+import.*requires_backend',
                'description': 'requires_backend imports'
            },
            'requires_backend_usage': {
                'pattern': r'@requires_backend',
                'description': 'requires_backend decorator usage'
            },
            'check_backend_or_exit': {
                'pattern': r'check_backend_or_exit\s*\([^)]*\)',
                'description': 'check_backend_or_exit calls'
            },
            'manual_backend_startup': {
                'pattern': r'backend_tester\.is_backend_available\(\)\s*:\s*\n\s*logger\.warning.*mock.*backend',
                'description': 'Manual backend startup logic'
            }
        }
        
        # Import patterns to update
        self.import_patterns = {
            'old_test_utils_import': {
                'pattern': r'from\s+tests\.utilities\.test_utils\s+import.*requires_backend',
                'replacement': 'from tests.utilities.backend_manager import requires_mock_backend'
            },
            'add_backend_manager_import': {
                'pattern': r'(from\s+tests\.utilities\.test_utils\s+import.*?)(\n)',
                'replacement': r'\1\nfrom tests.utilities.backend_manager import requires_mock_backend\2'
            }
        }
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'files_with_legacy_code': 0,
            'legacy_patterns_removed': 0,
            'imports_updated': 0,
            'skipped_files': 0,
            'error_files': 0,
            'backups_created': 0
        }
        
        # Changes made
        self.changes = []
    
    def detect_legacy_patterns(self, content: str) -> Dict[str, List[Tuple[int, str]]]:
        """
        Detect legacy patterns in file content
        
        Args:
            content: File content
            
        Returns:
            Dictionary mapping pattern names to lists of (line_number, matched_text)
        """
        lines = content.split('\n')
        detected_patterns = {}
        
        for pattern_name, pattern_info in self.legacy_patterns.items():
            pattern = pattern_info['pattern']
            matches = []
            
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line, re.MULTILINE | re.DOTALL):
                    matches.append((i, line.strip()))
            
            if matches:
                detected_patterns[pattern_name] = matches
        
        return detected_patterns
    
    def remove_legacy_patterns(self, content: str, detected_patterns: Dict[str, List[Tuple[int, str]]]) -> str:
        """
        Remove legacy patterns from file content
        
        Args:
            content: File content
            detected_patterns: Detected patterns to remove
            
        Returns:
            Updated file content
        """
        lines = content.split('\n')
        
        # Sort patterns by line number in reverse order to maintain line numbers
        all_matches = []
        for pattern_name, matches in detected_patterns.items():
            for line_num, matched_text in matches:
                all_matches.append((line_num, pattern_name, matched_text))
        
        all_matches.sort(key=lambda x: x[0], reverse=True)
        
        # Remove patterns from lines
        for line_num, pattern_name, matched_text in all_matches:
            if line_num <= len(lines):
                # Check if the line contains only the pattern (with whitespace)
                line = lines[line_num - 1]
                if re.match(r'^\s*' + re.escape(matched_text) + r'\s*$', line):
                    # Remove the entire line
                    lines.pop(line_num - 1)
                else:
                    # Remove just the pattern from the line
                    pattern = self.legacy_patterns[pattern_name]['pattern']
                    lines[line_num - 1] = re.sub(pattern, '', line)
        
        return '\n'.join(lines)
    
    def update_imports(self, content: str) -> str:
        """
        Update import statements to use new backend decorators
        
        Args:
            content: File content
            
        Returns:
            Updated file content
        """
        # Update old test_utils imports
        for pattern_name, pattern_info in self.import_patterns.items():
            pattern = pattern_info['pattern']
            replacement = pattern_info['replacement']
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        # Add backend_manager import if not present
        if 'from tests.utilities.backend_manager import' not in content:
            # Find the best place to add the import
            lines = content.split('\n')
            import_section_end = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_section_end = i + 1
                elif line.strip() and not line.strip().startswith('#'):
                    break
            
            # Add the import
            lines.insert(import_section_end, "from tests.utilities.backend_manager import requires_mock_backend")
            content = '\n'.join(lines)
        
        return content
    
    def validate_syntax(self, content: str, file_path: Path) -> bool:
        """
        Validate that the file has correct Python syntax after changes
        
        Args:
            content: File content
            file_path: Path to the file
            
        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            ast.parse(content)
            return True
        except SyntaxError as e:
            self.logger.error(f"Syntax error in {file_path.name} after cleanup: {e}")
            return False
    
    def create_backup(self, file_path: Path) -> Optional[Path]:
        """
        Create a backup of the file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Path to backup file, or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_suffix(f'.legacy_backup_{timestamp}{file_path.suffix}')
            shutil.copy2(file_path, backup_path)
            self.stats['backups_created'] += 1
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to create backup for {file_path}: {e}")
            return None
    
    def process_file(self, file_path: Path) -> bool:
        """
        Process a single file to remove legacy backend code
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if successful, False otherwise
        """
        self.stats['total_files'] += 1
        
        # Skip non-Python files
        if file_path.suffix != '.py':
            self.stats['skipped_files'] += 1
            return True
        
        self.logger.info(f"Processing {file_path.relative_to(self.project_root)}")
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.logger.error(f"Could not read {file_path}: {e}")
            self.stats['error_files'] += 1
            return False
        
        # Detect legacy patterns
        detected_patterns = self.detect_legacy_patterns(content)
        
        if not detected_patterns:
            self.logger.info(f"No legacy patterns found in {file_path.name}")
            self.stats['skipped_files'] += 1
            return True
        
        self.logger.info(f"Found legacy patterns in {file_path.name}:")
        for pattern_name, matches in detected_patterns.items():
            pattern_desc = self.legacy_patterns[pattern_name]['description']
            self.logger.info(f"  - {pattern_desc}: {len(matches)} instances")
        
        # Create backup if not dry run
        backup_path = None
        if not self.dry_run:
            backup_path = self.create_backup(file_path)
            if backup_path is None:
                self.logger.error(f"Failed to create backup for {file_path}")
                self.stats['error_files'] += 1
                return False
        
        # Apply changes
        try:
            # Remove legacy patterns
            content = self.remove_legacy_patterns(content, detected_patterns)
            
            # Update imports
            content = self.update_imports(content)
            
            # Validate syntax
            if not self.validate_syntax(content, file_path):
                self.stats['error_files'] += 1
                return False
            
            # Write changes if not dry run
            if not self.dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.logger.ok(f"Updated {file_path.name}")
            else:
                self.logger.info(f"[DRY RUN] Would update {file_path.name}")
            
            # Update statistics
            self.stats['files_with_legacy_code'] += 1
            total_patterns = sum(len(matches) for matches in detected_patterns.values())
            self.stats['legacy_patterns_removed'] += total_patterns
            
            # Record changes
            self.changes.append({
                'file': str(file_path.relative_to(self.project_root)),
                'patterns_removed': total_patterns,
                'pattern_details': {name: len(matches) for name, matches in detected_patterns.items()},
                'backup': str(backup_path) if backup_path else None
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update {file_path}: {e}")
            self.stats['error_files'] += 1
            return False
    
    def scan_and_cleanup_files(self) -> bool:
        """
        Scan all test files and remove legacy backend code
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.banner("Legacy Backend Code Cleanup", 60)
        
        if self.dry_run:
            self.logger.warning("DRY RUN MODE - No files will be modified")
        
        # Find all Python files in tests directory
        test_files = list(self.tests_dir.rglob("*.py"))
        
        self.logger.info(f"Found {len(test_files)} Python files in tests directory")
        
        # Process each file
        success_count = 0
        for file_path in test_files:
            if self.process_file(file_path):
                success_count += 1
        
        # Print summary
        self.logger.section("Cleanup Summary")
        self.logger.info(f"Total files processed: {self.stats['total_files']}")
        self.logger.info(f"Files with legacy code: {self.stats['files_with_legacy_code']}")
        self.logger.info(f"Legacy patterns removed: {self.stats['legacy_patterns_removed']}")
        self.logger.info(f"Imports updated: {self.stats['imports_updated']}")
        self.logger.info(f"Skipped files: {self.stats['skipped_files']}")
        self.logger.info(f"Error files: {self.stats['error_files']}")
        self.logger.info(f"Backups created: {self.stats['backups_created']}")
        
        if self.changes:
            self.logger.section("Changes Made")
            for change in self.changes:
                self.logger.info(f"  {change['file']}")
                self.logger.info(f"    - Removed {change['patterns_removed']} legacy patterns")
                for pattern_name, count in change['pattern_details'].items():
                    pattern_desc = self.legacy_patterns[pattern_name]['description']
                    self.logger.info(f"      - {pattern_desc}: {count}")
                if change['backup']:
                    self.logger.info(f"    - Backup: {change['backup']}")
        
        if self.stats['error_files'] > 0:
            self.logger.error(f"Failed to process {self.stats['error_files']} files")
            return False
        
        self.logger.success("Legacy backend code cleanup completed successfully!")
        return True


@requires_mock_backend("Legacy Backend Cleanup")
def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up legacy backend code from test files")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be changed without making changes")
    
    args = parser.parse_args()
    
    cleaner = LegacyBackendCleaner(dry_run=args.dry_run)
    
    success = cleaner.scan_and_cleanup_files()
    
    if success:
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("1. Run the compliance checker to verify cleanup:")
        print("   python tests/compliance_checker.py")
        print("2. Validate the changes:")
        print("   python tests/utilities/validate_backend_decorators.py")
        print("3. Run a subset of tests to ensure everything works:")
        print("   python tests/runners/master_test_runner.py --level core")
        print("="*60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 