#!/usr/bin/env python3
"""
Automated Backend Decorator Application Script
==============================================

This script automatically applies @requires_real_backend or @requires_mock_backend
decorators to all test files based on their directory location and content analysis.

The script categorizes files as follows:
- Real backend: tests/integration/, tests/core/, tests/performance/ (except speed_control.py)
- Mock backend: tests/training/, tests/mobile/, tests/frontend/, tests/playback/, tests/runners/

Features:
- Safe file modification with backup creation
- Content analysis to detect test functions and main functions
- Progress reporting and error handling
- Dry-run mode for testing
- Comprehensive logging of all changes
"""

import os
import sys
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tests.utilities.test_utils import TestLogger
from tests.utilities.backend_manager import requires_mock_backend


class BackendDecoratorApplier:
    """Automated backend decorator application system"""
    
    def __init__(self, dry_run: bool = False):
        self.logger = TestLogger()
        self.dry_run = dry_run
        self.project_root = Path(__file__).parent.parent.parent
        self.tests_dir = self.project_root / "tests"
        
        # File categorization rules
        self.real_backend_dirs = [
            "tests/integration",
            "tests/core", 
            "tests/performance"
        ]
        
        self.mock_backend_dirs = [
            "tests/training",
            "tests/mobile",
            "tests/frontend",
            "tests/playback",
            "tests/runners"
        ]
        
        # Files that should use mock backend despite being in real backend dirs
        self.mock_backend_exceptions = [
            "tests/performance/test_speed_control.py"
        ]
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'real_backend_files': 0,
            'mock_backend_files': 0,
            'skipped_files': 0,
            'error_files': 0,
            'backups_created': 0
        }
        
        # Changes made
        self.changes = []
    
    def categorize_file(self, file_path: Path) -> Optional[str]:
        """
        Categorize a file as needing real backend or mock backend
        
        Args:
            file_path: Path to the file
            
        Returns:
            'real', 'mock', or None if should be skipped
        """
        # Convert to relative path from project root
        rel_path = file_path.relative_to(self.project_root)
        rel_path_str = str(rel_path).replace('\\', '/')
        
        # Check exceptions first
        if rel_path_str in self.mock_backend_exceptions:
            return 'mock'
        
        # Check real backend directories
        for real_dir in self.real_backend_dirs:
            if rel_path_str.startswith(real_dir):
                return 'real'
        
        # Check mock backend directories
        for mock_dir in self.mock_backend_dirs:
            if rel_path_str.startswith(mock_dir):
                return 'mock'
        
        # Skip files not in test directories
        return None
    
    def analyze_file_content(self, file_path: Path) -> Dict[str, List[Tuple[int, str]]]:
        """
        Analyze file content to find functions that need decorators
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with 'test_functions' and 'main_functions' lists
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            self.logger.error(f"Could not read file {file_path}: {e}")
            return {'test_functions': [], 'main_functions': []}
        
        functions = {
            'test_functions': [],
            'main_functions': []
        }
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Find test functions
            if stripped.startswith('def test_'):
                func_name = stripped.split('(')[0].replace('def ', '')
                functions['test_functions'].append((i, func_name))
            
            # Find main functions
            elif stripped.startswith('def main('):
                functions['main_functions'].append((i, 'main'))
        
        return functions
    
    def has_backend_decorator(self, file_path: Path, line_number: int) -> bool:
        """
        Check if a function already has a backend decorator
        
        Args:
            file_path: Path to the file
            line_number: Line number of the function
            
        Returns:
            True if function already has a backend decorator
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception:
            return False
        
        # Check lines before the function (up to 5 lines back)
        start_line = max(0, line_number - 6)
        end_line = line_number - 1
        
        for i in range(start_line, end_line):
            if i < len(lines):
                line = lines[i].strip()
                if ('@requires_real_backend' in line or 
                    '@requires_mock_backend' in line):
                    return True
        
        return False
    
    def has_backend_import(self, file_path: Path) -> bool:
        """
        Check if file already has backend decorator imports
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file has backend decorator imports
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return False
        
        return ('requires_real_backend' in content or 
                'requires_mock_backend' in content)
    
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
            backup_path = file_path.with_suffix(f'.backup_{timestamp}{file_path.suffix}')
            shutil.copy2(file_path, backup_path)
            self.stats['backups_created'] += 1
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to create backup for {file_path}: {e}")
            return None
    
    def add_backend_import(self, content: str, backend_type: str) -> str:
        """
        Add backend decorator import to file content
        
        Args:
            content: File content
            backend_type: 'real' or 'mock'
            
        Returns:
            Updated file content
        """
        lines = content.split('\n')
        
        # Find the best place to add the import
        import_section_end = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_section_end = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                break
        
        # Create the import statement
        if backend_type == 'real':
            import_stmt = "from tests.utilities.backend_manager import requires_real_backend"
        else:
            import_stmt = "from tests.utilities.backend_manager import requires_mock_backend"
        
        # Add the import
        lines.insert(import_section_end, import_stmt)
        
        return '\n'.join(lines)
    
    def add_decorator_to_function(self, content: str, line_number: int, backend_type: str) -> str:
        """
        Add backend decorator to a specific function
        
        Args:
            content: File content
            line_number: Line number of the function
            backend_type: 'real' or 'mock'
            
        Returns:
            Updated file content
        """
        lines = content.split('\n')
        
        # Create the decorator
        if backend_type == 'real':
            decorator = "@requires_real_backend"
        else:
            decorator = "@requires_mock_backend"
        
        # Insert the decorator before the function
        lines.insert(line_number - 1, decorator)
        
        return '\n'.join(lines)
    
    def process_file(self, file_path: Path) -> bool:
        """
        Process a single file to add backend decorators
        
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
        
        # Categorize the file
        backend_type = self.categorize_file(file_path)
        if backend_type is None:
            self.stats['skipped_files'] += 1
            return True
        
        self.logger.info(f"Processing {file_path.relative_to(self.project_root)} ({backend_type} backend)")
        
        # Analyze file content
        functions = self.analyze_file_content(file_path)
        all_functions = functions['test_functions'] + functions['main_functions']
        
        if not all_functions:
            self.logger.info(f"No test or main functions found in {file_path.name}")
            self.stats['skipped_files'] += 1
            return True
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.logger.error(f"Could not read {file_path}: {e}")
            self.stats['error_files'] += 1
            return False
        
        # Check if file needs backend import
        needs_import = not self.has_backend_import(file_path)
        
        # Find functions that need decorators
        functions_needing_decorators = []
        for line_num, func_name in all_functions:
            if not self.has_backend_decorator(file_path, line_num):
                functions_needing_decorators.append((line_num, func_name))
        
        if not functions_needing_decorators and not needs_import:
            self.logger.info(f"No changes needed for {file_path.name}")
            self.stats['skipped_files'] += 1
            return True
        
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
            # Add import if needed
            if needs_import:
                content = self.add_backend_import(content, backend_type)
                self.logger.ok(f"Added {backend_type} backend import to {file_path.name}")
            
            # Add decorators to functions
            # Sort by line number in reverse order to maintain line numbers
            functions_needing_decorators.sort(key=lambda x: x[0], reverse=True)
            
            for line_num, func_name in functions_needing_decorators:
                content = self.add_decorator_to_function(content, line_num, backend_type)
                self.logger.ok(f"Added {backend_type} backend decorator to {func_name} in {file_path.name}")
            
            # Write changes if not dry run
            if not self.dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.logger.ok(f"Updated {file_path.name}")
            else:
                self.logger.info(f"[DRY RUN] Would update {file_path.name}")
            
            # Update statistics
            if backend_type == 'real':
                self.stats['real_backend_files'] += 1
            else:
                self.stats['mock_backend_files'] += 1
            
            # Record changes
            self.changes.append({
                'file': str(file_path.relative_to(self.project_root)),
                'backend_type': backend_type,
                'functions_updated': len(functions_needing_decorators),
                'import_added': needs_import,
                'backup': str(backup_path) if backup_path else None
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update {file_path}: {e}")
            self.stats['error_files'] += 1
            return False
    
    def scan_and_process_files(self) -> bool:
        """
        Scan all test files and apply decorators
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.banner("Automated Backend Decorator Application", 60)
        
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
        self.logger.section("Processing Summary")
        self.logger.info(f"Total files processed: {self.stats['total_files']}")
        self.logger.info(f"Real backend files: {self.stats['real_backend_files']}")
        self.logger.info(f"Mock backend files: {self.stats['mock_backend_files']}")
        self.logger.info(f"Skipped files: {self.stats['skipped_files']}")
        self.logger.info(f"Error files: {self.stats['error_files']}")
        self.logger.info(f"Backups created: {self.stats['backups_created']}")
        
        if self.changes:
            self.logger.section("Changes Made")
            for change in self.changes:
                self.logger.info(f"  {change['file']} ({change['backend_type']} backend)")
                if change['functions_updated'] > 0:
                    self.logger.info(f"    - Updated {change['functions_updated']} functions")
                if change['import_added']:
                    self.logger.info(f"    - Added backend import")
                if change['backup']:
                    self.logger.info(f"    - Backup: {change['backup']}")
        
        if self.stats['error_files'] > 0:
            self.logger.error(f"Failed to process {self.stats['error_files']} files")
            return False
        
        self.logger.success("Backend decorator application completed successfully!")
        return True


@requires_mock_backend("Backend Decorator Application")
def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply backend decorators to test files")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be changed without making changes")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    applier = BackendDecoratorApplier(dry_run=args.dry_run)
    
    success = applier.scan_and_process_files()
    
    if success:
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("1. Run the compliance checker to verify changes:")
        print("   python tests/compliance_checker.py")
        print("2. If issues remain, run the legacy cleanup script:")
        print("   python tests/utilities/cleanup_legacy_backend.py")
        print("3. Validate the changes:")
        print("   python tests/utilities/validate_backend_decorators.py")
        print("="*60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 