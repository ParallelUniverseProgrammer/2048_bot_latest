#!/usr/bin/env python3
"""
Test Suite Compliance Checker
============================

This script analyzes all test files in the tests directory to ensure they adhere
to the standards outlined in tests/README.md.

Compliance Requirements:
1. Use TestLogger for all output with standardized prefixes
2. Follow the test structure template
3. Include proper docstrings and documentation
4. Handle errors gracefully
5. Use shared utilities from test_utils
6. Have proper main() function with __name__ == "__main__" guard
7. Use descriptive test names
8. Include comprehensive error handling
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# No external dependencies needed for this utility script

class ComplianceLevel(Enum):
    COMPLIANT = "COMPLIANT"
    MINOR_ISSUES = "MINOR_ISSUES"
    MAJOR_ISSUES = "MAJOR_ISSUES"
    NON_COMPLIANT = "NON_COMPLIANT"

@dataclass
class ComplianceIssue:
    line_number: int
    issue_type: str
    description: str
    severity: str

@dataclass
class FileCompliance:
    file_path: str
    compliance_level: ComplianceLevel
    issues: List[ComplianceIssue]
    has_test_logger: bool
    has_proper_docstring: bool
    has_main_function: bool
    uses_shared_utilities: bool
    has_error_handling: bool

class ComplianceChecker:
    """Check test files for compliance with README standards"""
    
    def __init__(self):
        self.tests_dir = Path(__file__).parent
        self.required_prefixes = {
            'OK:', 'ERROR:', 'WARNING:', 'INFO:', 'GAME:', 'STARTING:', 
            'RUNNING:', 'TESTING:', 'CONTROLS:', 'PLAYBACK:', 'STEP:', 
            'PROGRESS:', 'SUCCESS:'
        }
        self.shared_utilities = [
            'TestLogger', 'BackendTester', 'GameTester', 'PlaybackTester',
            'requires_real_backend', 'requires_mock_backend', 'with_backend_fallback'
        ]
        
    def check_file(self, file_path: Path) -> FileCompliance:
        """Check a single file for compliance"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            return FileCompliance(
                file_path=str(file_path),
                compliance_level=ComplianceLevel.NON_COMPLIANT,
                issues=[ComplianceIssue(0, "FILE_READ_ERROR", f"Cannot read file: {e}", "CRITICAL")],
                has_test_logger=False,
                has_proper_docstring=False,
                has_main_function=False,
                uses_shared_utilities=False,
                has_error_handling=False
            )
        
        # Check for shebang
        if not content.startswith('#!/usr/bin/env python3'):
            issues.append(ComplianceIssue(1, "MISSING_SHEBANG", "Missing shebang line", "MINOR"))
        
        # Check for proper docstring
        has_proper_docstring = self._check_docstring(content)
        if not has_proper_docstring:
            issues.append(ComplianceIssue(1, "MISSING_DOCSTRING", "Missing or inadequate docstring", "MAJOR"))
        
        # Check for TestLogger usage
        has_test_logger = 'TestLogger' in content
        if not has_test_logger:
            issues.append(ComplianceIssue(1, "NO_TEST_LOGGER", "Not using TestLogger for output", "MAJOR"))
        
        # Check for proper logging prefixes
        self._check_logging_prefixes(content, lines, issues)
        
        # Check for shared utilities usage
        uses_shared_utilities = any(util in content for util in self.shared_utilities)
        if not uses_shared_utilities and 'test_utils' not in content:
            issues.append(ComplianceIssue(1, "NO_SHARED_UTILITIES", "Not using shared utilities from test_utils", "MAJOR"))
        
        # Check for main function
        has_main_function = 'def main():' in content and '__name__ == "__main__"' in content
        if not has_main_function:
            issues.append(ComplianceIssue(1, "NO_MAIN_FUNCTION", "Missing main() function or __name__ guard", "MAJOR"))
        
        # Check for error handling
        has_error_handling = self._check_error_handling(content)
        if not has_error_handling:
            issues.append(ComplianceIssue(1, "NO_ERROR_HANDLING", "No try/except blocks for error handling", "MAJOR"))
        
        # Check for print statements (should use TestLogger instead)
        self._check_print_statements(content, lines, issues)
        
        # Check for proper imports
        self._check_imports(content, lines, issues)
        
        # Check for backend decorator compliance
        self._check_backend_decorators(content, lines, issues)
        
        # Determine compliance level
        compliance_level = self._determine_compliance_level(issues)
        
        return FileCompliance(
            file_path=str(file_path),
            compliance_level=compliance_level,
            issues=issues,
            has_test_logger=has_test_logger,
            has_proper_docstring=has_proper_docstring,
            has_main_function=has_main_function,
            uses_shared_utilities=uses_shared_utilities,
            has_error_handling=has_error_handling
        )
    
    def _check_docstring(self, content: str) -> bool:
        """Check if file has proper docstring"""
        # Look for docstring after shebang
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('"""') or line.strip().startswith("'''"):
                # Found docstring, check if it's substantial
                docstring_lines = []
                j = i
                while j < len(lines) and ('"""' not in lines[j] or lines[j].count('"""') < 2):
                    docstring_lines.append(lines[j])
                    j += 1
                if j < len(lines):
                    docstring_lines.append(lines[j])
                
                docstring = '\n'.join(docstring_lines)
                # Check if docstring has substantial content
                return len(docstring) > 50 and ('test' in docstring.lower() or 'check' in docstring.lower())
        return False
    
    def _check_logging_prefixes(self, content: str, lines: List[str], issues: List[ComplianceIssue]):
        """Check for proper logging prefixes"""
        # Look for print statements that should use TestLogger
        for i, line in enumerate(lines, 1):
            if 'print(' in line and any(prefix in line for prefix in self.required_prefixes):
                issues.append(ComplianceIssue(i, "PRINT_WITH_PREFIX", 
                    f"Using print() with prefix '{line.strip()}' - should use TestLogger", "MINOR"))
    
    def _check_error_handling(self, content: str) -> bool:
        """Check for error handling patterns"""
        return ('try:' in content and 'except' in content) or 'TestLogger' in content
    
    def _check_print_statements(self, content: str, lines: List[str], issues: List[ComplianceIssue]):
        """Check for print statements that should use TestLogger"""
        for i, line in enumerate(lines, 1):
            if 'print(' in line and not line.strip().startswith('#'):
                # Allow print statements in specific contexts
                if not any(context in line for context in ['# pragma:', 'debug', 'DEBUG']):
                    issues.append(ComplianceIssue(i, "PRINT_STATEMENT", 
                        f"Using print() instead of TestLogger: {line.strip()}", "MINOR"))
    
    def _check_imports(self, content: str, lines: List[str], issues: List[ComplianceIssue]):
        """Check for proper imports"""
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                # Check for direct imports that should use test_utils
                if any(direct_import in line for direct_import in ['requests', 'json', 'time']):
                    if 'test_utils' not in content:
                        issues.append(ComplianceIssue(i, "DIRECT_IMPORT", 
                            f"Direct import that could use test_utils: {line.strip()}", "MINOR"))
    
    def _check_backend_decorators(self, content: str, lines: List[str], issues: List[ComplianceIssue]):
        """Check for backend decorator compliance"""
        # Check if this is a test file (contains test functions)
        has_test_functions = any('def test_' in line for line in lines)
        has_main_function = 'def main():' in content
        
        if not (has_test_functions or has_main_function):
            return  # Not a test file
        
        # Check for legacy backend logic
        legacy_patterns = [
            'check_backend_or_start_mock',
            'requires_backend',
            'check_backend_or_exit'
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in legacy_patterns:
                if pattern in line and not line.strip().startswith('#'):
                    issues.append(ComplianceIssue(i, "LEGACY_BACKEND_LOGIC", 
                        f"Legacy backend logic detected: {pattern}. Use @requires_real_backend or @requires_mock_backend instead.", "MAJOR"))
        
        # Check for missing backend decorators on test functions
        test_functions = []
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('def test_'):
                test_functions.append((i, line.strip()))
        
        # Check for missing backend decorators on main function
        main_functions = []
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('def main('):
                main_functions.append((i, line.strip()))
        
        # Check if any test functions or main functions are missing backend decorators
        all_functions = test_functions + main_functions
        
        for line_num, func_line in all_functions:
            # Look for decorators in the lines before this function
            has_backend_decorator = False
            for j in range(max(1, line_num - 5), line_num):
                if j < len(lines):
                    line_content = lines[j - 1].strip()
                    if ('@requires_real_backend' in line_content or 
                        '@requires_mock_backend' in line_content):
                        has_backend_decorator = True
                        break
            
            if not has_backend_decorator:
                func_name = func_line.split('(')[0].replace('def ', '')
                issues.append(ComplianceIssue(line_num, "MISSING_BACKEND_DECORATOR", 
                    f"Function '{func_name}' missing backend decorator. Use @requires_real_backend or @requires_mock_backend.", "MAJOR"))
    
    def _determine_compliance_level(self, issues: List[ComplianceIssue]) -> ComplianceLevel:
        """Determine compliance level based on issues"""
        critical_count = sum(1 for issue in issues if issue.severity == "CRITICAL")
        major_count = sum(1 for issue in issues if issue.severity == "MAJOR")
        minor_count = sum(1 for issue in issues if issue.severity == "MINOR")
        
        if critical_count > 0:
            return ComplianceLevel.NON_COMPLIANT
        elif major_count > 0:
            return ComplianceLevel.MAJOR_ISSUES
        elif minor_count > 0:
            return ComplianceLevel.MINOR_ISSUES
        else:
            return ComplianceLevel.COMPLIANT
    
    def check_all_files(self) -> Dict[str, FileCompliance]:
        """Check all test files for compliance"""
        results = {}
        
        for file_path in self.tests_dir.rglob('*.py'):
            if file_path.name != '__init__.py' and file_path.name != 'compliance_checker.py':
                results[str(file_path)] = self.check_file(file_path)
        
        return results
    
    def print_report(self, results: Dict[str, FileCompliance]):
        """Print compliance report"""
        print("=" * 80)
        print("TEST SUITE COMPLIANCE REPORT")
        print("=" * 80)
        
        # Group by compliance level
        compliant = []
        minor_issues = []
        major_issues = []
        non_compliant = []
        
        for file_path, compliance in results.items():
            if compliance.compliance_level == ComplianceLevel.COMPLIANT:
                compliant.append(compliance)
            elif compliance.compliance_level == ComplianceLevel.MINOR_ISSUES:
                minor_issues.append(compliance)
            elif compliance.compliance_level == ComplianceLevel.MAJOR_ISSUES:
                major_issues.append(compliance)
            else:
                non_compliant.append(compliance)
        
        # Print summary
        print(f"\nSUMMARY:")
        print(f"  Total files: {len(results)}")
        print(f"  Compliant: {len(compliant)}")
        print(f"  Minor issues: {len(minor_issues)}")
        print(f"  Major issues: {len(major_issues)}")
        print(f"  Non-compliant: {len(non_compliant)}")
        
        # Print details for each category
        if non_compliant:
            print(f"\n❌ NON-COMPLIANT FILES ({len(non_compliant)}):")
            for compliance in non_compliant:
                print(f"  {compliance.file_path}")
                for issue in compliance.issues:
                    print(f"    Line {issue.line_number}: {issue.issue_type} - {issue.description}")
        
        if major_issues:
            print(f"\n⚠️  MAJOR ISSUES ({len(major_issues)}):")
            for compliance in major_issues:
                print(f"  {compliance.file_path}")
                for issue in compliance.issues:
                    if issue.severity == "MAJOR":
                        print(f"    Line {issue.line_number}: {issue.issue_type} - {issue.description}")
        
        if minor_issues:
            print(f"\n🔧 MINOR ISSUES ({len(minor_issues)}):")
            for compliance in minor_issues:
                print(f"  {compliance.file_path}")
                for issue in compliance.issues:
                    if issue.severity == "MINOR":
                        print(f"    Line {issue.line_number}: {issue.issue_type} - {issue.description}")
        
        if compliant:
            print(f"\n✅ COMPLIANT FILES ({len(compliant)}):")
            for compliance in compliant:
                print(f"  {compliance.file_path}")
        
        # Print recommendations
        print(f"\nRECOMMENDATIONS:")
        if non_compliant or major_issues:
            print("  1. Fix all critical and major issues first")
            print("  2. Ensure all files use TestLogger for output")
            print("  3. Add proper docstrings to all test files")
            print("  4. Include main() functions with __name__ guards")
            print("  5. Use @requires_real_backend or @requires_mock_backend decorators for all test functions")
            print("  6. Remove legacy backend logic (check_backend_or_start_mock, requires_backend)")
        if minor_issues:
            print("  7. Replace print() statements with TestLogger calls")
            print("  8. Use shared utilities from test_utils where possible")
        
        print("\n" + "=" * 80)

def main():
    """Main entry point"""
    print("=" * 60)
    print("Test Suite Compliance Checker")
    print("=" * 60)
    
    try:
        checker = ComplianceChecker()
        results = checker.check_all_files()
        checker.print_report(results)
        
        # Return appropriate exit code
        non_compliant_count = sum(1 for compliance in results.values() 
                                 if compliance.compliance_level == ComplianceLevel.NON_COMPLIANT)
        major_issues_count = sum(1 for compliance in results.values() 
                                if compliance.compliance_level == ComplianceLevel.MAJOR_ISSUES)
        
        if non_compliant_count > 0:
            print(f"Found {non_compliant_count} non-compliant files")
            sys.exit(1)  # Critical issues
        elif major_issues_count > 0:
            print(f"Found {major_issues_count} files with major issues")
            sys.exit(2)  # Major issues
        else:
            print("All files are compliant!")
            sys.exit(0)  # All good
            
    except Exception as e:
        print(f"Compliance check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 