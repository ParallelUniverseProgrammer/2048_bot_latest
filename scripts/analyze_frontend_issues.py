#!/usr/bin/env python3
"""
Frontend code analysis tool to identify potential freezing issues.
This tool analyzes the React frontend code for performance problems and infinite loops.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

class FrontendAnalyzer:
    """Analyzes frontend code for potential freezing issues"""
    
    def __init__(self):
        self.frontend_dir = Path("frontend/src")
        self.issues = []
        self.warnings = []
        
    def _safe_read_file(self, file_path: Path) -> str:
        """Safely read file with proper encoding handling"""
        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                return file_path.read_text(encoding='latin-1')
            except:
                print(f"Warning: Could not read {file_path}")
                return ""
        
    def analyze_all(self) -> Dict[str, Any]:
        """Run all frontend analyses"""
        print("🔍 Analyzing Frontend Code for Freezing Issues")
        print("=" * 60)
        
        results = {
            'useEffect_issues': self.analyze_useEffect_loops(),
            'state_update_issues': self.analyze_state_updates(),
            'websocket_issues': self.analyze_websocket_handling(),
            'performance_issues': self.analyze_performance_problems(),
            'memory_issues': self.analyze_memory_usage(),
            'infinite_loop_risks': self.analyze_infinite_loop_risks(),
            'summary': self.generate_summary()
        }
        
        return results
        
    def analyze_useEffect_loops(self) -> List[Dict[str, Any]]:
        """Analyze useEffect hooks for potential infinite loops"""
        print("\n🔄 Analyzing useEffect hooks for infinite loops...")
        
        issues = []
        
        for file_path in self.frontend_dir.rglob("*.tsx"):
            content = self._safe_read_file(file_path)
            if not content:
                continue
            
            # Find useEffect patterns
            useEffect_pattern = r'useEffect\s*\(\s*\(\s*\)\s*=>\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}\s*,\s*\[([^\]]*)\]\s*\)'
            
            for match in re.finditer(useEffect_pattern, content, re.DOTALL):
                effect_body = match.group(1)
                dependencies = match.group(2)
                
                # Check for missing dependencies
                if self._has_missing_dependencies(effect_body, dependencies):
                    issues.append({
                        'file': str(file_path),
                        'type': 'missing_dependencies',
                        'severity': 'high',
                        'description': 'useEffect may have missing dependencies causing infinite re-renders',
                        'code_snippet': match.group(0)[:200] + "..."
                    })
                    
                # Check for state updates in useEffect without proper conditions
                if self._has_unconditional_state_updates(effect_body):
                    issues.append({
                        'file': str(file_path),
                        'type': 'unconditional_state_update',
                        'severity': 'critical',
                        'description': 'useEffect contains unconditional state updates that could cause infinite loops',
                        'code_snippet': match.group(0)[:200] + "..."
                    })
                    
        return issues
        
    def analyze_state_updates(self) -> List[Dict[str, Any]]:
        """Analyze state update patterns for performance issues"""
        print("\n🔄 Analyzing state update patterns...")
        
        issues = []
        
        for file_path in self.frontend_dir.rglob("*.tsx"):
            content = self._safe_read_file(file_path)
            if not content:
                continue
            
            # Check for rapid state updates
            if self._has_rapid_state_updates(content):
                issues.append({
                    'file': str(file_path),
                    'type': 'rapid_state_updates',
                    'severity': 'medium',
                    'description': 'File contains patterns that could cause rapid state updates',
                    'recommendation': 'Consider using useMemo, useCallback, or batching state updates'
                })
                
            # Check for state updates in render
            if self._has_state_updates_in_render(content):
                issues.append({
                    'file': str(file_path),
                    'type': 'state_update_in_render',
                    'severity': 'critical',
                    'description': 'State updates in render function can cause infinite re-renders',
                    'recommendation': 'Move state updates to useEffect or event handlers'
                })
                
        return issues
        
    def analyze_websocket_handling(self) -> List[Dict[str, Any]]:
        """Analyze WebSocket message handling for performance issues"""
        print("\n🌐 Analyzing WebSocket message handling...")
        
        issues = []
        
        websocket_file = self.frontend_dir / "utils" / "websocket.ts"
        if websocket_file.exists():
            content = self._safe_read_file(websocket_file)
            if content:
                # Check for potential blocking operations in message handler
                if self._has_blocking_operations_in_message_handler(content):
                    issues.append({
                        'file': str(websocket_file),
                        'type': 'blocking_message_handler',
                        'severity': 'high',
                        'description': 'WebSocket message handler may contain blocking operations',
                        'recommendation': 'Use async/await or setTimeout for heavy operations'
                    })
                    
                # Check for excessive message processing
                if self._has_excessive_message_processing(content):
                    issues.append({
                        'file': str(websocket_file),
                        'type': 'excessive_message_processing',
                        'severity': 'medium',
                        'description': 'WebSocket message handler processes many message types',
                        'recommendation': 'Consider message throttling or debouncing'
                    })
                
        return issues
        
    def analyze_performance_problems(self) -> List[Dict[str, Any]]:
        """Analyze for general performance problems"""
        print("\n⚡ Analyzing performance problems...")
        
        issues = []
        
        for file_path in self.frontend_dir.rglob("*.tsx"):
            content = self._safe_read_file(file_path)
            if not content:
                continue
            
            # Check for missing React.memo
            if self._should_use_react_memo(content):
                issues.append({
                    'file': str(file_path),
                    'type': 'missing_react_memo',
                    'severity': 'medium',
                    'description': 'Component could benefit from React.memo to prevent unnecessary re-renders',
                    'recommendation': 'Wrap component with React.memo'
                })
                
            # Check for expensive operations in render
            if self._has_expensive_operations_in_render(content):
                issues.append({
                    'file': str(file_path),
                    'type': 'expensive_render_operations',
                    'severity': 'high',
                    'description': 'Expensive operations in render function',
                    'recommendation': 'Move to useMemo or useCallback'
                })
                
            # Check for large inline objects
            if self._has_large_inline_objects(content):
                issues.append({
                    'file': str(file_path),
                    'type': 'large_inline_objects',
                    'severity': 'medium',
                    'description': 'Large inline objects created in render',
                    'recommendation': 'Move to useMemo or constants'
                })
                
        return issues
        
    def analyze_memory_usage(self) -> List[Dict[str, Any]]:
        """Analyze for potential memory leaks"""
        print("\n💾 Analyzing memory usage patterns...")
        
        issues = []
        
        for file_path in self.frontend_dir.rglob("*.tsx"):
            content = self._safe_read_file(file_path)
            if not content:
                continue
            
            # Check for missing cleanup in useEffect
            if self._has_missing_cleanup(content):
                issues.append({
                    'file': str(file_path),
                    'type': 'missing_cleanup',
                    'severity': 'high',
                    'description': 'useEffect may be missing cleanup function',
                    'recommendation': 'Add cleanup function to prevent memory leaks'
                })
                
            # Check for event listener leaks
            if self._has_event_listener_leaks(content):
                issues.append({
                    'file': str(file_path),
                    'type': 'event_listener_leaks',
                    'severity': 'high',
                    'description': 'Event listeners may not be properly cleaned up',
                    'recommendation': 'Remove event listeners in cleanup function'
                })
                
        return issues
        
    def analyze_infinite_loop_risks(self) -> List[Dict[str, Any]]:
        """Analyze for specific infinite loop risks"""
        print("\n🔄 Analyzing infinite loop risks...")
        
        issues = []
        
        for file_path in self.frontend_dir.rglob("*.tsx"):
            content = self._safe_read_file(file_path)
            if not content:
                continue
            
            # Check for recursive function calls
            if self._has_recursive_risks(content):
                issues.append({
                    'file': str(file_path),
                    'type': 'recursive_risks',
                    'severity': 'high',
                    'description': 'Potential recursive function calls detected',
                    'recommendation': 'Add base cases and limits to recursive functions'
                })
                
            # Check for while loops without exit conditions
            if self._has_unsafe_loops(content):
                issues.append({
                    'file': str(file_path),
                    'type': 'unsafe_loops',
                    'severity': 'critical',
                    'description': 'Loops without clear exit conditions',
                    'recommendation': 'Add timeout or iteration limits'
                })
                
        return issues
        
    # Helper methods for analysis
    def _has_missing_dependencies(self, effect_body: str, dependencies: str) -> bool:
        """Check if useEffect has missing dependencies"""
        # Look for state/prop usage in effect body
        state_refs = re.findall(r'\b(\w+)\s*\.\s*\w+|\b(\w+)\s*\[', effect_body)
        dep_list = [dep.strip() for dep in dependencies.split(',') if dep.strip()]
        
        for ref_match in state_refs:
            ref = ref_match[0] or ref_match[1]
            if ref and ref not in dep_list and ref not in ['console', 'window', 'document']:
                return True
        return False
        
    def _has_unconditional_state_updates(self, effect_body: str) -> bool:
        """Check for unconditional state updates in useEffect"""
        # Look for setState calls without conditions
        set_state_pattern = r'set\w+\s*\('
        if_pattern = r'\bif\s*\('
        
        has_set_state = bool(re.search(set_state_pattern, effect_body))
        has_conditions = bool(re.search(if_pattern, effect_body))
        
        return has_set_state and not has_conditions
        
    def _has_rapid_state_updates(self, content: str) -> bool:
        """Check for patterns that could cause rapid state updates"""
        # Look for multiple setState calls in quick succession
        set_state_calls = len(re.findall(r'set\w+\s*\(', content))
        return set_state_calls > 5
        
    def _has_state_updates_in_render(self, content: str) -> bool:
        """Check for state updates in render function"""
        # Look for setState calls outside of useEffect/event handlers
        lines = content.split('\n')
        in_render = False
        in_effect_or_handler = False
        
        for line in lines:
            if 'const ' in line and '= () => {' in line:
                in_render = True
                in_effect_or_handler = False
            elif 'useEffect' in line or 'onClick' in line or 'onChange' in line:
                in_effect_or_handler = True
            elif 'set' in line and '(' in line and in_render and not in_effect_or_handler:
                return True
                
        return False
        
    def _has_blocking_operations_in_message_handler(self, content: str) -> bool:
        """Check for blocking operations in WebSocket message handler"""
        blocking_patterns = [
            r'while\s*\(',
            r'for\s*\([^)]*;\s*[^)]*;\s*[^)]*\)\s*{[^}]*}',
            r'JSON\.parse\s*\([^)]*\)',  # Large JSON parsing
            r'\.map\s*\([^)]*\)\s*\.map',  # Chained maps
        ]
        
        for pattern in blocking_patterns:
            if re.search(pattern, content):
                return True
        return False
        
    def _has_excessive_message_processing(self, content: str) -> bool:
        """Check for excessive message processing"""
        message_types = len(re.findall(r'data\.type\s*===\s*[\'"]([^\'"]+)[\'"]', content))
        return message_types > 10
        
    def _should_use_react_memo(self, content: str) -> bool:
        """Check if component should use React.memo"""
        # Look for functional components with props
        has_props = 'props' in content or ': React.FC<' in content
        has_memo = 'React.memo' in content or 'memo(' in content
        return has_props and not has_memo
        
    def _has_expensive_operations_in_render(self, content: str) -> bool:
        """Check for expensive operations in render"""
        expensive_patterns = [
            r'\.sort\s*\(',
            r'\.filter\s*\([^)]*\)\.map',
            r'JSON\.parse\s*\(',
            r'new Date\s*\(',
            r'Math\.\w+\s*\(',
        ]
        
        for pattern in expensive_patterns:
            if re.search(pattern, content):
                return True
        return False
        
    def _has_large_inline_objects(self, content: str) -> bool:
        """Check for large inline objects"""
        # Look for object literals with many properties
        object_pattern = r'\{[^}]*:[^}]*:[^}]*:[^}]*\}'
        return bool(re.search(object_pattern, content))
        
    def _has_missing_cleanup(self, content: str) -> bool:
        """Check for missing cleanup in useEffect"""
        has_useEffect = 'useEffect' in content
        has_return = 'return () =>' in content or 'return function' in content
        return has_useEffect and not has_return
        
    def _has_event_listener_leaks(self, content: str) -> bool:
        """Check for event listener leaks"""
        has_add_listener = 'addEventListener' in content
        has_remove_listener = 'removeEventListener' in content
        return has_add_listener and not has_remove_listener
        
    def _has_recursive_risks(self, content: str) -> bool:
        """Check for recursive function risks"""
        # Look for functions that call themselves
        function_names = re.findall(r'const\s+(\w+)\s*=', content)
        for name in function_names:
            if name in content[content.find(name):]:
                return True
        return False
        
    def _has_unsafe_loops(self, content: str) -> bool:
        """Check for unsafe loops"""
        # Look for while loops without clear exit conditions
        while_pattern = r'while\s*\([^)]*\)\s*\{'
        return bool(re.search(while_pattern, content))
        
    def generate_summary(self) -> Dict[str, Any]:
        """Generate analysis summary"""
        return {
            'total_issues': len(self.issues),
            'critical_issues': len([i for i in self.issues if i.get('severity') == 'critical']),
            'high_issues': len([i for i in self.issues if i.get('severity') == 'high']),
            'medium_issues': len([i for i in self.issues if i.get('severity') == 'medium']),
            'recommendations': [
                "Consider using React.memo for components that re-render frequently",
                "Use useMemo and useCallback for expensive operations",
                "Implement proper cleanup in useEffect hooks",
                "Add timeout protection for WebSocket message processing",
                "Consider message throttling for high-frequency updates"
            ]
        }
        
    def print_analysis_results(self, results: Dict[str, Any]):
        """Print analysis results in a readable format"""
        print("\n" + "=" * 60)
        print("📊 Frontend Analysis Results")
        print("=" * 60)
        
        for category, issues in results.items():
            if category == 'summary':
                continue
                
            if issues:
                print(f"\n🔍 {category.replace('_', ' ').title()}:")
                for issue in issues:
                    severity_icon = {
                        'critical': '🚨',
                        'high': '⚠️',
                        'medium': '⚡',
                        'low': 'ℹ️'
                    }.get(issue.get('severity', 'low'), 'ℹ️')
                    
                    print(f"  {severity_icon} {issue['description']}")
                    print(f"     File: {issue['file']}")
                    if 'recommendation' in issue:
                        print(f"     💡 {issue['recommendation']}")
                    print()
                    
        # Print summary
        summary = results['summary']
        print(f"\n📈 Summary:")
        print(f"  Total Issues: {summary['total_issues']}")
        print(f"  Critical: {summary['critical_issues']}")
        print(f"  High: {summary['high_issues']}")
        print(f"  Medium: {summary['medium_issues']}")
        
        if summary['total_issues'] > 0:
            print(f"\n💡 Recommendations:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        else:
            print(f"\n✅ No major issues detected in frontend code!")

def main():
    """Main analysis runner"""
    analyzer = FrontendAnalyzer()
    results = analyzer.analyze_all()
    analyzer.print_analysis_results(results)
    
    # Save results to file
    with open('frontend_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Detailed results saved to scripts/frontend_analysis_results.json")

if __name__ == "__main__":
    main() 