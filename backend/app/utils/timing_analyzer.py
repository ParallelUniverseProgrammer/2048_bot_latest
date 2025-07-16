"""
Timing Analyzer for Performance Diagnostics
===========================================

Analyzes timing logs from PPO trainer and training manager to identify
performance bottlenecks and provide actionable insights for optimization.
"""

import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

class TimingAnalyzer:
    """Analyze timing logs and generate performance reports"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.ppo_log = os.path.join(log_dir, "ppo_training_timing.log")
        self.manager_log = os.path.join(log_dir, "training_manager_timing.log")
        self.ppo_summary = os.path.join(log_dir, "ppo_timing_summary.json")
        self.manager_summary = os.path.join(log_dir, "training_manager_timing_summary.json")
        
    def parse_log_file(self, log_file: str) -> pd.DataFrame:
        """Parse timing log file into DataFrame"""
        if not os.path.exists(log_file):
            print(f"Warning: Log file {log_file} not found")
            return pd.DataFrame()
        
        data = []
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('===') or line.startswith('timestamp,operation'):
                    continue
                
                try:
                    parts = line.split(',', 4)  # Split into 5 parts, last part may contain commas
                    if len(parts) >= 5:
                        timestamp, operation, phase, duration_ms, details = parts
                        data.append({
                            'timestamp': timestamp,
                            'operation': operation,
                            'phase': phase,
                            'duration_ms': float(duration_ms) if duration_ms != '0' else 0.0,
                            'details': details
                        })
                except Exception as e:
                    print(f"Warning: Could not parse line: {line[:100]}... Error: {e}")
        
        return pd.DataFrame(data)
    
    def load_summary(self, summary_file: str) -> Dict[str, Any]:
        """Load timing summary JSON file"""
        if not os.path.exists(summary_file):
            print(f"Warning: Summary file {summary_file} not found")
            return {}
        
        try:
            with open(summary_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading summary {summary_file}: {e}")
            return {}
    
    def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks from timing data"""
        
        # Load data
        ppo_df = self.parse_log_file(self.ppo_log)
        manager_df = self.parse_log_file(self.manager_log)
        ppo_summary = self.load_summary(self.ppo_summary)
        manager_summary = self.load_summary(self.manager_summary)
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'bottlenecks': [],
            'recommendations': [],
            'performance_metrics': {},
            'phase_breakdown': {},
            'operation_breakdown': {}
        }
        
        # Analyze PPO trainer bottlenecks
        if not ppo_df.empty:
            ppo_analysis = self._analyze_ppo_performance(ppo_df, ppo_summary)
            analysis['bottlenecks'].extend(ppo_analysis['bottlenecks'])
            analysis['recommendations'].extend(ppo_analysis['recommendations'])
            analysis['performance_metrics']['ppo'] = ppo_analysis['metrics']
            analysis['phase_breakdown']['ppo'] = ppo_analysis['phase_breakdown']
            analysis['operation_breakdown']['ppo'] = ppo_analysis['operation_breakdown']
        
        # Analyze training manager bottlenecks
        if not manager_df.empty:
            manager_analysis = self._analyze_manager_performance(manager_df, manager_summary)
            analysis['bottlenecks'].extend(manager_analysis['bottlenecks'])
            analysis['recommendations'].extend(manager_analysis['recommendations'])
            analysis['performance_metrics']['manager'] = manager_analysis['metrics']
            analysis['phase_breakdown']['manager'] = manager_analysis['phase_breakdown']
            analysis['operation_breakdown']['manager'] = manager_analysis['operation_breakdown']
        
        # Cross-component analysis
        cross_analysis = self._analyze_cross_component_performance(ppo_df, manager_df)
        analysis['bottlenecks'].extend(cross_analysis['bottlenecks'])
        analysis['recommendations'].extend(cross_analysis['recommendations'])
        
        return analysis
    
    def _analyze_ppo_performance(self, df: pd.DataFrame, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze PPO trainer performance"""
        
        analysis = {
            'bottlenecks': [],
            'recommendations': [],
            'metrics': {},
            'phase_breakdown': {},
            'operation_breakdown': {}
        }
        
        # Filter for END events (completed operations)
        end_events = df[df['operation'].str.startswith('END_')].copy()
        end_events['operation'] = end_events['operation'].str.replace('END_', '')
        
        if end_events.empty:
            return analysis
        
        # Phase breakdown
        phase_stats = end_events.groupby('phase').agg({
            'duration_ms': ['count', 'mean', 'std', 'min', 'max', 'sum']
        }).round(2)
        analysis['phase_breakdown'] = phase_stats.to_dict()
        
        # Operation breakdown
        operation_stats = end_events.groupby('operation').agg({
            'duration_ms': ['count', 'mean', 'std', 'min', 'max', 'sum']
        }).round(2)
        analysis['operation_breakdown'] = operation_stats.to_dict()
        
        # Identify bottlenecks
        avg_durations = end_events.groupby('operation')['duration_ms'].mean().sort_values(ascending=False)
        
        # Top bottlenecks
        for operation, avg_duration in avg_durations.head(5).items():
            if avg_duration > 100:  # Operations taking more than 100ms on average
                analysis['bottlenecks'].append({
                    'component': 'PPO Trainer',
                    'operation': operation,
                    'avg_duration_ms': avg_duration,
                    'severity': 'high' if avg_duration > 500 else 'medium',
                    'description': f"{operation} takes {avg_duration:.1f}ms on average"
                })
        
        # Specific recommendations based on bottlenecks
        if 'update_policy_training' in avg_durations.index:
            policy_duration = avg_durations['update_policy_training']
            if policy_duration > 200:
                analysis['recommendations'].append({
                    'priority': 'high',
                    'component': 'PPO Trainer',
                    'recommendation': f"Policy updates are slow ({policy_duration:.1f}ms avg). Consider reducing batch size or PPO epochs.",
                    'impact': 'training_speed'
                })
        
        if 'batch_forward_training' in avg_durations.index:
            forward_duration = avg_durations['batch_forward_training']
            if forward_duration > 50:
                analysis['recommendations'].append({
                    'priority': 'medium',
                    'component': 'PPO Trainer',
                    'recommendation': f"Model forward pass is slow ({forward_duration:.1f}ms avg). Consider model optimization or smaller batches.",
                    'impact': 'inference_speed'
                })
        
        # Overall metrics
        analysis['metrics'] = {
            'total_operations': len(end_events),
            'avg_operation_duration': end_events['duration_ms'].mean(),
            'total_time': end_events['duration_ms'].sum(),
            'slowest_operation': avg_durations.index[0] if len(avg_durations) > 0 else None,
            'slowest_duration': avg_durations.iloc[0] if len(avg_durations) > 0 else None
        }
        
        return analysis
    
    def _analyze_manager_performance(self, df: pd.DataFrame, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze training manager performance"""
        
        analysis = {
            'bottlenecks': [],
            'recommendations': [],
            'metrics': {},
            'phase_breakdown': {},
            'operation_breakdown': {}
        }
        
        # Filter for END events (completed operations)
        end_events = df[df['operation'].str.startswith('END_')].copy()
        end_events['operation'] = end_events['operation'].str.replace('END_', '')
        
        if end_events.empty:
            return analysis
        
        # Phase breakdown
        phase_stats = end_events.groupby('phase').agg({
            'duration_ms': ['count', 'mean', 'std', 'min', 'max', 'sum']
        }).round(2)
        analysis['phase_breakdown'] = phase_stats.to_dict()
        
        # Operation breakdown
        operation_stats = end_events.groupby('operation').agg({
            'duration_ms': ['count', 'mean', 'std', 'min', 'max', 'sum']
        }).round(2)
        analysis['operation_breakdown'] = operation_stats.to_dict()
        
        # Identify bottlenecks
        avg_durations = end_events.groupby('operation')['duration_ms'].mean().sort_values(ascending=False)
        
        # Top bottlenecks
        for operation, avg_duration in avg_durations.head(5).items():
            if avg_duration > 50:  # Operations taking more than 50ms on average
                analysis['bottlenecks'].append({
                    'component': 'Training Manager',
                    'operation': operation,
                    'avg_duration_ms': avg_duration,
                    'severity': 'high' if avg_duration > 200 else 'medium',
                    'description': f"{operation} takes {avg_duration:.1f}ms on average"
                })
        
        # Specific recommendations
        if 'parallel_training_training' in avg_durations.index:
            parallel_duration = avg_durations['parallel_training_training']
            if parallel_duration > 1000:  # More than 1 second
                analysis['recommendations'].append({
                    'priority': 'high',
                    'component': 'Training Manager',
                    'recommendation': f"Parallel training is very slow ({parallel_duration:.1f}ms avg). Consider reducing number of environments or optimizing PPO.",
                    'impact': 'training_speed'
                })
        
        if 'metrics_broadcast_websocket' in avg_durations.index:
            broadcast_duration = avg_durations['metrics_broadcast_websocket']
            if broadcast_duration > 100:
                analysis['recommendations'].append({
                    'priority': 'medium',
                    'component': 'Training Manager',
                    'recommendation': f"WebSocket broadcasting is slow ({broadcast_duration:.1f}ms avg). Consider reducing broadcast frequency or optimizing data size.",
                    'impact': 'ui_responsiveness'
                })
        
        # Overall metrics
        analysis['metrics'] = {
            'total_operations': len(end_events),
            'avg_operation_duration': end_events['duration_ms'].mean(),
            'total_time': end_events['duration_ms'].sum(),
            'slowest_operation': avg_durations.index[0] if len(avg_durations) > 0 else None,
            'slowest_duration': avg_durations.iloc[0] if len(avg_durations) > 0 else None
        }
        
        return analysis
    
    def _analyze_cross_component_performance(self, ppo_df: pd.DataFrame, manager_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance across components"""
        
        analysis = {
            'bottlenecks': [],
            'recommendations': []
        }
        
        # Check for coordination issues
        if not ppo_df.empty and not manager_df.empty:
            # Look for potential blocking operations
            ppo_end_events = ppo_df[ppo_df['operation'].str.startswith('END_')].copy()
            manager_end_events = manager_df[manager_df['operation'].str.startswith('END_')].copy()
            
            # Check if PPO operations are blocking manager operations
            ppo_avg_duration = ppo_end_events['duration_ms'].mean() if not ppo_end_events.empty else 0
            manager_avg_duration = manager_end_events['duration_ms'].mean() if not manager_end_events.empty else 0
            
            if ppo_avg_duration > manager_avg_duration * 2:
                analysis['recommendations'].append({
                    'priority': 'high',
                    'component': 'Cross-Component',
                    'recommendation': f"PPO operations ({ppo_avg_duration:.1f}ms avg) are much slower than manager operations ({manager_avg_duration:.1f}ms avg). Consider PPO optimization.",
                    'impact': 'overall_performance'
                })
        
        return analysis
    
    def _stringify_dict_keys(self, d):
        """Recursively convert all dict keys to strings for JSON serialization"""
        if isinstance(d, dict):
            return {str(k): self._stringify_dict_keys(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._stringify_dict_keys(i) for i in d]
        else:
            return d

    def generate_report(self, output_file: str = "logs/performance_analysis_report.json") -> str:
        """Generate comprehensive performance analysis report"""
        
        analysis = self.analyze_performance_bottlenecks()
        
        # Stringify all dict keys for JSON serialization
        for section in ["phase_breakdown", "operation_breakdown"]:
            for comp in analysis.get(section, {}):
                analysis[section][comp] = self._stringify_dict_keys(analysis[section][comp])
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write report
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Generate summary
        summary = self._generate_summary(analysis)
        
        # Write summary to console
        print("=" * 80)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Report generated: {output_file}")
        print(f"Analysis timestamp: {analysis['timestamp']}")
        print()
        
        print("TOP BOTTLENECKS:")
        for i, bottleneck in enumerate(analysis['bottlenecks'][:5], 1):
            print(f"{i}. {bottleneck['component']}: {bottleneck['description']} ({bottleneck['severity']} severity)")
        print()
        
        print("TOP RECOMMENDATIONS:")
        for i, rec in enumerate(analysis['recommendations'][:5], 1):
            print(f"{i}. [{rec['priority'].upper()}] {rec['recommendation']}")
        print()
        
        if 'ppo' in analysis['performance_metrics']:
            ppo_metrics = analysis['performance_metrics']['ppo']
            print(f"PPO TRAINER METRICS:")
            print(f"  Total operations: {ppo_metrics.get('total_operations', 0)}")
            print(f"  Average duration: {ppo_metrics.get('avg_operation_duration', 0):.1f}ms")
            print(f"  Total time: {ppo_metrics.get('total_time', 0):.1f}ms")
            print()
        
        if 'manager' in analysis['performance_metrics']:
            manager_metrics = analysis['performance_metrics']['manager']
            print(f"TRAINING MANAGER METRICS:")
            print(f"  Total operations: {manager_metrics.get('total_operations', 0)}")
            print(f"  Average duration: {manager_metrics.get('avg_operation_duration', 0):.1f}ms")
            print(f"  Total time: {manager_metrics.get('total_time', 0):.1f}ms")
            print()
        
        return output_file
    
    def _generate_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate a text summary of the analysis"""
        
        summary_lines = []
        summary_lines.append("PERFORMANCE ANALYSIS SUMMARY")
        summary_lines.append("=" * 50)
        summary_lines.append(f"Generated: {analysis['timestamp']}")
        summary_lines.append("")
        
        # Bottlenecks summary
        summary_lines.append("BOTTLENECKS:")
        for bottleneck in analysis['bottlenecks'][:3]:  # Top 3
            summary_lines.append(f"  - {bottleneck['description']} ({bottleneck['severity']})")
        summary_lines.append("")
        
        # Recommendations summary
        summary_lines.append("RECOMMENDATIONS:")
        for rec in analysis['recommendations'][:3]:  # Top 3
            summary_lines.append(f"  - [{rec['priority']}] {rec['recommendation']}")
        summary_lines.append("")
        
        return "\n".join(summary_lines)
    
    def create_visualizations(self, output_dir: str = "logs/visualizations"):
        """Create performance visualization charts"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        ppo_df = self.parse_log_file(self.ppo_log)
        manager_df = self.parse_log_file(self.manager_log)
        
        if not ppo_df.empty:
            self._create_ppo_visualizations(ppo_df, output_dir)
        
        if not manager_df.empty:
            self._create_manager_visualizations(manager_df, output_dir)
    
    def _create_ppo_visualizations(self, df: pd.DataFrame, output_dir: str):
        """Create PPO-specific visualizations"""
        
        # Filter for END events
        end_events = df[df['operation'].str.startswith('END_')].copy()
        end_events['operation'] = end_events['operation'].str.replace('END_', '')
        
        if end_events.empty:
            return
        
        # Operation duration distribution
        plt.figure(figsize=(12, 8))
        
        # Top operations by average duration
        top_ops = end_events.groupby('operation')['duration_ms'].mean().sort_values(ascending=False).head(10)
        
        plt.subplot(2, 2, 1)
        top_ops.plot(kind='bar')
        plt.title('Top 10 Operations by Average Duration (PPO)')
        plt.ylabel('Average Duration (ms)')
        plt.xticks(rotation=45, ha='right')
        
        # Phase breakdown
        plt.subplot(2, 2, 2)
        phase_durations = end_events.groupby('phase')['duration_ms'].sum()
        phase_durations.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Time Distribution by Phase (PPO)')
        
        # Duration over time (if timestamps are available)
        plt.subplot(2, 2, 3)
        if 'timestamp' in end_events.columns:
            try:
                end_events['timestamp'] = pd.to_datetime(end_events['timestamp'])
                end_events.set_index('timestamp')['duration_ms'].rolling(10).mean().plot()
                plt.title('Operation Duration Trend (PPO)')
                plt.ylabel('Duration (ms)')
            except:
                pass
        
        # Operation frequency
        plt.subplot(2, 2, 4)
        op_counts = end_events['operation'].value_counts().head(10)
        op_counts.plot(kind='bar')
        plt.title('Operation Frequency (PPO)')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ppo_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_manager_visualizations(self, df: pd.DataFrame, output_dir: str):
        """Create training manager visualizations"""
        
        # Filter for END events
        end_events = df[df['operation'].str.startswith('END_')].copy()
        end_events['operation'] = end_events['operation'].str.replace('END_', '')
        
        if end_events.empty:
            return
        
        # Operation duration distribution
        plt.figure(figsize=(12, 8))
        
        # Top operations by average duration
        top_ops = end_events.groupby('operation')['duration_ms'].mean().sort_values(ascending=False).head(10)
        
        plt.subplot(2, 2, 1)
        top_ops.plot(kind='bar')
        plt.title('Top 10 Operations by Average Duration (Manager)')
        plt.ylabel('Average Duration (ms)')
        plt.xticks(rotation=45, ha='right')
        
        # Phase breakdown
        plt.subplot(2, 2, 2)
        phase_durations = end_events.groupby('phase')['duration_ms'].sum()
        phase_durations.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Time Distribution by Phase (Manager)')
        
        # Duration over time
        plt.subplot(2, 2, 3)
        if 'timestamp' in end_events.columns:
            try:
                end_events['timestamp'] = pd.to_datetime(end_events['timestamp'])
                end_events.set_index('timestamp')['duration_ms'].rolling(10).mean().plot()
                plt.title('Operation Duration Trend (Manager)')
                plt.ylabel('Duration (ms)')
            except:
                pass
        
        # Operation frequency
        plt.subplot(2, 2, 4)
        op_counts = end_events['operation'].value_counts().head(10)
        op_counts.plot(kind='bar')
        plt.title('Operation Frequency (Manager)')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'manager_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run timing analysis"""
    
    analyzer = TimingAnalyzer()
    
    # Generate report
    report_file = analyzer.generate_report()
    
    # Create visualizations
    try:
        analyzer.create_visualizations()
        print("Visualizations created in logs/visualizations/")
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")
    
    print(f"\nAnalysis complete! Report saved to: {report_file}")


if __name__ == "__main__":
    main() 