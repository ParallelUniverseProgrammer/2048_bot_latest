#!/usr/bin/env python3
"""
Timing Analysis Script
=====================

Standalone script to analyze timing logs and generate performance reports.
Run this after training to identify bottlenecks and optimization opportunities.
"""

import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_path)

# Import after adding to path
try:
    from app.utils.timing_analyzer import TimingAnalyzer
except ImportError as e:
    print(f"Error importing timing analyzer: {e}")
    print(f"Backend path: {backend_path}")
    print("Make sure the backend directory structure is correct.")
    sys.exit(1)

def main():
    """Main function to run timing analysis"""
    
    print("Timing Analysis Tool")
    print("=" * 50)
    
    # Check if log files exist
    log_dir = "backend/logs"
    ppo_log = os.path.join(log_dir, "ppo_training_timing.log")
    manager_log = os.path.join(log_dir, "training_manager_timing.log")
    
    if not os.path.exists(ppo_log) and not os.path.exists(manager_log):
        print("No timing log files found!")
        print(f"Expected files:")
        print(f"  - {ppo_log}")
        print(f"  - {manager_log}")
        print()
        print("Make sure to run training first to generate timing logs.")
        return
    
    # Create analyzer
    analyzer = TimingAnalyzer(log_dir)
    
    # Generate report
    print("Analyzing timing data...")
    report_file = analyzer.generate_report()
    
    # Create visualizations
    print("Creating visualizations...")
    try:
        analyzer.create_visualizations()
        print("✓ Visualizations created in backend/logs/visualizations/")
    except Exception as e:
        print(f"⚠ Warning: Could not create visualizations: {e}")
        print("  (This requires matplotlib and seaborn)")
    
    print()
    print("Analysis complete!")
    print(f"Report saved to: {report_file}")
    print()
    print("Next steps:")
    print("1. Review the performance analysis report")
    print("2. Check the visualizations for performance patterns")
    print("3. Implement optimizations based on the recommendations")
    print("4. Re-run training and analysis to measure improvements")

if __name__ == "__main__":
    main() 