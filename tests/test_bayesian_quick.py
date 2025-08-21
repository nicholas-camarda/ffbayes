#!/usr/bin/env python3
"""
test_bayesian_quick.py - Quick Test for Bayesian Analysis
Runs the Bayesian model with fast parameters for testing.
"""

import sys
from pathlib import Path

# Use package import for analysis module
sys.path.append(str(Path.cwd() / 'src'))
from ffbayes.analysis.bayesian_hierarchical_ff_unified import bayesian_hierarchical_ff_unified


def main():
    """Run Bayesian analysis with quick test parameters."""
    print("=" * 60)
    print("QUICK TEST: Bayesian Hierarchical Fantasy Football Model")
    print("=" * 60)
    print("Using fast parameters for testing:")
    print("  • Cores: 2")
    print("  • Draws: 100")
    print("  • Tune: 50")
    print("  • Chains: 2")
    print("  • Predictive samples: 50")
    print("=" * 60)
    
    try:
        # Run with quick test parameters
        trace, results = bayesian_hierarchical_ff_unified(
            path_to_data_directory='datasets',
            cores=2,              # Fewer cores
            draws=100,            # Very few draws
            tune=50,              # Minimal tuning
            chains=2,             # Fewer chains
            predictive_samples=50, # Fewer samples
            use_existing_trace=True  # Try to use existing trace first
        )
        
        print("\n🎉 Quick test completed successfully!")
        print(f"📊 Results: {results}")
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        print("🔧 Check the error and fix issues")


if __name__ == "__main__":
    main()
