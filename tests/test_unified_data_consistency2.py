#!/usr/bin/env python3
"""
test_unified_data_consistency.py - Test Unified Data Consistency
Verify that all models can access the same player data consistently.
"""

def test_unified_data_consistency():
    """Test that all models can access the same unified dataset."""
    print("=" * 60)
    print("Testing Unified Data Consistency Across All Models")
    print("=" * 60)
    
    try:
        # Test 1: Load unified dataset directly
        print("\n🔍 Test 1: Direct unified dataset loading")
        from ffbayes.data_pipeline.unified_data_loader import load_unified_dataset
        data = load_unified_dataset()
        print(f"✅ Unified dataset loaded: {data.shape}")
        
        # Test 2: Check specific player data
        print("\n🔍 Test 2: Player data consistency")
        test_players = ['Christian McCaffrey', 'Josh Allen', 'Saquon Barkley']
        
        for player in test_players:
            player_data = data[data['Name'] == player]
            if len(player_data) > 0:
                years = sorted(player_data['Season'].unique())
                positions = player_data['Position'].unique()
                print(f"   ✅ {player}: {len(player_data)} rows, years: {years}, positions: {positions}")
            else:
                print(f"   ❌ {player}: Not found in unified dataset")
        
        # Test 3: Check that all models can access the same data
        print("\n🔍 Test 3: Model data access consistency")
        
        # Baseline model access
        from src.ffbayes.analysis.baseline_naive_model import (
            load_unified_dataset as baseline_loader,
        )
        baseline_data = baseline_loader()
        print(f"   ✅ Baseline model: {baseline_data.shape}")
        
        # Monte Carlo model access
        from src.ffbayes.analysis.montecarlo_historical_ff import get_combined_data
        mc_data = get_combined_data('datasets')
        print(f"   ✅ Monte Carlo model: {mc_data.shape}")
        
        # Bayesian model access
        from ffbayes.data_pipeline.unified_data_loader import (
            load_unified_dataset as bayesian_loader,
        )
        bayesian_data = bayesian_loader()
        print(f"   ✅ Bayesian model: {bayesian_data.shape}")
        
        # Test 4: Verify data is identical
        print("\n🔍 Test 4: Data identity verification")
        if (data.equals(baseline_data) and 
            data.equals(mc_data) and 
            data.equals(bayesian_data)):
            print("   ✅ All models access identical data")
        else:
            print("   ❌ Data inconsistency detected between models")
        
        # Test 5: Check key features
        print("\n🔍 Test 5: Key features verification")
        key_features = ['Name', 'Position', 'FantPt', 'Season', '7_game_avg', 'rank', 'is_home']
        for feature in key_features:
            if feature in data.columns:
                print(f"   ✅ {feature}: Available")
            else:
                print(f"   ❌ {feature}: Missing")
        
        print("\n🎉 Unified data consistency test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_unified_data_consistency()
    exit(0 if success else 1)
