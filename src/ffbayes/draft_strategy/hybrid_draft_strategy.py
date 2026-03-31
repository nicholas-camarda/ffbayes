#!/usr/bin/env python3
"""
Hybrid Draft Strategy Generator.

This module orchestrates the complete hybrid draft strategy workflow:
1. Load and integrate VOR and Bayesian data
2. Create risk-adjusted rankings
3. Generate pick-by-pick recommendations
4. Create comprehensive Excel output

The output provides a human-readable, draft-ready Excel file that combines
traditional VOR rankings with Bayesian uncertainty analysis.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ffbayes.data_pipeline.hybrid_data_integration import (
    create_hybrid_dataset,
    load_hybrid_data_sources,
)
from ffbayes.draft_strategy.hybrid_excel_generation import (
    create_hybrid_excel_file,
    validate_excel_structure,
)
from ffbayes.draft_strategy.pick_by_pick_strategy import (
    generate_pick_by_pick_recommendations,
)
from ffbayes.draft_strategy.risk_adjusted_rankings import create_risk_adjusted_rankings
from ffbayes.utils.config_loader import get_config
from ffbayes.utils.path_constants import get_draft_strategy_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Load configuration from command line arguments and user config.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with configuration settings
    """
    logger.info("Loading hybrid draft strategy configuration...")
    
    # Load user config
    user_config = get_config()
    
    # Merge command line args with user config
    config = {
        'draft_position': args.draft_position or user_config.get_league_setting('draft_position', 10),
        'league_size': args.league_size or user_config.get_league_setting('league_size', 10),
        'risk_tolerance': args.risk_tolerance or user_config.get_league_setting('risk_tolerance', 'medium'),
        'num_picks': args.league_size * 16 if args.league_size else 160,  # 16 rounds
        'output_dir': Path(args.output_dir) if args.output_dir else get_draft_strategy_dir(datetime.now().year)
    }
    
    logger.info("✅ Configuration loaded:")
    logger.info(f"   Draft Position: {config['draft_position']}")
    logger.info(f"   League Size: {config['league_size']}")
    logger.info(f"   Risk Tolerance: {config['risk_tolerance']}")
    logger.info(f"   Number of Picks: {config['num_picks']}")
    logger.info(f"   Output Directory: {config['output_dir']}")
    
    return config


def create_hybrid_draft_strategy(config: Dict[str, Any]) -> str:
    """
    Create complete hybrid draft strategy with Excel output.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to the generated Excel file
    """
    logger.info("🚀 Starting hybrid draft strategy generation...")
    
    try:
        # Step 1: Load and integrate data sources
        logger.info("📊 Step 1: Loading and integrating data sources...")
        vor_data, bayesian_data = load_hybrid_data_sources()
        hybrid_data = create_hybrid_dataset(vor_data, bayesian_data)
        
        logger.info(f"✅ Data integration complete: {len(hybrid_data)} players")
        
        # Step 2: Create risk-adjusted rankings
        logger.info("📈 Step 2: Creating risk-adjusted rankings...")
        risk_adjusted_data = create_risk_adjusted_rankings(hybrid_data)
        
        logger.info(f"✅ Risk-adjusted rankings complete: {len(risk_adjusted_data)} players")
        
        # Step 3: Generate pick-by-pick recommendations
        logger.info("🎯 Step 3: Generating pick-by-pick recommendations...")
        recommendations = generate_pick_by_pick_recommendations(
            risk_adjusted_data,
            num_picks=config['num_picks'],
            risk_tolerance=config['risk_tolerance']
        )
        
        logger.info(f"✅ Pick-by-pick recommendations complete: {len(recommendations)} picks")
        
        # Step 4: Create output directory
        output_dir = config['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 5: Generate Excel file
        logger.info("📋 Step 4: Generating Excel output...")
        excel_filename = f"HYBRID_DRAFT_STRATEGY_pos{config['draft_position']}_{config['league_size']}teams_2025.xlsx"
        excel_path = output_dir / excel_filename
        
        # Create the Excel file with all components including pick-by-pick recommendations
        create_hybrid_excel_file(risk_adjusted_data, str(excel_path), recommendations)
        
        # Step 6: Validate Excel structure
        logger.info("🔍 Step 5: Validating Excel structure...")
        validate_excel_structure(str(excel_path))
        
        logger.info("🎉 Hybrid draft strategy generation complete!")
        logger.info(f"📁 Output file: {excel_path}")
        
        return str(excel_path)
        
    except Exception as e:
        logger.error(f"❌ Hybrid draft strategy generation failed: {e}")
        raise


def main():
    """Main function for hybrid draft strategy generation."""
    parser = argparse.ArgumentParser(
        description="Generate hybrid VOR + Bayesian draft strategy with Excel output"
    )
    
    parser.add_argument(
        "--draft-position",
        type=int,
        help="Draft position (1-12, default from user config)"
    )
    
    parser.add_argument(
        "--league-size",
        type=int,
        help="League size (8-16, default from user config)"
    )
    
    parser.add_argument(
        "--risk-tolerance",
        choices=['low', 'medium', 'high'],
        help="Risk tolerance level (default from user config)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: runtime pre-draft draft_strategy directory)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_configuration(args)
        
        # Create hybrid draft strategy
        excel_path = create_hybrid_draft_strategy(config)
        
        print(f"\n{'='*80}")
        print("🎉 HYBRID DRAFT STRATEGY GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"📁 Excel File: {excel_path}")
        from ffbayes.utils.path_constants import SNAKE_DRAFT_DATASETS_DIR
        from ffbayes.utils.vor_filename_generator import get_vor_csv_filename
        vor_csv = SNAKE_DRAFT_DATASETS_DIR / get_vor_csv_filename(datetime.now().year)
        if vor_csv.exists():
            print(f"📊 Players Analyzed: {len(pd.read_csv(vor_csv))}")
        print(f"🎯 Pick Recommendations: {config['num_picks']}")
        print(f"⚖️  Risk Tolerance: {config['risk_tolerance']}")
        print(f"🏈 Draft Position: {config['draft_position']}")
        print(f"👥 League Size: {config['league_size']}")
        print("\n📋 Excel file contains:")
        print("   • Early Draft (Rounds 1-5)")
        print("   • Mid Draft (Rounds 6-10)")
        print("   • Late Draft (Rounds 11-16)")
        print("   • Round-by-Round Strategy")
        print("   • Strategy Summary")
        print("   • Pick-by-Pick Recommendations")
        print("   • Statistics Guide")
        print("\n🎯 Ready for draft day!")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Hybrid draft strategy generation failed: {e}")
        print(f"\n❌ ERROR: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
