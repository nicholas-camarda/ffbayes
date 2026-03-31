#!/usr/bin/env python3
"""
Pick-by-Pick Strategy Generation

This module implements position scarcity analysis and generates
pick-by-pick draft recommendations based on VOR methodology and uncertainty data.
"""

import logging
from typing import Dict, Optional

import openpyxl
import pandas as pd
from openpyxl.utils.dataframe import dataframe_to_rows

# Configure logging
logger = logging.getLogger(__name__)


def analyze_position_scarcity(risk_adjusted_data: pd.DataFrame) -> Dict:
    """
    Analyze position scarcity using VOR methodology.
    
    Args:
        risk_adjusted_data: DataFrame with risk-adjusted rankings
        
    Returns:
        Dictionary with position scarcity analysis
    """
    logger.info("Analyzing position scarcity...")
    
    if risk_adjusted_data.empty:
        raise ValueError("Cannot analyze position scarcity with empty data")
    
    scarcity_analysis = {}
    
    for position in risk_adjusted_data['POS'].unique():
        pos_data = risk_adjusted_data[risk_adjusted_data['POS'] == position].copy()
        
        # Calculate scarcity metrics
        player_count = len(pos_data)
        top_players = pos_data.head(5)[['PLAYER', 'VOR', 'composite_score', 'risk_category']].to_dict('records')
        
        # Calculate scarcity score (inverse of player count, weighted by quality)
        avg_vor = pos_data['VOR'].mean()
        scarcity_score = (1 / player_count) * avg_vor * 10  # Scale for readability
        
        # Generate recommendation based on scarcity
        if scarcity_score > 2.0:
            recommendation = f"Draft {position} early - high scarcity and value"
        elif scarcity_score > 1.0:
            recommendation = f"Consider {position} in early-mid rounds"
        else:
            recommendation = f"Wait on {position} - good depth available"
        
        scarcity_analysis[position] = {
            'player_count': player_count,
            'top_players': top_players,
            'scarcity_score': scarcity_score,
            'avg_vor': avg_vor,
            'recommendation': recommendation
        }
    
    logger.info(f"✅ Position scarcity analysis completed for {len(scarcity_analysis)} positions")
    
    # Log scarcity scores
    for pos, analysis in scarcity_analysis.items():
        logger.info(f"   {pos}: {analysis['player_count']} players, scarcity score: {analysis['scarcity_score']:.2f}")
    
    return scarcity_analysis


def create_draft_position_strategy(risk_adjusted_data: pd.DataFrame, pick_number: int) -> Dict:
    """
    Create draft position-specific strategy for a given pick number.
    
    Args:
        risk_adjusted_data: DataFrame with risk-adjusted rankings
        pick_number: Draft pick number (1-based)
        
    Returns:
        Dictionary with strategy for this pick
    """
    if pick_number < 1:
        raise ValueError(f"Invalid pick number: {pick_number}")
    
    # Get available players (all remaining players)
    available_players = risk_adjusted_data.copy()
    
    # Analyze position scarcity
    scarcity = analyze_position_scarcity(risk_adjusted_data)
    
    # Determine pick strategy based on round
    if pick_number <= 5:
        # Early rounds: Focus on best available with slight position consideration
        strategy_type = "early_round"
        position_strategy = "Best available player with slight preference for scarce positions"
    elif pick_number <= 10:
        # Mid-early rounds: Balance value and position needs
        strategy_type = "mid_early_round"
        position_strategy = "Balance value with position scarcity"
    elif pick_number <= 15:
        # Mid rounds: Focus on position scarcity and value
        strategy_type = "mid_round"
        position_strategy = "Prioritize scarce positions with good value"
    else:
        # Late rounds: Focus on upside and position scarcity
        strategy_type = "late_round"
        position_strategy = "Target upside players in scarce positions"
    
    # Select recommended player
    if strategy_type == "early_round":
        # Take best available
        recommended_player = available_players.iloc[0]
    else:
        # Consider position scarcity
        recommended_player = select_player_with_scarcity_consideration(available_players, scarcity)
    
    # Generate backup options
    backup_options = generate_backup_options(available_players, recommended_player, scarcity)
    
    # Generate reasoning
    reasoning = generate_pick_reasoning(recommended_player, pick_number, scarcity, strategy_type)
    
    return {
        'recommended_player': recommended_player['PLAYER'],
        'position': recommended_player['POS'],
        'vor': recommended_player['VOR'],
        'composite_score': recommended_player['composite_score'],
        'risk_category': recommended_player['risk_category'],
        'reasoning': reasoning,
        'backup_options': backup_options,
        'position_strategy': position_strategy
    }


def select_player_with_scarcity_consideration(available_players: pd.DataFrame, scarcity: Dict) -> pd.Series:
    """
    Select player considering position scarcity.
    
    Args:
        available_players: Available players for this pick
        scarcity: Position scarcity analysis
        
    Returns:
        Selected player series
    """
    # Score each player based on value and scarcity
    player_scores = []
    
    for _, player in available_players.iterrows():
        pos = player['POS']
        scarcity_score = scarcity[pos]['scarcity_score']
        
        # Combine VOR value with scarcity consideration
        combined_score = player['VOR'] * (1 + scarcity_score * 0.1)
        
        player_scores.append({
            'player': player,
            'score': combined_score
        })
    
    # Sort by combined score and return top player
    player_scores.sort(key=lambda x: x['score'], reverse=True)
    return player_scores[0]['player']


def generate_backup_options(available_players: pd.DataFrame, recommended_player: pd.Series, scarcity: Dict) -> str:
    """
    Generate backup options for the recommended player.
    
    Args:
        available_players: Available players for this pick
        recommended_player: Recommended player
        scarcity: Position scarcity analysis
        
    Returns:
        String with backup options
    """
    # Remove recommended player from available options
    backup_players = available_players[available_players['PLAYER'] != recommended_player['PLAYER']].copy()
    
    if backup_players.empty:
        return "No backup options available - take best available player"
    
    # Get diverse backup options: best available, best at same position, and best at different position
    backup_list = []
    
    # 1. Best available player overall
    best_available = backup_players.iloc[0]
    backup_list.append(f"{best_available['PLAYER']} ({best_available['POS']}) - Best Available")
    
    # 2. Best player at the same position (if different from best available)
    same_pos_players = backup_players[backup_players['POS'] == recommended_player['POS']]
    if not same_pos_players.empty and same_pos_players.iloc[0]['PLAYER'] != best_available['PLAYER']:
        best_same_pos = same_pos_players.iloc[0]
        backup_list.append(f"{best_same_pos['PLAYER']} ({best_same_pos['POS']}) - Best {recommended_player['POS']}")
    
    # 3. Best player at a different position (for position diversity)
    different_pos_players = backup_players[backup_players['POS'] != recommended_player['POS']]
    if not different_pos_players.empty and different_pos_players.iloc[0]['PLAYER'] != best_available['PLAYER']:
        best_diff_pos = different_pos_players.iloc[0]
        backup_list.append(f"{best_diff_pos['PLAYER']} ({best_diff_pos['POS']}) - Best {best_diff_pos['POS']}")
    
    # Limit to 3 backup options
    backup_list = backup_list[:3]
    
    return "; ".join(backup_list)


def generate_pick_reasoning(player: pd.Series, pick_number: int, scarcity: Dict, strategy_type: str) -> str:
    """
    Generate reasoning for the pick recommendation.
    
    Args:
        player: Selected player
        pick_number: Draft pick number
        scarcity: Position scarcity analysis
        strategy_type: Type of strategy (early_round, mid_round, etc.)
        
    Returns:
        Reasoning string
    """
    pos = player['POS']
    pos_scarcity = scarcity[pos]
    
    if strategy_type == "early_round":
        reasoning = f"Best available player with excellent value (VOR: {player['VOR']:.1f}). "
        reasoning += f"{pos} position has {pos_scarcity['player_count']} quality players available."
    elif strategy_type == "mid_early_round":
        reasoning = f"Strong value pick (VOR: {player['VOR']:.1f}) with moderate {pos} scarcity. "
        reasoning += f"Only {pos_scarcity['player_count']} quality {pos}s remaining."
    elif strategy_type == "mid_round":
        reasoning = f"Addressing {pos} scarcity (only {pos_scarcity['player_count']} quality players left). "
        reasoning += f"Good value (VOR: {player['VOR']:.1f}) with {player['risk_category']} risk profile."
    else:  # late_round
        reasoning = f"Late-round {pos} target with upside potential. "
        reasoning += f"Risk category: {player['risk_category']}, VOR: {player['VOR']:.1f}. "
        reasoning += f"Position scarcity: {pos_scarcity['scarcity_score']:.2f}"
    
    return reasoning


def generate_pick_by_pick_recommendations(
    risk_adjusted_data: pd.DataFrame, 
    num_picks: int = 160,
    position_preference: Optional[str] = None,
    risk_tolerance: str = 'medium'
) -> pd.DataFrame:
    """
    Generate pick-by-pick recommendations for the entire draft.
    
    Args:
        risk_adjusted_data: DataFrame with risk-adjusted rankings
        num_picks: Number of picks to generate (default 160 for 16 rounds * 10 teams)
        position_preference: Optional position preference for early picks
        risk_tolerance: Risk tolerance setting
        
    Returns:
        DataFrame with pick-by-pick recommendations
    """
    logger.info(f"Generating pick-by-pick recommendations for {num_picks} picks...")
    
    # Filter out extremely low-value players (VOR < -50) to avoid selecting backup QBs and other low-value players
    # This ensures we focus on players who actually provide value
    filtered_data = risk_adjusted_data[risk_adjusted_data['VOR'] > -50].copy()
    logger.info(f"Filtered out {len(risk_adjusted_data) - len(filtered_data)} players with VOR <= -50")
    
    recommendations = []
    available_players = filtered_data.copy()
    
    pick_number = 1
    while pick_number <= num_picks and not available_players.empty:
        # Create strategy for this pick using available players
        strategy = create_draft_position_strategy(available_players, pick_number)
        
        # Apply position preference if specified and in early rounds
        if position_preference and pick_number <= 5:
            strategy = apply_position_preference(strategy, position_preference, available_players)
        
        # Apply risk tolerance adjustments
        strategy = apply_risk_tolerance(strategy, risk_tolerance)
        
        recommendations.append({
            'pick_number': pick_number,
            'recommended_player': strategy['recommended_player'],
            'position': strategy['position'],
            'vor': strategy['vor'],
            'composite_score': strategy['composite_score'],
            'risk_category': strategy['risk_category'],
            'reasoning': strategy['reasoning'],
            'backup_options': strategy['backup_options'],
            'position_strategy': strategy['position_strategy']
        })
        
        # Remove the selected player from available players
        available_players = available_players[available_players['PLAYER'] != strategy['recommended_player']].copy()
        available_players = available_players.reset_index(drop=True)
        
        pick_number += 1
    
    recommendations_df = pd.DataFrame(recommendations)
    
    logger.info(f"✅ Generated {len(recommendations_df)} pick-by-pick recommendations")
    
    return recommendations_df


def apply_position_preference(strategy: Dict, position_preference: str, risk_adjusted_data: pd.DataFrame) -> Dict:
    """
    Apply position preference to early round strategy.
    
    Args:
        strategy: Current strategy
        position_preference: Preferred position
        risk_adjusted_data: Full dataset
        
    Returns:
        Modified strategy
    """
    if strategy['position'] == position_preference:
        return strategy  # Already preferred position
    
    # Look for preferred position players in top 20
    preferred_players = risk_adjusted_data[
        (risk_adjusted_data['POS'] == position_preference) & 
        (risk_adjusted_data['composite_rank'] <= 20)
    ]
    
    if not preferred_players.empty:
        best_preferred = preferred_players.iloc[0]
        
        # Only switch if the preferred player is close in value
        if best_preferred['VOR'] >= strategy['vor'] * 0.9:  # Within 10% of original recommendation
            strategy['recommended_player'] = best_preferred['PLAYER']
            strategy['position'] = best_preferred['POS']
            strategy['vor'] = best_preferred['VOR']
            strategy['composite_score'] = best_preferred['composite_score']
            strategy['risk_category'] = best_preferred['risk_category']
            strategy['reasoning'] = f"Position preference ({position_preference}) with strong value (VOR: {best_preferred['VOR']:.1f})"
    
    return strategy


def apply_risk_tolerance(strategy: Dict, risk_tolerance: str) -> Dict:
    """
    Apply risk tolerance adjustments to strategy.
    
    Args:
        strategy: Current strategy
        risk_tolerance: Risk tolerance setting
        
    Returns:
        Modified strategy
    """
    if risk_tolerance == 'low' and strategy['risk_category'] == 'High Risk':
        # For low risk tolerance, avoid high risk players in early rounds
        strategy['reasoning'] += " Note: High risk player - consider safer alternative if risk-averse."
    elif risk_tolerance == 'high' and strategy['risk_category'] == 'Low Risk':
        # For high risk tolerance, suggest considering higher upside players
        strategy['reasoning'] += " Note: Safe player - consider higher upside alternatives if risk-seeking."
    
    return strategy


def validate_pick_strategy_data(pick_strategy_data: pd.DataFrame) -> None:
    """
    Validate pick strategy dataset for required columns and data quality.
    
    Args:
        pick_strategy_data: DataFrame to validate
        
    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating pick strategy data...")
    
    if pick_strategy_data.empty:
        raise ValueError("Pick strategy dataset is empty")
    
    # Check required columns
    required_columns = [
        'pick_number', 'recommended_player', 'position', 'vor', 
        'composite_score', 'risk_category', 'reasoning', 'backup_options'
    ]
    
    missing_columns = [col for col in required_columns if col not in pick_strategy_data.columns]
    if missing_columns:
        raise ValueError(f"Pick strategy dataset missing required columns: {missing_columns}")
    
    # Check that pick numbers are sequential
    expected_picks = list(range(1, len(pick_strategy_data) + 1))
    actual_picks = pick_strategy_data['pick_number'].tolist()
    
    if actual_picks != expected_picks:
        raise ValueError("Pick numbers are not sequential")
    
    # Check that reasoning is provided
    empty_reasoning = pick_strategy_data[pick_strategy_data['reasoning'].isna() | (pick_strategy_data['reasoning'] == '')]
    if not empty_reasoning.empty:
        raise ValueError(f"Found {len(empty_reasoning)} picks with empty reasoning")
    
    # Check that backup options are provided
    empty_backups = pick_strategy_data[pick_strategy_data['backup_options'].isna() | (pick_strategy_data['backup_options'] == '')]
    if not empty_backups.empty:
        raise ValueError(f"Found {len(empty_backups)} picks with empty backup options")
    
    logger.info("✅ Pick strategy data validation passed")


def create_pick_by_pick_sheet(workbook: openpyxl.Workbook, risk_adjusted_data: pd.DataFrame) -> None:
    """
    Create Pick-by-Pick Strategy sheet with risk-adjusted recommendations.
    
    Args:
        workbook: OpenPyXL workbook to add sheet to
        risk_adjusted_data: DataFrame with risk-adjusted rankings
    """
    logger.info("Creating Pick-by-Pick Strategy sheet...")
    
    # Create sheet
    ws = workbook.create_sheet('Pick-by-Pick Strategy')
    
    # Generate pick-by-pick recommendations
    recommendations = generate_pick_by_pick_recommendations(risk_adjusted_data)
    
    # Select columns for display
    display_columns = [
        'pick_number', 'recommended_player', 'position', 'vor', 
        'composite_score', 'risk_category', 'reasoning', 'backup_options'
    ]
    
    sheet_data = recommendations[display_columns].copy()
    
    # Add data to sheet
    for r in dataframe_to_rows(sheet_data, index=False, header=True):
        ws.append(r)
    
    # Apply formatting
    from ffbayes.draft_strategy.hybrid_excel_generation import (
        apply_risk_color_formatting,
        apply_vor_formatting,
    )
    apply_vor_formatting(ws)
    apply_risk_color_formatting(ws, 'risk_category')
    
    logger.info(f"✅ Pick-by-Pick Strategy sheet created with {len(sheet_data)} recommendations")


def main():
    """Main function for testing pick-by-pick strategy generation."""
    try:
        # Load risk-adjusted data
        from ffbayes.data_pipeline.hybrid_data_integration import (
            create_hybrid_dataset,
            load_hybrid_data_sources,
        )
        from ffbayes.draft_strategy.risk_adjusted_rankings import (
            create_risk_adjusted_rankings,
        )

        # Load and create risk-adjusted data
        vor_data, bayesian_data = load_hybrid_data_sources()
        hybrid_data = create_hybrid_dataset(vor_data, bayesian_data)
        risk_adjusted_data = create_risk_adjusted_rankings(hybrid_data)
        
        # Generate pick-by-pick recommendations
        recommendations = generate_pick_by_pick_recommendations(risk_adjusted_data)
        
        print("🎉 Pick-by-pick strategy generation successful!")
        print(f"📊 Generated {len(recommendations)} pick recommendations")
        print("📈 Top 5 recommendations:")
        for i, row in recommendations.head().iterrows():
            print(f"   Pick {row['pick_number']}: {row['recommended_player']} ({row['position']}) - VOR: {row['vor']:.1f}, Risk: {row['risk_category']}")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"❌ Pick-by-pick strategy generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
