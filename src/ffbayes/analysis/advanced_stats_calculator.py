#!/usr/bin/env python3
"""
Advanced Stats Calculator for Fantasy Football
Calculates meaningful advanced metrics using existing fantasy football data.
"""

import os

import numpy as np
import pandas as pd

# Check for QUICK_TEST mode
QUICK_TEST = os.getenv('QUICK_TEST', 'false').lower() == 'true'


def calculate_advanced_stats_from_existing_data(base_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate advanced stats from existing fantasy football data.
    
    Args:
        base_data (pd.DataFrame): Existing dataset with fantasy points, player info, game context
        
    Returns:
        pd.DataFrame: Original data with advanced stats columns added
    """
    print("Calculating advanced stats from existing data...")
    
    # Make a copy to avoid modifying original data
    data = base_data.copy()
    
    # Ensure required columns exist
    required_cols = ['Name', 'Position', 'Season', 'G#', 'FantPt', 'Tm', 'Opp', 'Date', '7_game_avg']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert Date to datetime for temporal calculations
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Calculate ALL advanced stats (QUICK_TEST only affects PyMC parameters, not feature engineering)
    print("  🚀 Calculating full advanced stats...")
    data = _calculate_usage_consistency_metrics(data)
    data = _calculate_team_usage_patterns(data)
    data = _calculate_situational_performance(data)
    data = _calculate_position_specific_metrics(data)
    data = _calculate_trend_analysis(data)
    
    # Clean up NaN values - replace with 0 for numeric columns to avoid PyMC issues
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].fillna(0)
    
    print(f"Advanced stats calculation complete. Added {len(data.columns) - len(base_data.columns)} new columns.")
    return data


def _calculate_usage_consistency_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate usage consistency metrics."""
    print("  Calculating usage consistency metrics...")
    
    # Group by player and season for consistency calculations
    player_stats = data.groupby(['Name', 'Season']).agg({
        'FantPt': ['mean', 'std', 'count'],
        'G#': 'min'  # First game number
    }).reset_index()
    
    # Flatten column names
    player_stats.columns = ['Name', 'Season', 'fantasy_mean', 'fantasy_std', 'games_played', 'first_game']
    
    # Calculate consistency score (coefficient of variation)
    player_stats['consistency_score'] = np.where(
        player_stats['fantasy_mean'] > 0,
        player_stats['fantasy_std'] / player_stats['fantasy_mean'],
        np.nan
    )
    
    # Calculate boom/bust ratio (games above 20 vs below 5 points)
    boom_bust_data = data.groupby(['Name', 'Season'], group_keys=False).apply(
        lambda x: _calculate_boom_bust_ratio(x['FantPt'])
    ).reset_index(name='boom_bust_ratio')
    
    # Calculate floor/ceiling spread (90th - 10th percentile)
    floor_ceiling_data = data.groupby(['Name', 'Season'], group_keys=False).apply(
        lambda x: _calculate_floor_ceiling_spread(x['FantPt'])
    ).reset_index(name='floor_ceiling_spread')
    
    # Merge back to main data
    consistency_metrics = player_stats.merge(boom_bust_data, on=['Name', 'Season'], how='left')
    consistency_metrics = consistency_metrics.merge(floor_ceiling_data, on=['Name', 'Season'], how='left')
    
    # Select only the new columns to merge
    new_cols = ['consistency_score', 'boom_bust_ratio', 'floor_ceiling_spread']
    data = data.merge(
        consistency_metrics[['Name', 'Season'] + new_cols], 
        on=['Name', 'Season'], 
        how='left'
    )
    
    return data


def _calculate_boom_bust_ratio(fantasy_points: pd.Series) -> float:
    """Calculate boom/bust ratio for a player's fantasy points."""
    if len(fantasy_points) < 3:  # Need at least 3 games for meaningful ratio
        return np.nan
    
    boom_games = (fantasy_points >= 20).sum()
    bust_games = (fantasy_points <= 5).sum()
    
    if bust_games == 0:
        return boom_games if boom_games > 0 else np.nan
    
    return boom_games / bust_games


def _calculate_floor_ceiling_spread(fantasy_points: pd.Series) -> float:
    """Calculate floor/ceiling spread (90th - 10th percentile)."""
    if len(fantasy_points) < 5:  # Need at least 5 games for percentiles
        return np.nan
    
    try:
        p90 = fantasy_points.quantile(0.9)
        p10 = fantasy_points.quantile(0.1)
        return p90 - p10
    except:
        return np.nan


def _calculate_team_usage_patterns(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate team usage patterns."""
    print("  Calculating team usage patterns...")
    
    # Calculate team total fantasy points per game
    team_game_totals = data.groupby(['Tm', 'Season', 'G#'])['FantPt'].sum().reset_index()
    team_game_totals = team_game_totals.rename(columns={'FantPt': 'team_total_fantasy'})
    
    # Merge team totals back to main data
    data = data.merge(team_game_totals, on=['Tm', 'Season', 'G#'], how='left')
    
    # Calculate player's fantasy points as percentage of team total
    data['team_usage_pct'] = np.where(
        data['team_total_fantasy'] > 0,
        (data['FantPt'] / data['team_total_fantasy']) * 100,
        0
    )
    
    # Calculate position rank on team (among same position players)
    data = _calculate_position_rank_on_team(data)
    
    # Calculate team dependency (correlation with team success)
    data = _calculate_team_dependency(data)
    
    return data


def _calculate_position_rank_on_team(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate player's rank among same position on team."""
    # Group by team, season, game, position and rank by fantasy points
    position_ranks = data.groupby(['Tm', 'Season', 'G#', 'Position'])['FantPt'].rank(
        method='dense', ascending=False
    )
    
    # Add the rank column directly to the data
    data['position_rank_on_team'] = position_ranks.values
    
    return data


def _calculate_team_dependency(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate how much team success affects player performance."""
    # This is a simplified calculation - in practice, you might want more sophisticated correlation
    # For now, we'll use the ratio of player's fantasy points to team's average fantasy points per game
    
    # Calculate team's average fantasy points per game
    team_avg = data.groupby(['Tm', 'Season'])['team_total_fantasy'].mean().reset_index()
    team_avg = team_avg.rename(columns={'team_total_fantasy': 'team_avg_fantasy'})
    
    # Merge back to main data
    data = data.merge(team_avg, on=['Tm', 'Season'], how='left')
    
    # Calculate team dependency ratio
    data['team_dependency'] = np.where(
        data['team_avg_fantasy'] > 0,
        data['FantPt'] / data['team_avg_fantasy'],
        1.0  # Default to 1.0 if no team data
    )
    
    return data


def _calculate_situational_performance(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate situational performance metrics."""
    print("  Calculating situational performance metrics...")
    
    # Calculate home/away performance difference
    data = _calculate_home_away_diff(data)
    
    # Calculate early/late season performance difference
    data = _calculate_early_late_season_diff(data)
    
    # Calculate opponent strength impact
    data = _calculate_opponent_strength_impact(data)
    
    return data


def _calculate_home_away_diff(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate home vs away performance difference."""
    # Group by player, season and calculate home vs away performance
    home_away_stats = data.groupby(['Name', 'Season', 'is_home'])['FantPt'].mean().reset_index()
    
    # Pivot to get home and away performance side by side
    home_away_pivot = home_away_stats.pivot_table(
        index=['Name', 'Season'], 
        columns='is_home', 
        values='FantPt'
    ).reset_index()
    
    # Handle column names dynamically (pivot might not have both home and away)
    expected_cols = ['Name', 'Season']
    if False in home_away_pivot.columns:
        expected_cols.append('away_performance')
        home_away_pivot = home_away_pivot.rename(columns={False: 'away_performance'})
    else:
        home_away_pivot['away_performance'] = np.nan
        
    if True in home_away_pivot.columns:
        expected_cols.append('home_performance')
        home_away_pivot = home_away_pivot.rename(columns={True: 'home_performance'})
    else:
        home_away_pivot['home_performance'] = np.nan
    
    # Calculate difference (home - away)
    home_away_pivot['home_away_diff'] = (
        home_away_pivot['home_performance'] - home_away_pivot['away_performance']
    )
    
    # Merge back to main data
    data = data.merge(
        home_away_pivot[['Name', 'Season', 'home_away_diff']], 
        on=['Name', 'Season'], 
        how='left'
    )
    
    return data


def _calculate_early_late_season_diff(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate early vs late season performance difference."""
    # Define early season (weeks 1-4) and late season (weeks 13-17)
    # Note: G# represents game number, not necessarily week number
    
    # Group by player, season and calculate early vs late performance
    early_late_stats = data.groupby(['Name', 'Season'], group_keys=False).apply(
        lambda x: _calculate_early_late_diff(x)
    ).reset_index(name='early_late_season_diff')
    
    # Merge back to main data
    data = data.merge(early_late_stats, on=['Name', 'Season'], how='left')
    
    return data


def _calculate_early_late_diff(player_data: pd.DataFrame) -> float:
    """Calculate early vs late season difference for a single player."""
    if len(player_data) < 6:  # Need at least 6 games
        return np.nan
    
    # Sort by game number
    sorted_data = player_data.sort_values('G#')
    
    # Early season: first 4 games or first 25% of games
    early_games = min(4, len(sorted_data) // 4)
    early_performance = sorted_data.head(early_games)['FantPt'].mean()
    
    # Late season: last 4 games or last 25% of games
    late_games = min(4, len(sorted_data) // 4)
    late_performance = sorted_data.tail(late_games)['FantPt'].mean()
    
    return late_performance - early_performance


def _calculate_opponent_strength_impact(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate performance against strong vs weak opponents."""
    # Calculate opponent strength based on average fantasy points allowed
    opponent_strength = data.groupby(['Opp', 'Season'])['FantPt'].mean().reset_index()
    opponent_strength = opponent_strength.rename(columns={'FantPt': 'opponent_avg_allowed'})
    
    # Calculate league average
    league_avg = opponent_strength['opponent_avg_allowed'].mean()
    
    # Normalize opponent strength (negative = strong defense, positive = weak defense)
    opponent_strength['opponent_strength_normalized'] = (
        league_avg - opponent_strength['opponent_avg_allowed']
    )
    
    # Merge back to main data
    data = data.merge(
        opponent_strength[['Opp', 'Season', 'opponent_strength_normalized']], 
        on=['Opp', 'Season'], 
        how='left'
    )
    
    # Calculate opponent strength impact (player performance vs opponent strength)
    data['opponent_strength_impact'] = (
        data['FantPt'] * data['opponent_strength_normalized']
    )
    
    return data


def _calculate_position_specific_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate position-specific advanced metrics."""
    print("  Calculating position-specific metrics...")
    
    # Initialize all position-specific columns
    data['qb_efficiency'] = np.nan
    # Note: rb_workload_consistency, wr_big_play_dependency, te_usage_reliability 
    # will be created during the merge operations
    
    # QB Efficiency (relative to league QB average)
    qb_data = data[data['Position'] == 'QB'].copy()
    if len(qb_data) > 0:
        qb_league_avg = qb_data['FantPt'].mean()
        qb_data['qb_efficiency'] = qb_data['FantPt'] / qb_league_avg if qb_league_avg > 0 else 1.0
        # Update main data
        data.loc[data['Position'] == 'QB', 'qb_efficiency'] = qb_data['qb_efficiency']
    
    # RB Workload Consistency (variance in fantasy points)
    rb_data = data[data['Position'] == 'RB'].copy()
    if len(rb_data) > 0:
        # Calculate consistency for each RB player
        rb_consistency = rb_data.groupby(['Name', 'Season'])['FantPt'].std()
        # Map the values back to the main DataFrame using the index
        data['rb_workload_consistency'] = data.set_index(['Name', 'Season']).index.map(rb_consistency).values
    
    # WR Big Play Dependency (% from big games)
    wr_data = data[data['Position'] == 'WR'].copy()
    if len(wr_data) > 0:
        wr_big_play = wr_data.groupby(['Name', 'Season'], group_keys=False).apply(
            lambda x: _calculate_big_play_dependency(x['FantPt'])
        )
        # Map the values back to the main DataFrame using the index
        data['wr_big_play_dependency'] = data.set_index(['Name', 'Season']).index.map(wr_big_play).values
    
    # TE Usage Reliability (% above replacement level)
    te_data = data[data['Position'] == 'TE'].copy()
    if len(te_data) > 0:
        te_reliability = te_data.groupby(['Name', 'Season'], group_keys=False).apply(
            lambda x: _calculate_te_reliability(x['FantPt'])
        )
        # Map the values back to the main DataFrame using the index
        data['te_usage_reliability'] = data.set_index(['Name', 'Season']).index.map(te_reliability).values
    
    return data


def _calculate_big_play_dependency(fantasy_points: pd.Series) -> float:
    """Calculate percentage of points from big games (>=15 points)."""
    if len(fantasy_points) < 3:
        return np.nan
    
    big_games = (fantasy_points >= 15).sum()
    total_games = len(fantasy_points)
    
    return (big_games / total_games) * 100


def _calculate_te_reliability(fantasy_points: pd.Series) -> float:
    """Calculate percentage of games above replacement level (>=5 points)."""
    if len(fantasy_points) < 3:
        return np.nan
    
    # TE replacement level is typically around 5 points
    replacement_level = 5.0
    above_replacement = (fantasy_points >= replacement_level).sum()
    total_games = len(fantasy_points)
    
    return (above_replacement / total_games) * 100


def _calculate_trend_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate trend analysis metrics."""
    print("  Calculating trend analysis...")
    
    # Calculate recent form (last 4 games vs season average)
    data = _calculate_recent_form(data)
    
    # Calculate season trend (improving/declining)
    data = _calculate_season_trend(data)
    
    # Calculate consistency over time
    data = _calculate_consistency_over_time(data)
    
    return data


def _calculate_recent_form(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate recent form (last 4 games vs season average)."""
    # Group by player and season, then calculate recent form
    recent_form_data = data.groupby(['Name', 'Season'], group_keys=False).apply(
        lambda x: _calculate_player_recent_form(x)
    ).reset_index(name='recent_form')
    
    # Merge back to main data
    data = data.merge(recent_form_data, on=['Name', 'Season'], how='left')
    
    return data


def _calculate_player_recent_form(player_data: pd.DataFrame) -> float:
    """Calculate recent form for a single player."""
    if len(player_data) < 5:  # Need at least 5 games
        return np.nan
    
    # Sort by game number
    sorted_data = player_data.sort_values('G#')
    
    # Last 4 games (or fewer if not enough games)
    recent_games = min(4, len(sorted_data))
    recent_performance = sorted_data.tail(recent_games)['FantPt'].mean()
    
    # Season average
    season_avg = player_data['FantPt'].mean()
    
    return recent_performance - season_avg


def _calculate_season_trend(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate season trend (improving/declining)."""
    # Group by player and season, then calculate season trend
    season_trend_data = data.groupby(['Name', 'Season'], group_keys=False).apply(
        lambda x: _calculate_player_season_trend(x)
    ).reset_index(name='season_trend')
    
    # Merge back to main data
    data = data.merge(season_trend_data, on=['Name', 'Season'], how='left')
    
    return data


def _calculate_player_season_trend(player_data: pd.DataFrame) -> float:
    """Calculate season trend for a single player."""
    if len(player_data) < 8:  # Need at least 8 games for trend
        return np.nan
    
    # Sort by game number
    sorted_data = player_data.sort_values('G#')
    
    # Split season into halves
    mid_point = len(sorted_data) // 2
    first_half = sorted_data.head(mid_point)['FantPt'].mean()
    second_half = sorted_data.tail(mid_point)['FantPt'].mean()
    
    return second_half - first_half


def _calculate_consistency_over_time(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate consistency throughout season."""
    # Group by player and season, then calculate consistency over time
    consistency_data = data.groupby(['Name', 'Season'], group_keys=False).apply(
        lambda x: _calculate_player_consistency_over_time(x)
    ).reset_index(name='consistency_over_time')
    
    # Merge back to main data
    data = data.merge(consistency_data, on=['Name', 'Season'], how='left')
    
    return data


def _calculate_player_consistency_over_time(player_data: pd.DataFrame) -> float:
    """Calculate consistency over time for a single player."""
    if len(player_data) < 6:  # Need at least 6 games
        return np.nan
    
    # Sort by game number
    sorted_data = player_data.sort_values('G#')
    
    # Calculate rolling standard deviation (consistency)
    rolling_std = sorted_data['FantPt'].rolling(window=3, min_periods=3).std()
    
    # Return average consistency (lower = more consistent)
    return rolling_std.mean()


if __name__ == "__main__":
    # Test the function with sample data
    print("Testing advanced stats calculator...")
    
    # Create sample data with all required columns
    sample_data = pd.DataFrame({
        'Name': ['Player1', 'Player1', 'Player2', 'Player2', 'Player3', 'Player3'],
        'Position': ['QB', 'QB', 'RB', 'RB', 'WR', 'WR'],
        'Season': [2023, 2023, 2023, 2023, 2023, 2023],
        'G#': [1, 2, 1, 2, 1, 2],
        'FantPt': [20.0, 15.0, 12.0, 18.0, 8.0, 22.0],
        'Tm': ['Team1', 'Team1', 'Team2', 'Team2', 'Team3', 'Team3'],
        'Opp': ['Team2', 'Team3', 'Team1', 'Team3', 'Team1', 'Team2'],
        'Date': ['2023-09-01', '2023-09-08', '2023-09-01', '2023-09-08', '2023-09-01', '2023-09-08'],
        '7_game_avg': [17.5, 17.5, 15.0, 15.0, 15.0, 15.0],
        'is_home': [1, 0, 0, 1, 0, 1]
    })
    
    try:
        result = calculate_advanced_stats_from_existing_data(sample_data)
        print("✅ Advanced stats calculation successful!")
        print(f"Input shape: {sample_data.shape}")
        print(f"Output shape: {result.shape}")
        print(f"New columns: {[col for col in result.columns if col not in sample_data.columns]}")
    except Exception as e:
        print(f"❌ Error in advanced stats calculation: {e}")
        import traceback
        traceback.print_exc()
