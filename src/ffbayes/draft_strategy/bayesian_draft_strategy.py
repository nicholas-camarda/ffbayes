"""
Bayesian Draft Strategy Module

This module implements a tier-based Bayesian draft strategy that generates
multiple options per pick for optimal team construction during snake drafts.

Key Features:
- Tier-based approach with multiple options per pick (10+ options)
- Team construction optimization using Monte Carlo and Bayesian projections
- Uncertainty-aware decision making with confidence intervals
- Position scarcity management and risk-adjusted decisions
- Pre-generated strategy for practical draft use
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ..utils.interface_standards import handle_exception, setup_logger


@dataclass
class DraftConfig:
    """Configuration for draft strategy."""
    league_size: int
    draft_position: int
    scoring_type: str  # 'PPR', 'Standard', 'Half-PPR'
    roster_positions: Dict[str, int]
    risk_tolerance: str = 'medium'  # 'low', 'medium', 'high'
    
    def __post_init__(self):
        """Validate draft configuration."""
        if self.league_size <= 0 or self.league_size > 20:
            raise ValueError(f"Invalid league size: {self.league_size}")
        
        if self.draft_position <= 0 or self.draft_position > self.league_size:
            raise ValueError(f"Invalid draft position: {self.draft_position}")
        
        if self.scoring_type not in ['PPR', 'Standard', 'Half-PPR']:
            raise ValueError(f"Invalid scoring type: {self.scoring_type}")
        
        if self.risk_tolerance not in ['low', 'medium', 'high']:
            raise ValueError(f"Invalid risk tolerance: {self.risk_tolerance}")


class TierBasedStrategy:
    """Generate tier-based draft strategy with multiple options per pick."""
    
    def __init__(self, predictions: pd.DataFrame):
        """
        Initialize tier-based strategy.
        
        Args:
            predictions: DataFrame with player predictions including:
                - player_name: Player name
                - position: Player position
                - predicted_points: Predicted fantasy points
                - uncertainty_score: Uncertainty measure (0-1)
        """
        self.predictions = predictions.copy()
        self.logger = setup_logger(__name__)
        
        # Validate predictions
        required_columns = ['player_name', 'position', 'predicted_points', 'uncertainty_score']
        missing_columns = [col for col in required_columns if col not in self.predictions.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def create_tiers(self, num_tiers: int = 5) -> Dict[str, List[str]]:
        """
        Create player tiers based on predicted points and uncertainty.
        
        Args:
            num_tiers: Number of tiers to create
            
        Returns:
            Dictionary mapping tier names to lists of player names
        """
        # Sort players by predicted points (descending)
        sorted_players = self.predictions.sort_values('predicted_points', ascending=False)
        
        # Calculate tier boundaries
        players_per_tier = len(sorted_players) // num_tiers
        
        tiers = {}
        for i in range(num_tiers):
            start_idx = i * players_per_tier
            end_idx = start_idx + players_per_tier if i < num_tiers - 1 else len(sorted_players)
            
            tier_name = f"Tier {i + 1}"
            tier_players = sorted_players.iloc[start_idx:end_idx]['player_name'].tolist()
            tiers[tier_name] = tier_players
        
        self.logger.info(f"Created {num_tiers} tiers with {len(sorted_players)} players")
        return tiers
    
    def generate_pick_options(self, draft_position: int, league_size: int, 
                            config: DraftConfig) -> Dict[str, Any]:
        """
        Generate multiple options for a specific pick with uncertainty awareness.
        
        Args:
            draft_position: Draft position (1-based)
            league_size: Number of teams in league
            config: Draft configuration
            
        Returns:
            Dictionary with pick options and strategy including uncertainty analysis
        """
        tiers = self.create_tiers()
        
        # Calculate which picks this team has
        team_picks = self._calculate_team_picks(draft_position, league_size)
        
        # Generate options for current pick
        pick_options = {
            'primary_targets': [],
            'backup_options': [],
            'fallback_options': [],
            'position_priority': '',
            'reasoning': '',
            'uncertainty_analysis': {},
            'confidence_intervals': {}
        }
        
        # Get available players (not yet drafted)
        available_players = self._get_available_players(team_picks, draft_position)
        
        if not available_players.empty:
            # Apply uncertainty-aware selection based on risk tolerance
            pick_options = self._apply_uncertainty_aware_selection(
                available_players, draft_position, config, pick_options
            )
            
            # Determine position priority based on scarcity
            pick_options['position_priority'] = self._determine_position_priority(
                available_players, config
            )
            
            # Generate reasoning with uncertainty considerations
            pick_options['reasoning'] = self._generate_uncertainty_aware_reasoning(
                available_players, draft_position, config, pick_options
            )
            
            # Add confidence intervals for selected players
            pick_options['confidence_intervals'] = self._calculate_confidence_intervals(
                available_players, pick_options
            )
        
        return pick_options
    
    def _apply_uncertainty_aware_selection(self, available_players: pd.DataFrame, 
                                         draft_position: int, config: DraftConfig,
                                         pick_options: Dict[str, Any]) -> Dict[str, Any]:
        """Apply uncertainty-aware player selection based on risk tolerance."""
        risk_tolerance = config.risk_tolerance
        
        if risk_tolerance == 'low':
            # Conservative: Prefer players with lower uncertainty
            sorted_players = available_players.sort_values('uncertainty_score', ascending=True)
        elif risk_tolerance == 'high':
            # Aggressive: Prefer players with higher upside despite uncertainty
            # Sort by predicted points but weight by uncertainty
            available_players['risk_adjusted_score'] = (
                available_players['predicted_points'] * 
                (1 + available_players['uncertainty_score'] * 0.3)
            )
            sorted_players = available_players.sort_values('risk_adjusted_score', ascending=False)
        else:  # medium
            # Balanced: Standard sorting by predicted points
            sorted_players = available_players.sort_values('predicted_points', ascending=False)
        
        # Select players based on risk-adjusted ranking
        pick_options['primary_targets'] = sorted_players.head(3)['player_name'].tolist()
        pick_options['backup_options'] = sorted_players.iloc[3:7]['player_name'].tolist()
        pick_options['fallback_options'] = sorted_players.iloc[7:10]['player_name'].tolist()
        
        # Add uncertainty analysis
        pick_options['uncertainty_analysis'] = {
            'risk_tolerance': risk_tolerance,
            'primary_avg_uncertainty': sorted_players.head(3)['uncertainty_score'].mean(),
            'backup_avg_uncertainty': sorted_players.iloc[3:7]['uncertainty_score'].mean(),
            'fallback_avg_uncertainty': sorted_players.iloc[7:10]['uncertainty_score'].mean(),
            'overall_uncertainty': sorted_players.head(10)['uncertainty_score'].mean()
        }
        
        return pick_options
    
    def _generate_uncertainty_aware_reasoning(self, available_players: pd.DataFrame,
                                            draft_position: int, config: DraftConfig,
                                            pick_options: Dict[str, Any]) -> str:
        """Generate reasoning that considers uncertainty and risk tolerance."""
        reasoning_parts = []
        
        # Add position scarcity reasoning
        top_player = available_players.iloc[0]
        top_position = top_player['position']
        position_counts = available_players['position'].value_counts()
        
        if top_position in position_counts:
            count = position_counts[top_position]
            if count <= 3:
                reasoning_parts.append(f"{top_position} scarcity high")
            elif count <= 8:
                reasoning_parts.append(f"{top_position} moderately scarce")
            else:
                reasoning_parts.append(f"{top_position} depth available")
        
        # Add draft position reasoning
        if draft_position <= 3:
            reasoning_parts.append("early pick - target elite players")
        elif draft_position <= 6:
            reasoning_parts.append("mid-early pick - balance elite and depth")
        else:
            reasoning_parts.append("later pick - focus on value and depth")
        
        # Add uncertainty reasoning
        uncertainty_analysis = pick_options.get('uncertainty_analysis', {})
        risk_tolerance = config.risk_tolerance
        
        if risk_tolerance == 'low':
            reasoning_parts.append("conservative approach - prioritizing certainty")
        elif risk_tolerance == 'high':
            reasoning_parts.append("aggressive approach - accepting uncertainty for upside")
        else:
            reasoning_parts.append("balanced approach - moderate risk tolerance")
        
        # Add specific uncertainty insights
        if 'overall_uncertainty' in uncertainty_analysis:
            uncertainty = uncertainty_analysis['overall_uncertainty']
            if uncertainty < 0.15:
                reasoning_parts.append("high confidence in projections")
            elif uncertainty < 0.25:
                reasoning_parts.append("moderate confidence in projections")
            else:
                reasoning_parts.append("low confidence - consider alternatives")
        
        return ", ".join(reasoning_parts)
    
    def _calculate_confidence_intervals(self, available_players: pd.DataFrame,
                                      pick_options: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence intervals for selected players."""
        confidence_intervals = {}
        
        for option_type in ['primary_targets', 'backup_options', 'fallback_options']:
            players = pick_options.get(option_type, [])
            if not players:
                continue
            
            player_data = available_players[available_players['player_name'].isin(players)]
            if player_data.empty:
                continue
            
            intervals = {}
            for _, player in player_data.iterrows():
                intervals[player['player_name']] = {
                    'lower_bound': player.get('confidence_interval_lower', player['predicted_points'] * 0.9),
                    'upper_bound': player.get('confidence_interval_upper', player['predicted_points'] * 1.1),
                    'uncertainty_score': player['uncertainty_score']
                }
            
            confidence_intervals[option_type] = intervals
        
        return confidence_intervals
    
    def _calculate_team_picks(self, draft_position: int, league_size: int) -> List[int]:
        """Calculate all picks for a team in snake draft."""
        picks = []
        for round_num in range(1, 17):  # Assume 16 rounds
            if round_num % 2 == 1:  # Odd rounds: forward
                pick = draft_position + (round_num - 1) * league_size
            else:  # Even rounds: backward
                pick = (league_size - draft_position + 1) + (round_num - 1) * league_size
            
            picks.append(pick)
        
        return picks
    
    def _get_available_players(self, team_picks: List[int], current_pick: int) -> pd.DataFrame:
        """Get players available at current pick (simplified simulation)."""
        # For now, return all players (in real implementation, this would
        # simulate which players are already drafted)
        return self.predictions.copy()
    
    def _determine_position_priority(self, available_players: pd.DataFrame, 
                                   config: DraftConfig) -> str:
        """Determine position priority based on scarcity and team needs."""
        # Analyze position scarcity
        position_counts = available_players['position'].value_counts()
        
        # Calculate scarcity scores (lower count = higher scarcity)
        scarcity_scores = {}
        for position in ['QB', 'RB', 'WR', 'TE']:
            if position in position_counts:
                scarcity_scores[position] = 1.0 / position_counts[position]
            else:
                scarcity_scores[position] = 1.0  # High scarcity if no players
        
        # Sort positions by scarcity (highest first)
        sorted_positions = sorted(scarcity_scores.items(), 
                                key=lambda x: x[1], reverse=True)
        
        # Create priority string
        priority_parts = [f"{pos}" for pos, _ in sorted_positions[:3]]
        return " > ".join(priority_parts)
    
    def _generate_reasoning(self, available_players: pd.DataFrame, 
                          draft_position: int, config: DraftConfig) -> str:
        """Generate reasoning for pick strategy."""
        if available_players.empty:
            return "No players available"
        
        # Get top player
        top_player = available_players.iloc[0]
        top_position = top_player['position']
        
        # Analyze position distribution
        position_counts = available_players['position'].value_counts()
        
        reasoning_parts = []
        
        # Add position scarcity reasoning
        if top_position in position_counts:
            count = position_counts[top_position]
            if count <= 3:
                reasoning_parts.append(f"{top_position} scarcity high")
            elif count <= 8:
                reasoning_parts.append(f"{top_position} moderately scarce")
            else:
                reasoning_parts.append(f"{top_position} depth available")
        
        # Add draft position reasoning
        if draft_position <= 3:
            reasoning_parts.append("early pick - target elite players")
        elif draft_position <= 6:
            reasoning_parts.append("mid-early pick - balance elite and depth")
        else:
            reasoning_parts.append("later pick - focus on value and depth")
        
        return ", ".join(reasoning_parts)


class TeamConstructionOptimizer:
    """Optimize team construction using Monte Carlo and Bayesian projections."""
    
    def __init__(self, predictions: pd.DataFrame, 
                 monte_carlo_results: pd.DataFrame,
                 roster_requirements: Dict[str, int]):
        """
        Initialize team construction optimizer.
        
        Args:
            predictions: Player-level predictions
            monte_carlo_results: Team-level Monte Carlo results
            roster_requirements: Required players per position
        """
        self.predictions = predictions.copy()
        self.monte_carlo_results = monte_carlo_results.copy()
        self.roster_requirements = roster_requirements
        self.logger = setup_logger(__name__)
    
    def optimize_team_construction(self, draft_position: int) -> Dict[str, Any]:
        """
        Optimize team construction for given draft position using Monte Carlo and Bayesian projections.
        
        Args:
            draft_position: Draft position (1-based)
            
        Returns:
            Dictionary with optimal team construction and projections
        """
        optimal_team = {}
        team_projection = {}
        
        # For each position, select optimal players
        for position, required_count in self.roster_requirements.items():
            if position == 'FLEX':
                # FLEX can be RB, WR, or TE
                flex_players = self._select_flex_players(required_count)
                optimal_team[position] = flex_players
            else:
                # Select position-specific players
                position_players = self._select_position_players(position, required_count)
                optimal_team[position] = position_players
        
        # Calculate team projections using Monte Carlo insights
        team_projection = self._calculate_team_projection(optimal_team)
        
        # Add Bayesian uncertainty analysis
        uncertainty_analysis = self._analyze_team_uncertainty(optimal_team)
        
        return {
            'optimal_team': optimal_team,
            'team_projection': team_projection,
            'uncertainty_analysis': uncertainty_analysis,
            'draft_position': draft_position
        }
    
    def _calculate_team_projection(self, optimal_team: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate team projections using Monte Carlo insights."""
        total_projected_points = 0
        position_breakdown = {}
        
        for position, players in optimal_team.items():
            if not players:
                continue
                
            # Get player predictions
            player_predictions = self.predictions[
                self.predictions['player_name'].isin(players)
            ]
            
            if len(player_predictions) > 0:
                position_total = player_predictions['predicted_points'].sum()
                position_breakdown[position] = {
                    'players': players,
                    'total_points': position_total,
                    'avg_points_per_player': position_total / len(players)
                }
                total_projected_points += position_total
        
        # Compare with Monte Carlo team projections
        mc_comparison = self._compare_with_monte_carlo(total_projected_points)
        
        return {
            'total_projected_points': total_projected_points,
            'position_breakdown': position_breakdown,
            'monte_carlo_comparison': mc_comparison
        }
    
    def _compare_with_monte_carlo(self, team_points: float) -> Dict[str, Any]:
        """Compare team projection with Monte Carlo results."""
        if self.monte_carlo_results.empty:
            return {'available': False}
        
        # Calculate percentile of team projection vs Monte Carlo distribution
        mc_totals = self.monte_carlo_results['projected_total']
        percentile = (mc_totals < team_points).mean() * 100
        
        return {
            'available': True,
            'team_percentile': percentile,
            'mc_mean': mc_totals.mean(),
            'mc_std': mc_totals.std(),
            'relative_performance': (team_points - mc_totals.mean()) / mc_totals.std()
        }
    
    def _analyze_team_uncertainty(self, optimal_team: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze uncertainty in team construction."""
        team_uncertainty = {}
        overall_uncertainty = 0
        total_players = 0
        
        for position, players in optimal_team.items():
            if not players:
                continue
                
            # Get player uncertainty scores
            player_uncertainty = self.predictions[
                self.predictions['player_name'].isin(players)
            ]['uncertainty_score']
            
            if len(player_uncertainty) > 0:
                position_uncertainty = player_uncertainty.mean()
                team_uncertainty[position] = {
                    'players': players,
                    'avg_uncertainty': position_uncertainty,
                    'max_uncertainty': player_uncertainty.max(),
                    'min_uncertainty': player_uncertainty.min()
                }
                
                overall_uncertainty += position_uncertainty * len(players)
                total_players += len(players)
        
        if total_players > 0:
            overall_uncertainty /= total_players
        
        return {
            'overall_uncertainty': overall_uncertainty,
            'position_uncertainty': team_uncertainty,
            'risk_assessment': self._assess_team_risk(overall_uncertainty)
        }
    
    def _assess_team_risk(self, uncertainty: float) -> str:
        """Assess team risk based on uncertainty."""
        if uncertainty < 0.15:
            return "Low Risk - High confidence in projections"
        elif uncertainty < 0.25:
            return "Medium Risk - Moderate confidence in projections"
        else:
            return "High Risk - Low confidence in projections"
    
    def analyze_position_scarcity(self) -> Dict[str, Any]:
        """Analyze position scarcity using hierarchical model insights."""
        scarcity_analysis = {}
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            position_players = self.predictions[self.predictions['position'] == position]
            
            if len(position_players) > 0:
                # Basic scarcity metrics
                total_players = len(position_players)
                avg_points = position_players['predicted_points'].mean()
                std_points = position_players['predicted_points'].std()
                
                # Hierarchical model insights
                position_insights = self._analyze_position_hierarchical_insights(
                    position_players, position
                )
                
                # Calculate comprehensive scarcity score
                scarcity_score = self._calculate_comprehensive_scarcity_score(
                    total_players, avg_points, std_points, position_insights
                )
                
                scarcity_analysis[position] = {
                    'total_players': total_players,
                    'avg_points': avg_points,
                    'std_points': std_points,
                    'scarcity_score': scarcity_score,
                    'hierarchical_insights': position_insights,
                    'quality_distribution': self._analyze_quality_distribution(position_players),
                    'uncertainty_profile': self._analyze_position_uncertainty(position_players)
                }
            else:
                scarcity_analysis[position] = {
                    'total_players': 0,
                    'avg_points': 0.0,
                    'std_points': 0.0,
                    'scarcity_score': 0.0,
                    'hierarchical_insights': {},
                    'quality_distribution': {},
                    'uncertainty_profile': {}
                }
        
        return scarcity_analysis
    
    def _analyze_position_hierarchical_insights(self, position_players: pd.DataFrame, 
                                              position: str) -> Dict[str, Any]:
        """Analyze hierarchical model insights for a position."""
        insights = {}
        
        # Analyze tier distribution
        top_10_percentile = position_players['predicted_points'].quantile(0.9)
        top_25_percentile = position_players['predicted_points'].quantile(0.75)
        median = position_players['predicted_points'].quantile(0.5)
        
        insights['tier_distribution'] = {
            'elite_threshold': top_10_percentile,
            'tier1_threshold': top_25_percentile,
            'median_threshold': median,
            'elite_count': len(position_players[position_players['predicted_points'] >= top_10_percentile]),
            'tier1_count': len(position_players[position_players['predicted_points'] >= top_25_percentile]),
            'above_median_count': len(position_players[position_players['predicted_points'] >= median])
        }
        
        # Analyze uncertainty patterns
        insights['uncertainty_patterns'] = {
            'avg_uncertainty': position_players['uncertainty_score'].mean(),
            'uncertainty_variance': position_players['uncertainty_score'].var(),
            'high_uncertainty_count': len(position_players[position_players['uncertainty_score'] > 0.25]),
            'low_uncertainty_count': len(position_players[position_players['uncertainty_score'] < 0.15])
        }
        
        # Position-specific insights
        if position == 'RB':
            insights['position_specific'] = self._analyze_rb_specific_insights(position_players)
        elif position == 'WR':
            insights['position_specific'] = self._analyze_wr_specific_insights(position_players)
        elif position == 'QB':
            insights['position_specific'] = self._analyze_qb_specific_insights(position_players)
        elif position == 'TE':
            insights['position_specific'] = self._analyze_te_specific_insights(position_players)
        
        return insights
    
    def _analyze_rb_specific_insights(self, rb_players: pd.DataFrame) -> Dict[str, Any]:
        """Analyze RB-specific hierarchical insights."""
        return {
            'workload_dependent': len(rb_players) > 0,  # RBs are highly workload dependent
            'injury_risk_factor': 0.8,  # High injury risk for RBs
            'scarcity_multiplier': 1.2  # RBs are typically more scarce
        }
    
    def _analyze_wr_specific_insights(self, wr_players: pd.DataFrame) -> Dict[str, Any]:
        """Analyze WR-specific hierarchical insights."""
        return {
            'target_dependent': len(wr_players) > 0,  # WRs are target dependent
            'injury_risk_factor': 0.6,  # Moderate injury risk for WRs
            'scarcity_multiplier': 1.0  # Standard scarcity for WRs
        }
    
    def _analyze_qb_specific_insights(self, qb_players: pd.DataFrame) -> Dict[str, Any]:
        """Analyze QB-specific hierarchical insights."""
        return {
            'system_dependent': len(qb_players) > 0,  # QBs are system dependent
            'injury_risk_factor': 0.4,  # Lower injury risk for QBs
            'scarcity_multiplier': 0.8  # QBs are less scarce (only need 1)
        }
    
    def _analyze_te_specific_insights(self, te_players: pd.DataFrame) -> Dict[str, Any]:
        """Analyze TE-specific hierarchical insights."""
        return {
            'scheme_dependent': len(te_players) > 0,  # TEs are scheme dependent
            'injury_risk_factor': 0.7,  # Moderate-high injury risk for TEs
            'scarcity_multiplier': 1.5  # TEs are very scarce
        }
    
    def _calculate_comprehensive_scarcity_score(self, total_players: int, avg_points: float,
                                              std_points: float, insights: Dict[str, Any]) -> float:
        """Calculate comprehensive scarcity score using hierarchical insights."""
        # Base scarcity (fewer players = higher scarcity)
        base_scarcity = 1.0 / max(total_players, 1)
        
        # Quality scarcity (higher average points = higher scarcity)
        quality_scarcity = avg_points / 300.0  # Normalize to typical max points
        
        # Variance scarcity (higher variance = more elite players = higher scarcity)
        variance_scarcity = std_points / 100.0  # Normalize variance
        
        # Position-specific multiplier
        position_multiplier = insights.get('position_specific', {}).get('scarcity_multiplier', 1.0)
        
        # Tier distribution impact
        tier_dist = insights.get('tier_distribution', {})
        elite_ratio = tier_dist.get('elite_count', 0) / max(total_players, 1)
        tier_scarcity = 1.0 - elite_ratio  # Fewer elite players = higher scarcity
        
        # Combine all factors
        comprehensive_score = (
            base_scarcity * 0.3 +
            quality_scarcity * 0.25 +
            variance_scarcity * 0.2 +
            tier_scarcity * 0.25
        ) * position_multiplier
        
        return comprehensive_score
    
    def _analyze_quality_distribution(self, position_players: pd.DataFrame) -> Dict[str, Any]:
        """Analyze quality distribution within a position."""
        if len(position_players) == 0:
            return {}
        
        points = position_players['predicted_points']
        
        return {
            'min_points': points.min(),
            'max_points': points.max(),
            'q1': points.quantile(0.25),
            'q3': points.quantile(0.75),
            'iqr': points.quantile(0.75) - points.quantile(0.25),
            'skewness': points.skew(),
            'kurtosis': points.kurtosis()
        }
    
    def _analyze_position_uncertainty(self, position_players: pd.DataFrame) -> Dict[str, Any]:
        """Analyze uncertainty profile for a position."""
        if len(position_players) == 0:
            return {}
        
        uncertainty = position_players['uncertainty_score']
        
        return {
            'mean_uncertainty': uncertainty.mean(),
            'std_uncertainty': uncertainty.std(),
            'min_uncertainty': uncertainty.min(),
            'max_uncertainty': uncertainty.max(),
            'high_uncertainty_ratio': (uncertainty > 0.25).mean(),
            'low_uncertainty_ratio': (uncertainty < 0.15).mean()
        }
    
    def _select_position_players(self, position: str, count: int) -> List[str]:
        """Select optimal players for a specific position."""
        position_players = self.predictions[self.predictions['position'] == position]
        
        if len(position_players) == 0:
            return []
        
        # Sort by predicted points and select top players
        sorted_players = position_players.sort_values('predicted_points', ascending=False)
        selected_players = sorted_players.head(count)['player_name'].tolist()
        
        return selected_players
    
    def _select_flex_players(self, count: int) -> List[str]:
        """Select optimal FLEX players (RB, WR, TE)."""
        flex_positions = ['RB', 'WR', 'TE']
        flex_players = self.predictions[self.predictions['position'].isin(flex_positions)]
        
        if len(flex_players) == 0:
            return []
        
        # Sort by predicted points and select top players
        sorted_players = flex_players.sort_values('predicted_points', ascending=False)
        selected_players = sorted_players.head(count)['player_name'].tolist()
        
        return selected_players


class UncertaintyAwareSelector:
    """Select players based on uncertainty and risk tolerance."""
    
    def __init__(self, predictions: pd.DataFrame):
        """
        Initialize uncertainty-aware selector.
        
        Args:
            predictions: Player predictions with uncertainty scores
        """
        self.predictions = predictions.copy()
        self.logger = setup_logger(__name__)
    
    def select_players(self, position: str, count: int, 
                      risk_tolerance: str = 'medium') -> List[str]:
        """
        Select players based on position and risk tolerance.
        
        Args:
            position: Player position
            count: Number of players to select
            risk_tolerance: Risk tolerance level ('low', 'medium', 'high')
            
        Returns:
            List of selected player names
        """
        position_players = self.predictions[self.predictions['position'] == position]
        
        if len(position_players) == 0:
            return []
        
        # Apply risk tolerance adjustment
        adjusted_scores = self._apply_risk_adjustment(position_players, risk_tolerance)
        
        # Sort by adjusted scores and select top players
        sorted_players = adjusted_scores.sort_values('adjusted_score', ascending=False)
        selected_players = sorted_players.head(count)['player_name'].tolist()
        
        return selected_players
    
    def calculate_uncertainty_metrics(self) -> Dict[str, float]:
        """Calculate overall uncertainty metrics."""
        metrics = {}
        
        # Overall uncertainty
        metrics['overall_uncertainty'] = self.predictions['uncertainty_score'].mean()
        
        # Position-specific uncertainty
        position_uncertainty = {}
        for position in ['QB', 'RB', 'WR', 'TE']:
            position_players = self.predictions[self.predictions['position'] == position]
            if len(position_players) > 0:
                position_uncertainty[position] = position_players['uncertainty_score'].mean()
            else:
                position_uncertainty[position] = 0.0
        
        metrics['position_uncertainty'] = position_uncertainty
        
        return metrics
    
    def _apply_risk_adjustment(self, players: pd.DataFrame, 
                              risk_tolerance: str) -> pd.DataFrame:
        """Apply risk tolerance adjustment to player scores."""
        adjusted_players = players.copy()
        
        if risk_tolerance == 'low':
            # Conservative: Prefer lower uncertainty
            adjusted_players['adjusted_score'] = (
                players['predicted_points'] * (1 - players['uncertainty_score'])
            )
        elif risk_tolerance == 'high':
            # Aggressive: Prefer higher upside despite uncertainty
            adjusted_players['adjusted_score'] = (
                players['predicted_points'] * (1 + players['uncertainty_score'] * 0.5)
            )
        else:  # medium
            # Balanced: Standard scoring
            adjusted_players['adjusted_score'] = players['predicted_points']
        
        return adjusted_players


class BayesianDraftStrategy:
    """Main Bayesian draft strategy class."""
    
    def __init__(self, bayesian_predictions: pd.DataFrame,
                 monte_carlo_results: pd.DataFrame,
                 draft_config: DraftConfig):
        """
        Initialize Bayesian draft strategy.
        
        Args:
            bayesian_predictions: Player-level Bayesian predictions
            monte_carlo_results: Team-level Monte Carlo results
            draft_config: Draft configuration
        """
        self.bayesian_predictions = bayesian_predictions.copy()
        self.monte_carlo_results = monte_carlo_results.copy()
        self.draft_config = draft_config
        self.logger = setup_logger(__name__)
        
        # Initialize components
        self.tier_strategy = TierBasedStrategy(bayesian_predictions)
        self.team_optimizer = TeamConstructionOptimizer(
            bayesian_predictions, monte_carlo_results, draft_config.roster_positions
        )
        self.uncertainty_selector = UncertaintyAwareSelector(bayesian_predictions)
        
        # Extract config values
        self.league_size = draft_config.league_size
        self.draft_position = draft_config.draft_position
        self.scoring_type = draft_config.scoring_type
        self.risk_tolerance = draft_config.risk_tolerance
    
    def generate_draft_strategy(self) -> Dict[str, Any]:
        """
        Generate complete draft strategy for all picks.
        
        Returns:
            Dictionary with strategy for each pick
        """
        self.logger.info(f"Generating draft strategy for position {self.draft_position}")
        
        # Calculate all picks for this team
        team_picks = self._calculate_team_picks()
        
        # Generate strategy for each pick
        draft_strategy = {}
        for pick_num in team_picks:
            pick_key = f"Pick {pick_num}"
            
            try:
                pick_strategy = self.tier_strategy.generate_pick_options(
                    pick_num, self.league_size, self.draft_config
                )
                draft_strategy[pick_key] = pick_strategy
                
                self.logger.debug(f"Generated strategy for {pick_key}")
                
            except Exception as e:
                error_msg = handle_exception(e, f"Error generating strategy for {pick_key}")
                self.logger.error(error_msg)
                draft_strategy[pick_key] = {
                    'primary_targets': [],
                    'backup_options': [],
                    'fallback_options': [],
                    'position_priority': 'Unknown',
                    'reasoning': f'Error: {str(e)}'
                }
        
        self.logger.info(f"Generated strategy for {len(draft_strategy)} picks")
        return draft_strategy
    
    def generate_complete_draft_strategy(self) -> Dict[str, Any]:
        """
        Generate complete draft strategy with additional metadata.
        
        Returns:
            Complete draft strategy with metadata
        """
        base_strategy = self.generate_draft_strategy()
        
        # Add metadata
        complete_strategy = {
            'strategy': base_strategy,
            'metadata': {
                'draft_position': self.draft_position,
                'league_size': self.league_size,
                'scoring_type': self.scoring_type,
                'risk_tolerance': self.risk_tolerance,
                'generation_timestamp': pd.Timestamp.now().isoformat(),
                'uncertainty_metrics': self.uncertainty_selector.calculate_uncertainty_metrics(),
                'position_scarcity': self.team_optimizer.analyze_position_scarcity()
            }
        }
        
        return complete_strategy
    
    def validate_strategy(self, draft_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate generated draft strategy.
        
        Args:
            draft_strategy: Generated draft strategy
            
        Returns:
            Validation results
        """
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'recommendations': []
        }
        
        # Check that strategy has required structure
        if not isinstance(draft_strategy, dict):
            validation_result['is_valid'] = False
            validation_result['warnings'].append("Strategy is not a dictionary")
            return validation_result
        
        # Check each pick
        for pick_key, pick_strategy in draft_strategy.items():
            if not isinstance(pick_strategy, dict):
                validation_result['warnings'].append(f"{pick_key}: Pick strategy is not a dictionary")
                continue
            
            # Check required fields
            required_fields = ['primary_targets', 'backup_options', 'fallback_options']
            for field in required_fields:
                if field not in pick_strategy:
                    validation_result['warnings'].append(f"{pick_key}: Missing field '{field}'")
                elif not isinstance(pick_strategy[field], list):
                    validation_result['warnings'].append(f"{pick_key}: Field '{field}' is not a list")
            
            # Check that we have multiple options
            total_options = sum(len(pick_strategy.get(field, [])) for field in required_fields)
            if total_options < 10:
                validation_result['recommendations'].append(
                    f"{pick_key}: Consider adding more options (currently {total_options})"
                )
        
        return validation_result
    
    def _calculate_team_picks(self) -> List[int]:
        """Calculate all picks for this team in snake draft."""
        picks = []
        for round_num in range(1, 17):  # Assume 16 rounds
            if round_num % 2 == 1:  # Odd rounds: forward
                pick = self.draft_position + (round_num - 1) * self.league_size
            else:  # Even rounds: backward
                pick = (self.league_size - self.draft_position + 1) + (round_num - 1) * self.league_size
            
            picks.append(pick)
        
        return picks


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Bayesian draft strategy")
    parser.add_argument("--draft-position", type=int, required=True,
                       help="Draft position (1-based)")
    parser.add_argument("--league-size", type=int, default=12,
                       help="League size (default: 12)")
    parser.add_argument("--scoring-type", default="PPR",
                       choices=["PPR", "Standard", "Half-PPR"],
                       help="Scoring type (default: PPR)")
    parser.add_argument("--risk-tolerance", default="medium",
                       choices=["low", "medium", "high"],
                       help="Risk tolerance (default: medium)")
    parser.add_argument("--output-file", type=str,
                       help="Output file for strategy (JSON)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(__name__)
    
    try:
        # Load actual predictions from Bayesian and Monte Carlo results
        logger.info("Loading predictions...")
        
        # Load Bayesian predictions
        bayesian_results_file = 'results/bayesian-hierarchical-results/modern_model_results.json'
        if not os.path.exists(bayesian_results_file):
            logger.error(f"Bayesian results not found: {bayesian_results_file}")
            logger.error("Please run 'ffbayes-bayes' first to generate predictions")
            return 1
        
        # Load actual player data with positions
        logger.info("Loading player data with positions...")
        
        # Find the most recent 5-year combined dataset
        combined_datasets_dir = Path('datasets/combined_datasets')
        if not combined_datasets_dir.exists():
            logger.error("Combined datasets directory not found")
            logger.error("Please run data collection first")
            return 1
        
        # Look for 5-year interval datasets (e.g., 2019-2024, 2018-2023, etc.)
        dataset_files = list(combined_datasets_dir.glob('*season_modern.csv'))
        
        if not dataset_files:
            logger.error("No combined datasets found")
            logger.error("Please run data collection first")
            return 1
        
        # Sort by modification time to get the most recent
        dataset_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        player_data_file = str(dataset_files[0])
        
        logger.info(f"Using most recent dataset: {player_data_file}")
        
        if not os.path.exists(player_data_file):
            logger.error(f"Player data not found: {player_data_file}")
            logger.error("Please run data collection first")
            return 1
        
        # Load player data
        player_data = pd.read_csv(player_data_file)
        logger.info(f"Loaded player data: {player_data.shape}")
        
        # Get unique players with their positions and calculate season averages
        current_season_data = player_data[player_data['Season'] == player_data['Season'].max()]
        
        # Calculate player projections from historical data
        player_projections = current_season_data.groupby(['Name', 'Position']).agg({
            'FantPtPPR': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        player_projections.columns = ['player_name', 'position', 'mean_points', 'std_points', 'games_played']
        
        # Calculate uncertainty scores and confidence intervals
        player_stats = []
        for _, row in player_projections.iterrows():
            mean_points = row['mean_points'] * 17  # Project to full season (17 games)
            std_points = row['std_points'] * (17 ** 0.5) if row['std_points'] > 0 else mean_points * 0.15
            
            # Calculate confidence intervals
            confidence_lower = mean_points - 1.96 * std_points
            confidence_upper = mean_points + 1.96 * std_points
            
            # Calculate uncertainty score (coefficient of variation)
            uncertainty_score = std_points / mean_points if mean_points > 0 else 0.5
            uncertainty_score = min(max(uncertainty_score, 0.05), 0.5)
            
            # Only include players with reasonable projections and enough games
            if mean_points > 50 and row['games_played'] >= 3:
                player_stats.append({
                    'player_name': row['player_name'],
                    'position': row['position'],
                    'predicted_points': mean_points,
                    'confidence_interval_lower': max(0, confidence_lower),
                    'confidence_interval_upper': confidence_upper,
                    'uncertainty_score': uncertainty_score
                })
        
        bayesian_predictions = pd.DataFrame(player_stats)
        logger.info(f"Processed {len(bayesian_predictions)} players with valid projections")
        logger.info(f"Position breakdown: {bayesian_predictions['position'].value_counts().to_dict()}")
        
        # Generate team projections from Monte Carlo data
        logger.info("Loading Monte Carlo team projections...")
        
        # Find the most recent Monte Carlo results for team projections
        mc_results_dir = Path('results/montecarlo_results')
        if not mc_results_dir.exists():
            logger.warning("Monte Carlo results directory not found, using simplified team projections")
            # Create simplified team projections based on player data
            team_totals = pd.Series([bayesian_predictions['predicted_points'].sum()] * 100)
        else:
            # Look for Monte Carlo projection files
            mc_files = list(mc_results_dir.glob('*projections*.tsv'))
            
            if not mc_files:
                logger.warning("No Monte Carlo projection files found, using simplified team projections")
                # Create simplified team projections based on player data
                team_totals = pd.Series([bayesian_predictions['predicted_points'].sum()] * 100)
            else:
                # Sort by modification time to get the most recent
                mc_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                mc_file = mc_files[0]
                logger.info(f"Using most recent Monte Carlo results: {mc_file}")
                
                mc_data = pd.read_csv(mc_file, sep='\t', index_col=0)
                team_totals = mc_data['Total']
        
        # Create team projection data
        monte_carlo_data = []
        for i, total in enumerate(team_totals):
            # Calculate confidence intervals based on the distribution
            std_dev = team_totals.std()
            lower_bound = total - 1.96 * std_dev  # 95% CI
            upper_bound = total + 1.96 * std_dev
            
            monte_carlo_data.append({
                'team_id': i + 1,
                'projected_total': total,
                'confidence_interval_lower': max(0, lower_bound),  # Don't allow negative
                'confidence_interval_upper': upper_bound
            })
        
        monte_carlo_results = pd.DataFrame(monte_carlo_data)
        logger.info(f"Generated {len(monte_carlo_results)} team projections from Monte Carlo data")
        
        # Create draft config
        draft_config = DraftConfig(
            league_size=args.league_size,
            draft_position=args.draft_position,
            scoring_type=args.scoring_type,
            risk_tolerance=args.risk_tolerance,
            roster_positions={
                'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1, 'DST': 1, 'K': 1
            }
        )
        
        # Generate strategy
        logger.info("Generating draft strategy...")
        strategy = BayesianDraftStrategy(
            bayesian_predictions=bayesian_predictions,
            monte_carlo_results=monte_carlo_results,
            draft_config=draft_config
        )
        
        draft_strategy = strategy.generate_complete_draft_strategy()
        
        # Validate strategy
        validation_result = strategy.validate_strategy(draft_strategy['strategy'])
        
        # Output results
        if args.output_file:
            # Ensure output directory exists
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(draft_strategy, f, indent=2)
            logger.info(f"Strategy saved to {output_path}")
        else:
            # Default to results/draft_strategy/ directory
            output_dir = Path('results/draft_strategy')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            default_filename = f'draft_strategy_pos{args.draft_position}_{timestamp}.json'
            default_path = output_dir / default_filename
            
            with open(default_path, 'w') as f:
                json.dump(draft_strategy, f, indent=2)
            logger.info(f"Strategy saved to {default_path}")
            
            # Removed verbose console dump to avoid flooding logs
            # Use --verbose if detailed output is ever needed
            # print(json.dumps(draft_strategy, indent=2))
        
        # Report validation results
        if validation_result['warnings']:
            logger.warning(f"Validation warnings: {validation_result['warnings']}")
        
        if validation_result['recommendations']:
            logger.info(f"Recommendations: {validation_result['recommendations']}")
        
        logger.info("Draft strategy generation completed successfully")
        
    except Exception as e:
        error_msg = handle_exception(e, "Draft strategy generation failed")
        logger.error(error_msg)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
