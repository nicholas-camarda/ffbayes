"""
Draft Strategy Comparison Framework

This module provides comprehensive comparison functionality between VOR and Bayesian draft strategies,
including side-by-side analysis, performance metrics, and visualization tools.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Production mode by default - test mode must be explicitly enabled
QUICK_TEST = os.getenv('QUICK_TEST', 'false').lower() == 'true'


class DraftStrategyComparison:
    """
    Comprehensive comparison framework for VOR vs Bayesian draft strategies.
    
    This class provides tools to:
    - Load and parse both VOR and Bayesian strategy results
    - Compare player rankings and recommendations
    - Analyze strategy differences and similarities
    - Generate performance metrics and insights
    - Create visualizations for strategy comparison
    """
    
    def __init__(self, vor_file: str = None, bayesian_file: str = None):
        """
        Initialize the comparison framework.
        
        Args:
            vor_file: Path to VOR strategy CSV file
            bayesian_file: Path to Bayesian strategy JSON file
        """
        self.vor_file = vor_file
        if not self.vor_file:
            # CRITICAL: Configurable VOR file path - no hardcoding
            # Priority: 1. Environment variable, 2. Default based on QUICK_TEST
            self.vor_file = os.getenv('VOR_FILE')
            if not self.vor_file:
                if QUICK_TEST:
                    from ffbayes.utils.path_constants import \
                        get_pre_draft_plots_dir
                    self.vor_file = str(get_pre_draft_plots_dir(datetime.now().year) / 'vor_strategy' / f'test_vor_{datetime.now().year}.csv')
                    print(f"   QUICK_TEST mode - using test VOR file: {self.vor_file}")
                else:
                    # Production mode - try to find VOR file in organized pre_draft structure
                    current_year = datetime.now().year
                    from ffbayes.utils.vor_filename_generator import \
                        get_vor_strategy_path
                    vor_file = get_vor_strategy_path(current_year)
                    if os.path.exists(vor_file):
                        self.vor_file = vor_file
                        print(f"   Production mode - found VOR file: {self.vor_file}")
                    else:
                        raise ValueError(
                            "VOR file path must be specified in production mode. "
                            "Set VOR_FILE environment variable or pass vor_file parameter. "
                            "No hardcoded paths allowed."
                        )
        self.bayesian_file = bayesian_file or self._find_latest_bayesian_file()
        
        self.vor_data = None
        self.bayesian_data = None
        self.comparison_results = {}
        
        # Load data
        self._load_data()
    
    def _find_latest_bayesian_file(self) -> str:
        """Find the most recent Bayesian strategy file."""
        current_year = datetime.now().year
        from ffbayes.utils.path_constants import get_draft_strategy_dir
        results_dir = get_draft_strategy_dir(current_year)
        if not results_dir.exists():
            return None
        
        json_files = list(results_dir.glob("draft_strategy_pos*.json"))
        if not json_files:
            return None
        
        # Return the most recent file
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        return str(latest_file)
    
    def _load_data(self):
        """Load VOR and Bayesian strategy data."""
        try:
            # Load VOR data
            if os.path.exists(self.vor_file):
                if self.vor_file.endswith('.xlsx'):
                    self.vor_data = pd.read_excel(self.vor_file)
                else:
                    self.vor_data = pd.read_csv(self.vor_file)
                logger.info(f"Loaded VOR data: {len(self.vor_data)} players")
            else:
                logger.warning(f"VOR file not found: {self.vor_file}")
            
            # Load Bayesian data
            if self.bayesian_file and os.path.exists(self.bayesian_file):
                with open(self.bayesian_file, 'r') as f:
                    self.bayesian_data = json.load(f)
                logger.info(f"Loaded Bayesian data from: {self.bayesian_file}")
            else:
                logger.warning(f"Bayesian file not found: {self.bayesian_file}")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def compare_player_rankings(self) -> Dict[str, Any]:
        """
        Compare player rankings between VOR and Bayesian strategies.
        
        Returns:
            Dictionary containing comparison metrics and insights
        """
        if self.vor_data is None or self.bayesian_data is None:
            return {"error": "Both VOR and Bayesian data required for comparison"}
        
        # Extract player rankings from both strategies
        vor_players = self.vor_data[['PLAYER', 'POS', 'VALUERANK']].copy()
        vor_players.columns = ['player', 'position', 'vor_rank']
        
        # Extract Bayesian rankings (simplified - would need more complex parsing)
        bayesian_players = self._extract_bayesian_rankings()
        
        # Merge data
        comparison_df = pd.merge(
            vor_players, 
            bayesian_players, 
            on='player', 
            how='outer'
        )
        
        # Calculate ranking differences
        comparison_df['rank_difference'] = comparison_df['vor_rank'] - comparison_df['bayesian_rank']
        comparison_df['rank_difference_abs'] = abs(comparison_df['rank_difference'])
        
        # Analysis metrics
        metrics = {
            'total_players_vor': len(vor_players),
            'total_players_bayesian': len(bayesian_players),
            'common_players': len(comparison_df.dropna()),
            'avg_rank_difference': comparison_df['rank_difference_abs'].mean(),
            'max_rank_difference': comparison_df['rank_difference_abs'].max(),
            'players_with_large_differences': len(comparison_df[comparison_df['rank_difference_abs'] > 20])
        }
        
        # Position-specific analysis
        position_analysis = {}
        for pos in ['QB', 'RB', 'WR', 'TE']:
            pos_data = comparison_df[comparison_df['position'] == pos]
            if len(pos_data) > 0:
                position_analysis[pos] = {
                    'count': len(pos_data),
                    'avg_rank_difference': pos_data['rank_difference_abs'].mean(),
                    'max_rank_difference': pos_data['rank_difference_abs'].max()
                }
        
        self.comparison_results['player_rankings'] = {
            'metrics': metrics,
            'position_analysis': position_analysis,
            'data': comparison_df.to_dict('records')
        }
        
        return self.comparison_results['player_rankings']
    
    def _extract_bayesian_rankings(self) -> pd.DataFrame:
        """
        Extract player rankings from Bayesian strategy data.
        
        Returns:
            DataFrame with player rankings
        """
        players = []
        ranks = []
        player_rankings = {}  # Track best rank for each player
        
        # Check both possible structure keys
        strategy_key = None
        if 'draft_strategy' in self.bayesian_data:
            strategy_key = 'draft_strategy'
        elif 'strategy' in self.bayesian_data:
            strategy_key = 'strategy'
        
        logger.info(f"Using strategy key: {strategy_key}")
        
        if strategy_key:
            strategy = self.bayesian_data[strategy_key]
            logger.info(f"Strategy has {len(strategy)} picks")
            
            # Extract pick numbers and sort them
            pick_numbers = []
            for pick_key in strategy.keys():
                if pick_key.startswith('Pick '):
                    try:
                        pick_num = int(pick_key.split(' ')[1])
                        pick_numbers.append(pick_num)
                    except (ValueError, IndexError):
                        continue
            
            pick_numbers.sort()
            logger.info(f"Found pick numbers: {pick_numbers}")
            
            for pick_num in pick_numbers:
                pick_key = f"Pick {pick_num}"
                if pick_key in strategy:
                    pick_data = strategy[pick_key]
                    
                    if isinstance(pick_data, dict):
                        # Extract players from primary, backup, and fallback options
                        for option_type in ['primary_targets', 'backup_options', 'fallback_options']:
                            if option_type in pick_data:
                                option_players = pick_data[option_type]
                                logger.info(f"Pick {pick_num} {option_type}: {option_players}")
                                for player in option_players:
                                    # Use the earliest (best) rank for each player
                                    if player not in player_rankings:
                                        player_rankings[player] = pick_num
        
        logger.info(f"Extracted {len(player_rankings)} unique players: {list(player_rankings.keys())}")
        
        # Convert to DataFrame
        for player, rank in player_rankings.items():
            players.append(player)
            ranks.append(rank)
        
        return pd.DataFrame({
            'player': players,
            'bayesian_rank': ranks
        })
    
    def compare_strategy_approaches(self) -> Dict[str, Any]:
        """
        Compare the fundamental approaches of VOR vs Bayesian strategies.
        
        Returns:
            Dictionary containing strategy comparison analysis
        """
        comparison = {
            'vor_approach': {
                'type': 'Static Rankings',
                'data_source': 'FantasyPros',
                'methodology': 'Value Over Replacement',
                'uncertainty': 'None',
                'position_scarcity': 'Basic',
                'team_construction': 'Individual rankings',
                'execution_speed': 'Fast (~30 seconds)',
                'dependencies': 'None'
            },
            'bayesian_approach': {
                'type': 'Dynamic Strategy',
                'data_source': 'Bayesian model predictions',
                'methodology': 'Tier-based with uncertainty',
                'uncertainty': 'Full quantification',
                'position_scarcity': 'Advanced analysis',
                'team_construction': 'Team optimization',
                'execution_speed': 'Moderate (~2-3 minutes)',
                'dependencies': 'Requires Bayesian model'
            },
            'key_differences': [
                'VOR provides static rankings, Bayesian provides dynamic pick-by-pick strategy',
                'VOR has no uncertainty quantification, Bayesian includes full uncertainty analysis',
                'VOR focuses on individual player value, Bayesian optimizes for team construction',
                'VOR is faster and simpler, Bayesian is more sophisticated but slower',
                'VOR uses external data (FantasyPros), Bayesian uses internal model predictions'
            ],
            'complementary_aspects': [
                'VOR provides established, trusted rankings',
                'Bayesian provides advanced statistical insights',
                'VOR is good for quick reference',
                'Bayesian is good for detailed strategy planning',
                'Both can be used together for comprehensive analysis'
            ]
        }
        
        self.comparison_results['strategy_approaches'] = comparison
        return comparison
    
    def generate_performance_metrics(self) -> Dict[str, Any]:
        """
        Generate meaningful fantasy football performance metrics for both strategies.
        
        Returns:
            Dictionary containing performance metrics that matter to drafters
        """
        # Calculate actual fantasy football performance metrics
        vor_metrics = {}
        bayesian_metrics = {}
        
        if self.vor_data is not None:
            # VOR performance metrics
            vor_metrics = {
                'avg_projected_points': self.vor_data.get('FantPt', pd.Series([0])).mean(),
                'top_10_players': len(self.vor_data.head(10)),
                'rb_wr_balance': self._calculate_position_balance(self.vor_data, ['RB', 'WR']),
                'value_consistency': self._calculate_value_consistency(self.vor_data),
                'bust_risk_players': self._identify_bust_risks(self.vor_data),
                'sleeper_opportunities': self._identify_sleepers(self.vor_data)
            }
        
        if self.bayesian_data is not None:
            # Bayesian performance metrics
            bayesian_metrics = {
                'avg_projected_points': self._calculate_bayesian_avg_points(),
                'uncertainty_quantified': self._calculate_uncertainty_coverage(),
                'position_optimization': self._calculate_position_optimization(),
                'team_construction_score': self._calculate_team_construction_score(),
                'risk_adjusted_value': self._calculate_risk_adjusted_value(),
                'draft_position_adaptation': self._calculate_draft_adaptation()
            }
        
        metrics = {
            'vor_performance': vor_metrics,
            'bayesian_performance': bayesian_metrics,
            'comparison_metrics': {
                'projection_accuracy': self._compare_projection_accuracy(),
                'risk_management': self._compare_risk_management(),
                'position_scarcity_handling': self._compare_position_scarcity(),
                'team_construction_effectiveness': self._compare_team_construction()
            }
        }
        
        self.comparison_results['performance_metrics'] = metrics
        return metrics
    
    def _calculate_position_balance(self, data: pd.DataFrame, positions: list) -> float:
        """Calculate balance between specified positions."""
        if 'Position' not in data.columns:
            return 0.0
        pos_counts = data['Position'].value_counts()
        total = sum(pos_counts.get(pos, 0) for pos in positions)
        return total / len(data) if len(data) > 0 else 0.0
    
    def _calculate_value_consistency(self, data: pd.DataFrame) -> float:
        """Calculate consistency of value across players."""
        if 'FantPt' not in data.columns:
            return 0.0
        return 1.0 - (data['FantPt'].std() / data['FantPt'].mean()) if data['FantPt'].mean() > 0 else 0.0
    
    def _identify_bust_risks(self, data: pd.DataFrame) -> int:
        """Identify players with high bust risk."""
        if 'FantPt' not in data.columns:
            return 0
        # Players with low projections but high ADP
        return len(data[(data['FantPt'] < data['FantPt'].quantile(0.25)) & 
                       (data.get('ADP', 999) < data.get('ADP', 999).quantile(0.5))])
    
    def _identify_sleepers(self, data: pd.DataFrame) -> int:
        """Identify potential sleeper players."""
        if 'FantPt' not in data.columns:
            return 0
        # Players with high projections but low ADP
        return len(data[(data['FantPt'] > data['FantPt'].quantile(0.75)) & 
                       (data.get('ADP', 999) > data.get('ADP', 999).quantile(0.5))])
    
    def _calculate_bayesian_avg_points(self) -> float:
        """Calculate average projected points from Bayesian predictions."""
        if not self.bayesian_data or 'strategy' not in self.bayesian_data:
            return 0.0
        
        total_points = 0
        count = 0
        for pick_info in self.bayesian_data['strategy'].values():
            if isinstance(pick_info, dict) and 'selected_player' in pick_info:
                player = pick_info['selected_player']
                if isinstance(player, dict) and 'predicted_points' in player:
                    total_points += player['predicted_points']
                    count += 1
        
        return total_points / count if count > 0 else 0.0
    
    def _calculate_uncertainty_coverage(self) -> float:
        """Calculate how well uncertainty is quantified."""
        if not self.bayesian_data or 'strategy' not in self.bayesian_data:
            return 0.0
        
        uncertainty_count = 0
        total_count = 0
        for pick_info in self.bayesian_data['strategy'].values():
            if isinstance(pick_info, dict) and 'selected_player' in pick_info:
                player = pick_info['selected_player']
                if isinstance(player, dict):
                    total_count += 1
                    if 'uncertainty_score' in player or 'confidence_interval' in player:
                        uncertainty_count += 1
        
        return uncertainty_count / total_count if total_count > 0 else 0.0
    
    def _calculate_position_optimization(self) -> float:
        """Calculate position optimization score."""
        if not self.bayesian_data or 'strategy' not in self.bayesian_data:
            return 0.0
        
        # Count how many positions are covered optimally
        positions_covered = set()
        for pick_info in self.bayesian_data['strategy'].values():
            if isinstance(pick_info, dict) and 'selected_player' in pick_info:
                player = pick_info['selected_player']
                if isinstance(player, dict) and 'position' in player:
                    positions_covered.add(player['position'])
        
        return len(positions_covered) / 4.0  # Assuming 4 main positions
    
    def _calculate_team_construction_score(self) -> float:
        """Calculate team construction effectiveness."""
        if not self.bayesian_data or 'strategy' not in self.bayesian_data:
            return 0.0
        
        # Check if strategy considers team construction
        has_team_consideration = any(
            'team_construction' in str(pick_info) or 'roster_balance' in str(pick_info)
            for pick_info in self.bayesian_data['strategy'].values()
        )
        
        return 1.0 if has_team_consideration else 0.5
    
    def _calculate_risk_adjusted_value(self) -> float:
        """Calculate risk-adjusted value score."""
        if not self.bayesian_data or 'strategy' not in self.bayesian_data:
            return 0.0
        
        risk_adjusted_count = 0
        total_count = 0
        for pick_info in self.bayesian_data['strategy'].values():
            if isinstance(pick_info, dict) and 'selected_player' in pick_info:
                player = pick_info['selected_player']
                if isinstance(player, dict):
                    total_count += 1
                    if 'risk_tolerance' in str(pick_info) or 'uncertainty_score' in player:
                        risk_adjusted_count += 1
        
        return risk_adjusted_count / total_count if total_count > 0 else 0.0
    
    def _calculate_draft_adaptation(self) -> float:
        """Calculate draft position adaptation score."""
        if not self.bayesian_data or 'strategy' not in self.bayesian_data:
            return 0.0
        
        # Check if strategy adapts to draft position
        has_adaptation = any(
            'draft_position' in str(pick_info) or 'pick_number' in str(pick_info)
            for pick_info in self.bayesian_data['strategy'].values()
        )
        
        return 1.0 if has_adaptation else 0.5
    
    def _compare_projection_accuracy(self) -> str:
        """Compare projection accuracy between strategies."""
        vor_avg = self.comparison_results.get('performance_metrics', {}).get('vor_performance', {}).get('avg_projected_points', 0)
        bayesian_avg = self.comparison_results.get('performance_metrics', {}).get('bayesian_performance', {}).get('avg_projected_points', 0)
        
        if vor_avg > bayesian_avg * 1.1:
            return "VOR provides higher projections"
        elif bayesian_avg > vor_avg * 1.1:
            return "Bayesian provides higher projections"
        else:
            return "Similar projection levels"
    
    def _compare_risk_management(self) -> str:
        """Compare risk management between strategies."""
        vor_risk = self.comparison_results.get('performance_metrics', {}).get('vor_performance', {}).get('bust_risk_players', 0)
        bayesian_risk = self.comparison_results.get('performance_metrics', {}).get('bayesian_performance', {}).get('risk_adjusted_value', 0)
        
        if bayesian_risk > 0.7:
            return "Bayesian has better risk management"
        elif vor_risk < 3:
            return "VOR has lower bust risk"
        else:
            return "Similar risk profiles"
    
    def _compare_position_scarcity(self) -> str:
        """Compare position scarcity handling."""
        vor_balance = self.comparison_results.get('performance_metrics', {}).get('vor_performance', {}).get('rb_wr_balance', 0)
        bayesian_balance = self.comparison_results.get('performance_metrics', {}).get('bayesian_performance', {}).get('position_optimization', 0)
        
        if bayesian_balance > 0.8:
            return "Bayesian better handles position scarcity"
        elif vor_balance > 0.6:
            return "VOR provides good position balance"
        else:
            return "Both handle scarcity similarly"
    
    def _compare_team_construction(self) -> str:
        """Compare team construction effectiveness."""
        bayesian_team = self.comparison_results.get('performance_metrics', {}).get('bayesian_performance', {}).get('team_construction_score', 0)
        
        if bayesian_team > 0.8:
            return "Bayesian optimizes team construction"
        else:
            return "VOR focuses on individual value"
    
    def create_comparison_visualizations(self, output_dir: str = None):
        """
        Create visualizations comparing VOR and Bayesian strategies.
        
        Args:
            output_dir: Directory to save visualization files
        """
        if output_dir is None:
            from ffbayes.utils.path_constants import get_pre_draft_plots_dir
            current_year = datetime.now().year
            output_dir = str(get_pre_draft_plots_dir(current_year) / "draft_strategy_comparison")
        """
        Create visualizations comparing VOR and Bayesian strategies.
        
        Args:
            output_dir: Directory to save visualization files
        """
        if not self.comparison_results:
            logger.warning("No comparison results available. Run comparison methods first.")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create meaningful fantasy football comparison dashboard
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fantasy Football Draft Strategy Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Projected Points Comparison (what drafters actually care about)
        if 'performance_metrics' in self.comparison_results:
            self._plot_projected_points_comparison(axes[0, 0])
        
        # 2. Risk Management Comparison (bust risk vs sleeper opportunities)
        if 'performance_metrics' in self.comparison_results:
            self._plot_risk_management_comparison(axes[0, 1])
        
        # 3. Position Balance Analysis (RB/WR balance, position scarcity)
        if 'performance_metrics' in self.comparison_results:
            self._plot_position_balance_comparison(axes[1, 0])
        
        # 4. Strategy Effectiveness Summary (overall performance metrics)
        if 'performance_metrics' in self.comparison_results:
            self._plot_strategy_effectiveness_summary(axes[1, 1])
        
        plt.tight_layout()
        
        # Save visualization with draft year instead of timestamp
        current_year = datetime.now().year
        filename = f"{output_dir}/draft_strategy_comparison_{current_year}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison visualization: {filename}")
        
        return filename
    
    
    def _plot_projected_points_comparison(self, ax):
        """Plot projected points comparison - what drafters actually care about."""
        if 'performance_metrics' not in self.comparison_results:
            ax.text(0.5, 0.5, 'No performance data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Projected Points Comparison')
            return
        
        metrics = self.comparison_results['performance_metrics']
        vor_metrics = metrics.get('vor_performance', {})
        bayesian_metrics = metrics.get('bayesian_performance', {})
        
        # Extract projected points data
        vor_points = vor_metrics.get('avg_projected_points', 0)
        bayesian_points = bayesian_metrics.get('avg_projected_points', 0)
        
        strategies = ['VOR Strategy', 'Bayesian Strategy']
        points = [vor_points, bayesian_points]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(strategies, points, color=colors, alpha=0.8)
        ax.set_ylabel('Average Projected Points')
        ax.set_title('Projected Points Comparison\n(Higher is Better)')
        
        # Add value labels on bars
        for bar, value in zip(bars, points):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Add comparison text
        if vor_points > bayesian_points:
            ax.text(0.5, 0.9, f'VOR projects {vor_points - bayesian_points:.1f} more points', 
                   ha='center', va='top', transform=ax.transAxes, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        else:
            ax.text(0.5, 0.9, f'Bayesian projects {bayesian_points - vor_points:.1f} more points', 
                   ha='center', va='top', transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    def _plot_risk_management_comparison(self, ax):
        """Plot risk management comparison - bust risk vs sleeper opportunities."""
        if 'performance_metrics' not in self.comparison_results:
            ax.text(0.5, 0.5, 'No performance data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Risk Management Comparison')
            return
        
        metrics = self.comparison_results['performance_metrics']
        vor_metrics = metrics.get('vor_performance', {})
        bayesian_metrics = metrics.get('bayesian_performance', {})
        
        # Extract risk management data
        vor_bust_risk = vor_metrics.get('bust_risk_players', 0)
        vor_sleepers = vor_metrics.get('sleeper_opportunities', 0)
        bayesian_risk_score = bayesian_metrics.get('risk_adjusted_value', 0) * 10  # Scale to 0-10
        
        # Create grouped bar chart
        categories = ['Bust Risk\n(Lower is Better)', 'Sleeper Opportunities\n(Higher is Better)', 'Risk Management\n(Higher is Better)']
        vor_values = [vor_bust_risk, vor_sleepers, 5]  # Default risk management score
        bayesian_values = [0, 0, bayesian_risk_score]  # Bayesian doesn't have bust/sleeper counts
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, vor_values, width, label='VOR Strategy', color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, bayesian_values, width, label='Bayesian Strategy', color='#4ECDC4', alpha=0.8)
        
        ax.set_xlabel('Risk Management Metrics')
        ax.set_ylabel('Score/Count')
        ax.set_title('Risk Management Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    def _plot_position_balance_comparison(self, ax):
        """Plot position balance comparison - RB/WR balance and position scarcity."""
        if 'performance_metrics' not in self.comparison_results:
            ax.text(0.5, 0.5, 'No performance data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Position Balance Comparison')
            return
        
        metrics = self.comparison_results['performance_metrics']
        vor_metrics = metrics.get('vor_performance', {})
        bayesian_metrics = metrics.get('bayesian_performance', {})
        
        # Extract position balance data
        vor_balance = vor_metrics.get('rb_wr_balance', 0) * 100  # Convert to percentage
        bayesian_optimization = bayesian_metrics.get('position_optimization', 0) * 100  # Convert to percentage
        
        categories = ['RB/WR Balance\n(VOR)', 'Position Optimization\n(Bayesian)']
        values = [vor_balance, bayesian_optimization]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        ax.set_ylabel('Score (%)')
        ax.set_title('Position Balance & Scarcity Handling\n(Higher is Better)')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add interpretation
        if vor_balance > 60:
            ax.text(0.5, 0.9, 'VOR provides good position balance', 
                   ha='center', va='top', transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        if bayesian_optimization > 80:
            ax.text(0.5, 0.8, 'Bayesian optimizes for position scarcity', 
                   ha='center', va='top', transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    def _plot_strategy_effectiveness_summary(self, ax):
        """Plot overall strategy effectiveness summary."""
        if 'performance_metrics' not in self.comparison_results:
            ax.text(0.5, 0.5, 'No performance data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Strategy Effectiveness Summary')
            return
        
        metrics = self.comparison_results['performance_metrics']
        vor_metrics = metrics.get('vor_performance', {})
        bayesian_metrics = metrics.get('bayesian_performance', {})
        
        # Create effectiveness scores
        vor_effectiveness = 0
        bayesian_effectiveness = 0
        
        # VOR effectiveness based on available metrics
        if 'avg_projected_points' in vor_metrics:
            vor_effectiveness += 30  # Base score for having projections
        if 'value_consistency' in vor_metrics:
            vor_effectiveness += 20  # Consistency bonus
        if 'sleeper_opportunities' in vor_metrics and vor_metrics['sleeper_opportunities'] > 0:
            vor_effectiveness += 25  # Sleeper identification bonus
        
        # Bayesian effectiveness based on available metrics
        if 'avg_projected_points' in bayesian_metrics:
            bayesian_effectiveness += 30  # Base score for having projections
        if 'uncertainty_quantified' in bayesian_metrics:
            bayesian_effectiveness += 25  # Uncertainty quantification bonus
        if 'team_construction_score' in bayesian_metrics:
            bayesian_effectiveness += 20  # Team construction bonus
        if 'draft_position_adaptation' in bayesian_metrics:
            bayesian_effectiveness += 15  # Draft adaptation bonus
        
        strategies = ['VOR Strategy', 'Bayesian Strategy']
        effectiveness = [vor_effectiveness, bayesian_effectiveness]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(strategies, effectiveness, color=colors, alpha=0.8)
        ax.set_ylabel('Effectiveness Score')
        ax.set_title('Overall Strategy Effectiveness\n(Higher is Better)')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, effectiveness):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Add recommendation
        if vor_effectiveness > bayesian_effectiveness:
            ax.text(0.5, 0.9, 'VOR Strategy Recommended', 
                   ha='center', va='top', transform=ax.transAxes, fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        elif bayesian_effectiveness > vor_effectiveness:
            ax.text(0.5, 0.9, 'Bayesian Strategy Recommended', 
                   ha='center', va='top', transform=ax.transAxes, fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        else:
            ax.text(0.5, 0.9, 'Both Strategies Comparable', 
                   ha='center', va='top', transform=ax.transAxes, fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    
    def generate_comparison_report(self, output_file: str = None) -> str:
        """
        Generate a comprehensive comparison report.
        
        Args:
            output_file: Path to save the report (optional)
            
        Returns:
            Report content as string
        """
        # Run all comparisons if not already done
        if not self.comparison_results:
            self.compare_player_rankings()
            self.compare_strategy_approaches()
            self.generate_performance_metrics()
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("DRAFT STRATEGY COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Strategy Approaches
        if 'strategy_approaches' in self.comparison_results:
            report.append("STRATEGY APPROACHES")
            report.append("-" * 40)
            approaches = self.comparison_results['strategy_approaches']
            
            report.append("VOR Strategy:")
            for key, value in approaches['vor_approach'].items():
                report.append(f"  {key.replace('_', ' ').title()}: {value}")
            
            report.append("\nBayesian Strategy:")
            for key, value in approaches['bayesian_approach'].items():
                report.append(f"  {key.replace('_', ' ').title()}: {value}")
            
            report.append("\nKey Differences:")
            for diff in approaches['key_differences']:
                report.append(f"  • {diff}")
            
            report.append("\nComplementary Aspects:")
            for aspect in approaches['complementary_aspects']:
                report.append(f"  • {aspect}")
        
        # Performance Metrics
        if 'performance_metrics' in self.comparison_results:
            report.append("\n\nPERFORMANCE METRICS")
            report.append("-" * 40)
            metrics = self.comparison_results['performance_metrics']
            
            for strategy, perf in metrics.items():
                if strategy != 'comparison_metrics':
                    report.append(f"\n{strategy.replace('_', ' ').title()}:")
                    for key, value in perf.items():
                        report.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Player Rankings
        if 'player_rankings' in self.comparison_results:
            report.append("\n\nPLAYER RANKING ANALYSIS")
            report.append("-" * 40)
            rankings = self.comparison_results['player_rankings']
            
            report.append("Overall Metrics:")
            for key, value in rankings['metrics'].items():
                report.append(f"  {key.replace('_', ' ').title()}: {value}")
            
            report.append("\nPosition Analysis:")
            for pos, analysis in rankings['position_analysis'].items():
                report.append(f"  {pos}:")
                for key, value in analysis.items():
                    report.append(f"    {key.replace('_', ' ').title()}: {value}")
        
        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        report_content = "\n".join(report)
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            logger.info(f"Saved comparison report: {output_file}")
        
        return report_content
    
    def run_complete_comparison(self) -> Dict[str, Any]:
        """
        Run complete comparison analysis.
        
        Returns:
            Dictionary containing all comparison results
        """
        logger.info("Starting complete draft strategy comparison...")
        
        # Run all comparison methods
        self.compare_player_rankings()
        self.compare_strategy_approaches()
        self.generate_performance_metrics()
        
        # Generate visualizations
        viz_file = self.create_comparison_visualizations()
        
        # Generate report
        report_content = self.generate_comparison_report()
        
        # Add metadata
        self.comparison_results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'vor_file': self.vor_file,
            'bayesian_file': self.bayesian_file,
            'visualization_file': viz_file,
            'report_length': len(report_content)
        }
        
        logger.info("Complete comparison analysis finished")
        return self.comparison_results


def main():
    """Main function to run draft strategy comparison."""
    parser = argparse.ArgumentParser(description="Compare VOR and Bayesian draft strategies")
    parser.add_argument('--vor-file', type=str, default=None, help='Path to VOR CSV file')
    parser.add_argument('--bayesian-file', type=str, default=None, help='Path to Bayesian strategy JSON')
    parser.add_argument('--scenario-glob', type=str, default=None, help='Glob pattern for Bayesian JSONs to batch-compare (e.g., results/draft_strategy/draft_strategy_pos*.json)')
    args = parser.parse_args()

    if args.scenario_glob:
        # Support both absolute and relative patterns
        pattern = args.scenario_glob
        matched = list(Path().glob(pattern)) if any(ch in pattern for ch in ['/', '*', '?']) else list(Path('results/draft_strategy').glob(pattern))
        matched = sorted(matched)
        print(f"Found {len(matched)} scenario files for comparison")
        for json_path in matched:
            print(f"\n>>> Comparing using {json_path}")
            comparison = DraftStrategyComparison(vor_file=args.vor_file, bayesian_file=str(json_path))
            results = comparison.run_complete_comparison()
            # Save per-scenario report
            base = Path(json_path).stem
            # CRITICAL FIX: Ensure reports go to the correct organized subfolder
            from ffbayes.utils.path_constants import get_pre_draft_dir
            current_year = datetime.now().year
            report_dir = get_pre_draft_dir(current_year) / "draft_strategy_comparison"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f'draft_strategy_comparison_report_{base}.txt'
            comparison.generate_comparison_report(str(report_path))
            print(f"Report saved: {report_path}")
        return

    # Single run
    comparison = DraftStrategyComparison(vor_file=args.vor_file, bayesian_file=args.bayesian_file)
    results = comparison.run_complete_comparison()

    # Print summary
    print("\n" + "=" * 60)
    print("DRAFT STRATEGY COMPARISON SUMMARY")
    print("=" * 60)
    
    if 'player_rankings' in results:
        metrics = results['player_rankings']['metrics']
        avg_diff = metrics.get('avg_rank_difference')
        max_diff = metrics.get('max_rank_difference')
        print(f"Players analyzed: {metrics.get('common_players', 0)}")
        if isinstance(avg_diff, (int, float)):
            print(f"Average rank difference: {avg_diff:.1f}")
        if isinstance(max_diff, (int, float)):
            print(f"Max rank difference: {max_diff:.1f}")
    
    if 'performance_metrics' in results:
        vor_time = results['performance_metrics'].get('vor_performance', {}).get('execution_time', 'N/A')
        bayesian_time = results['performance_metrics'].get('bayesian_performance', {}).get('execution_time', 'N/A')
        print(f"\nVOR execution time: {vor_time}")
        print(f"Bayesian execution time: {bayesian_time}")
    
    print(f"\nVisualization saved: {results.get('metadata', {}).get('visualization_file', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
