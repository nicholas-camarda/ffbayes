"""
Position Color Scheme for FFBayes Visualizations

This module defines consistent position colors used across all visualization scripts.
Centralizing these colors ensures consistency and makes maintenance easier.
"""

# Consistent position color scheme used across all FFBayes visualizations
POSITION_COLORS = {
    'QB': '#3498db',  # Blue
    'RB': '#e67e22',  # Orange  
    'WR': '#2ecc71',  # Green
    'TE': '#e74c3c',  # Red
    'FB': '#f39c12',  # Yellow/Orange
    'K': '#9b59b6',   # Purple
    'DEF': '#34495e', # Dark gray
    'DST': '#34495e', # Dark gray (alias for DEF)
    'UNK': '#95a5a6'  # Light gray
}

def get_position_color(position):
    """
    Get the color for a given position.
    
    Args:
        position (str): Position abbreviation (QB, RB, WR, TE, FB, K, DEF, DST, UNK)
        
    Returns:
        str: Hex color code for the position
    """
    return POSITION_COLORS.get(position, '#95a5a6')  # Default to light gray

def get_position_colors_for_players(positions):
    """
    Get colors for a list of positions.
    
    Args:
        positions (list): List of position strings
        
    Returns:
        list: List of hex color codes
    """
    return [get_position_color(pos) for pos in positions]

def get_legend_elements(unique_positions):
    """
    Create matplotlib legend elements for position colors.
    
    Args:
        unique_positions (list): List of unique positions to include in legend
        
    Returns:
        list: List of matplotlib Patch objects for legend
    """
    from matplotlib.patches import Patch
    
    legend_elements = []
    for position in unique_positions:
        if position in POSITION_COLORS:
            color = POSITION_COLORS[position]
            legend_elements.append(Patch(facecolor=color, label=position))
    
    return legend_elements
