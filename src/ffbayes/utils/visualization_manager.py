"""
Visualization Management Utilities

This module handles automatic copying of pipeline visualizations to docs/images
and updating the README with explanations of what each visualization shows.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Visualization descriptions for README
VISUALIZATION_DESCRIPTIONS = {
    # Pre-draft visualizations
    "draft_strategy_comparison": {
        "title": "Draft Strategy Comparison",
        "description": "Compares different draft strategies (VOR vs Bayesian) showing projected team strength and risk profiles. Useful for choosing the best drafting approach based on your league settings and risk tolerance.",
        "when_to_use": "Before your draft to understand which strategy aligns with your goals"
    },
    "position_distribution_analysis": {
        "title": "Position Distribution Analysis", 
        "description": "Shows optimal position allocation across your draft picks. Helps you understand when to target specific positions and avoid over-drafting any one position.",
        "when_to_use": "During draft planning to optimize position balance"
    },
    "draft_summary_dashboard": {
        "title": "Draft Summary Dashboard",
        "description": "Comprehensive overview of your draft strategy including player rankings, position targets, and risk assessment. Your main reference during the draft.",
        "when_to_use": "During your actual draft as your primary decision-making tool"
    },
    "uncertainty_analysis": {
        "title": "Uncertainty Analysis",
        "description": "Shows the confidence intervals and risk profiles for player projections. Helps you understand which players have more predictable vs volatile projections.",
        "when_to_use": "When evaluating players with similar projections to assess risk"
    },
    # REMOVED: vor_vs_bayesian_comparison - was just a useless diagonal line with no insights
    
    # Post-draft visualizations
    "team_composition": {
        "title": "Team Composition Chart",
        "description": "Visual breakdown of your drafted team by position, showing roster balance and depth. Helps identify strengths and potential weaknesses in your team structure.",
        "when_to_use": "After your draft to assess roster balance and identify waiver wire needs"
    },
    "team_strength_analysis": {
        "title": "Team Strength Analysis",
        "description": "Shows your team's projected weekly scoring potential with confidence intervals. Includes both mean projections and uncertainty ranges.",
        "when_to_use": "To understand your team's expected performance and variance"
    },
    "player_performance_projections": {
        "title": "Player Performance Projections",
        "description": "Individual player projections with confidence intervals. Shows which players are expected to be your top performers and which have high upside/downside risk.",
        "when_to_use": "For lineup decisions and trade evaluations"
    },
    "monte_carlo_validation": {
        "title": "Monte Carlo Validation",
        "description": "Simulation results showing your team's performance distribution across 5000 scenarios. Provides realistic range of outcomes based on historical data.",
        "when_to_use": "To understand your team's floor, ceiling, and most likely outcomes"
    },
    "team_summary_dashboard": {
        "title": "Team Summary Dashboard",
        "description": "Comprehensive post-draft analysis combining all metrics into one view. Shows team projections, player contributions, and key insights for the season.",
        "when_to_use": "Your main reference for understanding your team's outlook"
    },
    
    # Model comparison visualizations
    "model_quality_comparison": {
        "title": "Model Quality Comparison",
        "description": "Compares the accuracy and reliability of different projection models (Monte Carlo vs Bayesian). Shows which approach provides better predictions.",
        "when_to_use": "To understand the strengths and limitations of different projection methods"
    }
}


def copy_visualizations_to_docs(current_year: int = None) -> List[str]:
    """
    Copy all pipeline visualizations to docs/images folder.
    
    Args:
        current_year: Year to copy visualizations for (defaults to current year)
        
    Returns:
        List of copied file paths
    """
    if current_year is None:
        current_year = datetime.now().year
    
    # Source directories
    from ffbayes.utils.path_constants import (get_post_draft_plots_dir,
                                              get_pre_draft_plots_dir)
    pre_draft_dir = get_pre_draft_plots_dir(current_year) / "visualizations"
    post_draft_dir = get_post_draft_plots_dir(current_year)

    
    # Destination directory
    docs_images_dir = Path("docs/images")
    docs_images_dir.mkdir(exist_ok=True)
    
    copied_files = []
    
    # Copy pre-draft visualizations
    if pre_draft_dir.exists():
        for file_path in pre_draft_dir.glob("*.png"):
            dest_path = docs_images_dir / f"pre_draft_{file_path.stem}.png"
            shutil.copy2(file_path, dest_path)
            copied_files.append(str(dest_path))
    
    # Copy post-draft visualizations
    if post_draft_dir.exists():
        for file_path in post_draft_dir.glob("*.png"):
            dest_path = docs_images_dir / f"post_draft_{file_path.stem}.png"
            shutil.copy2(file_path, dest_path)
            copied_files.append(str(dest_path))
    

    
    return copied_files


def update_readme_with_visualizations() -> bool:
    """
    Update the README.md file to include visualization explanations.
    
    Returns:
        True if README was updated successfully
    """
    readme_path = Path("README.md")
    if not readme_path.exists():
        print("⚠️  README.md not found")
        return False
    
    # Read current README
    with open(readme_path, 'r') as f:
        readme_content = f.read()
    
    # Check if visualizations section already exists
    if "## 📊 Visualizations" in readme_content:
        print("📝 Visualizations section already exists in README")
        return True
    
    # Create visualizations section
    viz_section = create_visualizations_section()
    
    # Add to README (before the last section)
    sections = readme_content.split("\n## ")
    if len(sections) > 1:
        # Insert before the last section
        sections.insert(-1, f"\n## {viz_section}")
        new_content = "\n## ".join(sections)
    else:
        # Append to end
        new_content = readme_content + "\n\n" + viz_section
    
    # Write updated README
    with open(readme_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Updated README.md with visualization explanations")
    return True


def create_visualizations_section() -> str:
    """Create the visualizations section content for README."""
    
    section = """📊 Visualizations

FFBayes generates comprehensive visualizations to help you make informed fantasy football decisions. All visualizations are automatically copied to `docs/images/` after each pipeline run.

### Pre-Draft Visualizations

These help you prepare for your draft:

"""
    
    # Add pre-draft visualizations
    pre_draft_viz = [
        "draft_strategy_comparison",
        "position_distribution_analysis", 
        "draft_summary_dashboard",
        "uncertainty_analysis",
        "vor_vs_bayesian_comparison"
    ]
    
    for viz_key in pre_draft_viz:
        if viz_key in VISUALIZATION_DESCRIPTIONS:
            desc = VISUALIZATION_DESCRIPTIONS[viz_key]
            section += f"**{desc['title']}** - {desc['description']} *({desc['when_to_use']})*\n\n"
    
    section += """### Post-Draft Visualizations

These help you analyze your drafted team:

"""
    
    # Add post-draft visualizations
    post_draft_viz = [
        "team_composition",
        "team_strength_analysis",
        "player_performance_projections", 
        "monte_carlo_validation",
        "team_summary_dashboard"
    ]
    
    for viz_key in post_draft_viz:
        if viz_key in VISUALIZATION_DESCRIPTIONS:
            desc = VISUALIZATION_DESCRIPTIONS[viz_key]
            section += f"**{desc['title']}** - {desc['description']} *({desc['when_to_use']})*\n\n"
    
    section += """### Model Comparison Visualizations

These help you understand projection accuracy:

"""
    
    # Add model comparison visualizations
    model_viz = ["model_quality_comparison"]
    
    for viz_key in model_viz:
        if viz_key in VISUALIZATION_DESCRIPTIONS:
            desc = VISUALIZATION_DESCRIPTIONS[viz_key]
            section += f"**{desc['title']}** - {desc['description']} *({desc['when_to_use']})*\n\n"
    
    section += """### How to Use These Visualizations

1. **Before Your Draft**: Review pre-draft visualizations to understand optimal strategies
2. **During Your Draft**: Use the draft summary dashboard as your primary reference
3. **After Your Draft**: Analyze post-draft visualizations to understand your team's outlook
4. **Throughout the Season**: Refer back to projections and uncertainty analysis for lineup decisions

All visualizations are automatically updated after each pipeline run, ensuring you always have the latest analysis."""
    
    return section


def cleanup_old_visualizations() -> int:
    """
    Remove old visualization files from docs/images that are no longer relevant.
    
    Returns:
        Number of files removed
    """
    docs_images_dir = Path("docs/images")
    if not docs_images_dir.exists():
        return 0
    
    removed_count = 0
    current_year = datetime.now().year
    
    # Remove files older than current year
    for file_path in docs_images_dir.glob("*.png"):
        # Check if file is from a previous year (simple heuristic)
        if file_path.stat().st_mtime < datetime(current_year, 1, 1).timestamp():
            file_path.unlink()
            removed_count += 1
    
    return removed_count


def manage_visualizations(current_year: int = None) -> Dict[str, any]:
    """
    Main function to manage all visualization operations.
    
    Args:
        current_year: Year to process (defaults to current year)
        
    Returns:
        Dictionary with operation results
    """
    if current_year is None:
        current_year = datetime.now().year
    
    print(f"🖼️  Managing visualizations for {current_year}...")
    
    # Copy new visualizations
    copied_files = copy_visualizations_to_docs(current_year)
    
    # Update README
    readme_updated = update_readme_with_visualizations()
    
    # Cleanup old files
    removed_count = cleanup_old_visualizations()
    
    results = {
        "copied_files": copied_files,
        "readme_updated": readme_updated,
        "removed_old_files": removed_count,
        "year": current_year
    }
    
    print("✅ Visualization management complete:")
    print(f"   📁 Copied {len(copied_files)} files to docs/images/")
    print(f"   📝 README updated: {readme_updated}")
    print(f"   🗑️  Removed {removed_count} old files")
    
    return results


if __name__ == "__main__":
    # Test the visualization management
    results = manage_visualizations()
    print(f"Results: {results}")
