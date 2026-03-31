"""Visualization and publication helpers.

The runtime pipeline writes artifacts locally first. This module is only used
when a user explicitly asks to mirror selected runtime outputs into the cloud
SideProjects workspace.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _copy_tree(source_dir: Path, destination_dir: Path) -> List[str]:
    """Copy a directory tree while preserving relative paths."""
    copied_files: List[str] = []
    if not source_dir.exists():
        return copied_files

    for file_path in source_dir.rglob('*'):
        if not file_path.is_file():
            continue
        relative_path = file_path.relative_to(source_dir)
        dest_path = destination_dir / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest_path)
        copied_files.append(str(dest_path))

    return copied_files


def _phase_directories(
    current_year: int, phase: str | None
) -> list[tuple[str, Path, Path, Path, Path]]:
    """Return runtime/cloud directory pairs for the requested phase."""
    from ffbayes.utils.path_constants import (
        get_cloud_pre_draft_dir,
        get_cloud_pre_draft_plots_dir,
        get_pre_draft_artifacts_dir,
        get_pre_draft_diagnostics_dir,
    )

    resolved = (phase or 'both').lower()
    mappings: list[tuple[str, Path, Path, Path, Path]] = []
    if resolved in {'pre_draft', 'both'}:
        mappings.append(
            (
                'pre_draft',
                get_pre_draft_artifacts_dir(current_year),
                get_cloud_pre_draft_dir(current_year),
                get_pre_draft_diagnostics_dir(current_year),
                get_cloud_pre_draft_plots_dir(current_year),
            )
        )
    if not mappings:
        raise ValueError(f'Unknown publish phase: {phase}')
    return mappings


# Visualization descriptions for README
VISUALIZATION_DESCRIPTIONS = {
    # Pre-draft visualizations
    'draft_strategy_comparison': {
        'title': 'Draft Strategy Comparison',
        'description': 'Compares different draft strategies (VOR vs Bayesian) showing projected team strength and risk profiles. Useful for choosing the best drafting approach based on your league settings and risk tolerance.',
        'when_to_use': 'Before your draft to understand which strategy aligns with your goals',
    },
    'position_distribution_analysis': {
        'title': 'Position Distribution Analysis',
        'description': 'Shows optimal position allocation across your draft picks. Helps you understand when to target specific positions and avoid over-drafting any one position.',
        'when_to_use': 'During draft planning to optimize position balance',
    },
    'draft_summary_dashboard': {
        'title': 'Draft Summary Dashboard',
        'description': 'Comprehensive overview of your draft strategy including player rankings, position targets, and risk assessment. Your main reference during the draft.',
        'when_to_use': 'During your actual draft as your primary decision-making tool',
    },
    'uncertainty_analysis': {
        'title': 'Uncertainty Analysis',
        'description': 'Shows the confidence intervals and risk profiles for player projections. Helps you understand which players have more predictable vs volatile projections.',
        'when_to_use': 'When evaluating players with similar projections to assess risk',
    },
    # REMOVED: vor_vs_bayesian_comparison - was just a useless diagonal line with no insights
    # Model comparison visualizations
    'model_quality_comparison': {
        'title': 'Model Quality Comparison',
        'description': 'Compares the accuracy and reliability of different projection models (Monte Carlo vs Bayesian). Shows which approach provides better predictions.',
        'when_to_use': 'To understand the strengths and limitations of different projection methods',
    },
}


def copy_visualizations_to_docs(
    current_year: int = None, phase: str | None = None
) -> List[str]:
    """Copy rendered visualization PNGs into the cloud docs/images folder."""
    if current_year is None:
        current_year = datetime.now().year

    from ffbayes.utils.path_constants import get_cloud_docs_images_dir

    docs_images_dir = get_cloud_docs_images_dir()

    copied_files = []

    for phase_name, _, _, runtime_plot_dir, _ in _phase_directories(
        current_year, phase
    ):
        for file_path in runtime_plot_dir.rglob('*.png'):
            dest_path = docs_images_dir / f'{phase_name}_{file_path.stem}.png'
            shutil.copy2(file_path, dest_path)
            copied_files.append(str(dest_path))

    return copied_files


def update_readme_with_visualizations() -> bool:
    """
    Deprecated helper preserved for compatibility.

    Returns:
        False, because README is no longer mutated automatically
    """
    print('ℹ️  README publication is disabled; use ffbayes-publish for cloud mirroring')
    return False


def create_visualizations_section() -> str:
    """Create the visualizations section content for README."""

    section = """📊 Visualizations

FFBayes generates comprehensive visualizations to help you make informed fantasy football decisions. Runtime artifacts stay local until you explicitly publish them.

### Pre-Draft Visualizations

These help you prepare for your draft:

"""

    # Add pre-draft visualizations
    pre_draft_viz = [
        'draft_strategy_comparison',
        'position_distribution_analysis',
        'draft_summary_dashboard',
        'uncertainty_analysis',
    ]

    for viz_key in pre_draft_viz:
        if viz_key in VISUALIZATION_DESCRIPTIONS:
            desc = VISUALIZATION_DESCRIPTIONS[viz_key]
            section += f'**{desc["title"]}** - {desc["description"]} *({desc["when_to_use"]})*\n\n'

    section += """### Model Comparison Visualizations

These help you understand projection accuracy:

"""

    # Add model comparison visualizations
    model_viz = ['model_quality_comparison']

    for viz_key in model_viz:
        if viz_key in VISUALIZATION_DESCRIPTIONS:
            desc = VISUALIZATION_DESCRIPTIONS[viz_key]
            section += f'**{desc["title"]}** - {desc["description"]} *({desc["when_to_use"]})*\n\n'

    section += """### How to Use These Visualizations

1. **Before Your Draft**: Review the live command center to understand optimal strategies
2. **During Your Draft**: Use the draft board as your primary reference
3. **Throughout the Season**: Refer back to projections and uncertainty analysis for lineup decisions

All visualizations remain local unless you run the explicit publish command."""

    return section


def cleanup_old_visualizations() -> int:
    """
    Remove old visualization files from the cloud docs/images tree that are no longer relevant.

    Returns:
        Number of files removed
    """
    from ffbayes.utils.path_constants import get_cloud_docs_images_dir

    docs_images_dir = get_cloud_docs_images_dir()
    if not docs_images_dir.exists():
        return 0

    removed_count = 0
    current_year = datetime.now().year

    # Remove files older than current year
    for file_path in docs_images_dir.glob('*.png'):
        # Check if file is from a previous year (simple heuristic)
        if file_path.stat().st_mtime < datetime(current_year, 1, 1).timestamp():
            file_path.unlink()
            removed_count += 1

    return removed_count


def manage_visualizations(
    current_year: int = None, phase: str | None = None
) -> Dict[str, Any]:
    """
    Main function to publish visualizations and related outputs.

    Args:
        current_year: Year to process (defaults to current year)

    Returns:
        Dictionary with operation results
    """
    if current_year is None:
        current_year = datetime.now().year

    resolved_phase = (phase or 'pre_draft').lower()
    if resolved_phase != 'pre_draft':
        raise ValueError('Only pre_draft visualization publication is supported')
    print(f'🖼️  Publishing {resolved_phase} visualizations for {current_year}...')

    from ffbayes.utils.path_constants import (
        get_cloud_pre_draft_dashboard_dir,
        get_pre_draft_artifacts_dir,
    )

    published_result_files = []
    published_plot_files = []
    published_dashboard_files = []

    for (
        phase_name,
        runtime_result_dir,
        cloud_result_dir,
        runtime_plot_dir,
        cloud_plot_dir,
    ) in _phase_directories(current_year, phase):
        published_result_files.extend(_copy_tree(runtime_result_dir, cloud_result_dir))
        published_plot_files.extend(_copy_tree(runtime_plot_dir, cloud_plot_dir))

        runtime_dashboard_dir = get_pre_draft_artifacts_dir(current_year)
        cloud_dashboard_dir = get_cloud_pre_draft_dashboard_dir(current_year)
        published_dashboard_files.extend(
            _copy_tree(runtime_dashboard_dir, cloud_dashboard_dir)
        )

    # Copy rendered PNGs into the published docs tree.
    docs_image_files = copy_visualizations_to_docs(current_year, phase)

    # Cleanup old files from the cloud docs tree.
    removed_count = cleanup_old_visualizations()

    results = {
        'copied_files': published_result_files
        + published_plot_files
        + published_dashboard_files
        + docs_image_files,
        'published_docs_images': docs_image_files,
        'published_result_files': published_result_files,
        'published_plot_files': published_plot_files,
        'published_dashboard_files': published_dashboard_files,
        'readme_updated': False,
        'removed_old_files': removed_count,
        'year': current_year,
        'phase': resolved_phase,
    }

    print('✅ Visualization management complete:')
    print(f'   📁 Published {len(published_result_files)} result files to cloud')
    print(f'   📁 Published {len(published_plot_files)} plot files to cloud')
    print(f'   📁 Published {len(published_dashboard_files)} dashboard files to cloud')
    print(f'   🖼️  Copied {len(docs_image_files)} docs images to cloud')
    print(f'   🗑️  Removed {removed_count} old cloud docs images')

    return results


if __name__ == '__main__':
    # Test the visualization management
    results = manage_visualizations()
    print(f'Results: {results}')
