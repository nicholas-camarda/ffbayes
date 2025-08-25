#!/usr/bin/env python3
"""
Build team aggregation results JSON from the latest Monte Carlo TSV output.
Generates results/team_aggregation/team_aggregation_results_YYYYMMDD_HHMMSS.json
compatible with create_team_aggregation_visualizations.py.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def find_latest_mc_tsv() -> Optional[Path]:
    candidates = []
    # Search both results and plots directories for projections TSVs
    from ffbayes.utils.training_config import get_monte_carlo_training_years
    training_years = get_monte_carlo_training_years()
    current_year = datetime.now().year
    
    from ffbayes.utils.path_constants import get_monte_carlo_dir, get_plots_dir
    for pattern in [
        str(get_monte_carlo_dir(current_year) / f"mc_projections_{current_year}_*.tsv"),
        str(get_plots_dir(current_year) / "*projections*.tsv"),
    ]:
        candidates.extend(Path().glob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_latest_combined_dataset() -> Optional[Path]:
    from ffbayes.utils.path_constants import COMBINED_DATASETS_DIR
    candidates = list(Path(COMBINED_DATASETS_DIR).glob("*season_modern.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def build_positions_map(player_names: pd.Index) -> Dict[str, str]:
    positions: Dict[str, str] = {}
    ds_path = find_latest_combined_dataset()
    if ds_path is None:
        return {name: "UNK" for name in player_names}
    df = pd.read_csv(ds_path)
    # Use most recent season available in file
    if "Season" in df.columns:
        df = df[df["Season"] == df["Season"].max()]
    if {"Name", "Position"}.issubset(df.columns):
        name_to_pos = (df[["Name", "Position"]]
                       .dropna()
                       .drop_duplicates()
                       .set_index("Name")["Position"].to_dict())
        for name in player_names:
            positions[name] = str(name_to_pos.get(name, "UNK"))
    else:
        positions = {name: "UNK" for name in player_names}
    return positions


def build_aggregation_from_tsv(tsv_path: Path) -> dict:
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    if df.empty or "Total" not in df.columns:
        raise ValueError("Monte Carlo TSV missing 'Total' column or is empty")

    # Team projection stats
    team_scores = df["Total"].astype(float)
    mean_score = float(team_scores.mean())
    std_score = float(team_scores.std())
    se_score = float(std_score / np.sqrt(len(team_scores))) if len(team_scores) > 0 else 0.0
    min_score = float(team_scores.min())
    max_score = float(team_scores.max())
    ci_lower = float(mean_score - 1.96 * se_score)
    ci_upper = float(mean_score + 1.96 * se_score)
    percentiles = {
        "p5": float(team_scores.quantile(0.05)),
        "p25": float(team_scores.quantile(0.25)),
        "p50": float(team_scores.quantile(0.50)),
        "p75": float(team_scores.quantile(0.75)),
        "p95": float(team_scores.quantile(0.95)),
    }

    # Player contributions
    player_cols = [c for c in df.columns if c != "Total"]
    means = df[player_cols].mean().astype(float)
    stds = df[player_cols].std().astype(float)
    total_player_points = float(means.sum()) if len(means) else 0.0

    # Build positions map from latest combined dataset
    positions_map = build_positions_map(pd.Index(player_cols))

    contributions = {}
    for player in player_cols:
        mean_val = float(means.get(player, 0.0))
        std_val = float(stds.get(player, 0.0))
        contrib_pct = (mean_val / total_player_points * 100.0) if total_player_points > 0 else 0.0
        contributions[player] = {
            "mean": mean_val,
            "std": std_val,
            "contribution_pct": contrib_pct,
            "position": positions_map.get(player, "UNK"),
        }

    team_projection_block = {
        "total_score": {
            "mean": mean_score,
            "std": std_score,
            "min": min_score,
            "max": max_score,
            "confidence_interval": [ci_lower, ci_upper],
            "percentiles": percentiles,
        }
    }

    results = {
        "monte_carlo_projection": {
            "team_projection": team_projection_block,
            "player_contributions": contributions,
        },
        # Duplicate at top-level for visualization convenience
        "team_projection": team_projection_block,
        "player_positions": positions_map,
        "simulation_metadata": {
            "number_of_simulations": int(len(df)),
            "execution_time": 0,
            "convergence_status": "unknown",
        },
        "timestamp": datetime.now().isoformat(),
        # Minimal roster analysis placeholder
        "roster_analysis": {
            "total_roster_size": int(len(player_cols)),
            "missing_players_count": 0,
            "roster_coverage_percentage": 100.0 if len(player_cols) > 0 else 0.0,
        },
    }
    return results


def main() -> int:
    latest = find_latest_mc_tsv()
    if latest is None:
        print("❌ No Monte Carlo TSV found. Run Phase B first.")
        return 1

    from ffbayes.utils.path_constants import get_team_aggregation_dir
    current_year = datetime.now().year
    out_dir = get_team_aggregation_dir(current_year)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_aggregation_from_tsv(latest)

    # Generate filename with draft year instead of timestamp
    current_year = datetime.now().year
    ts = f"{current_year}"
    out_path = out_dir / f"team_aggregation_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"✅ Team aggregation results saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
