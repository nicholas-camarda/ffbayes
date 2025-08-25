#!/usr/bin/env python3
"""
Hybrid Model Evaluation - Compute MAE against historical ground truth

- Select a holdout season (default: last completed season)
- Load unified dataset and Hybrid MC predictions
- Build ground truth per-player (mean FantPt in holdout season)
- Align players and compute MAE between predicted mean and truth
- Save evaluation JSON under results/<year>/model_evaluation/
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def get_output_dir() -> Path:
	current_year = datetime.now().year
	from ffbayes.utils.path_constants import get_results_dir
	out_dir = get_results_dir(current_year) / "model_evaluation"
	out_dir.mkdir(parents=True, exist_ok=True)
	return out_dir


def load_unified() -> pd.DataFrame:
	from ffbayes.data_pipeline.unified_data_loader import load_unified_dataset
	return load_unified_dataset('datasets')


def load_hybrid_predictions() -> dict:
	from ffbayes.utils.path_constants import get_hybrid_mc_dir
	hyb_dir = get_hybrid_mc_dir(datetime.now().year)
	path = hyb_dir / 'hybrid_model_results.json'
	if not path.exists():
		raise FileNotFoundError(f"Hybrid predictions not found: {path}")
	with open(path, 'r') as f:
		return json.load(f)


def compute_ground_truth(df: pd.DataFrame, holdout_year: int) -> pd.Series:
	"""Return per-player mean FantPt in holdout season."""
	season_df = df[df['Season'] == holdout_year]
	if season_df.empty:
		raise ValueError(f"No data for holdout season {holdout_year}")
	# Group by player and average per-game fantasy points
	return season_df.groupby('Name')['FantPt'].mean()


def compute_mae(pred: pd.Series, truth: pd.Series) -> float:
	common = pred.index.intersection(truth.index)
	if len(common) == 0:
		raise ValueError("No overlapping players between predictions and ground truth")
	diff = (pred.loc[common] - truth.loc[common]).abs()
	return float(diff.mean())


def main():
	print("==============================")
	print("Hybrid Model Evaluation (MAE)")
	print("==============================")
	current_year = datetime.now().year
	# Choose last completed season as holdout
	holdout_year = current_year - 1
	print(f"Holdout season: {holdout_year}")
	
	# Load data
	unified = load_unified()
	hybrid = load_hybrid_predictions()
	
	# Extract predicted per-player mean (per-game) from Hybrid file
	predicted_means = {}
	for name, pdata in hybrid.items():
		if not isinstance(pdata, dict):
			continue
		mc = pdata.get('monte_carlo', {})
		if 'mean' in mc:
			predicted_means[name] = float(mc['mean'])
	pred_series = pd.Series(predicted_means, name='pred_mean')
	print(f"Loaded predictions for {len(pred_series)} players")
	
	# Ground truth from holdout season
	truth_series = compute_ground_truth(unified, holdout_year)
	truth_series.name = 'truth_mean'
	print(f"Computed ground truth for {len(truth_series)} players")
	
	# Compute MAE on overlap
	mae = compute_mae(pred_series, truth_series)
	common_players = int(pred_series.index.intersection(truth_series.index).size)
	
	# Save evaluation JSON
	out_dir = get_output_dir()
	out_path = out_dir / f"hybrid_evaluation_{current_year}_vs_{holdout_year}.json"
	result = {
		'timestamp': datetime.now().isoformat(),
		'evaluation': 'hybrid_vs_ground_truth',
		'current_year': current_year,
		'holdout_year': holdout_year,
		'metrics': {
			'mae_bayesian': mae,
			'num_players_evaluated': common_players
		}
	}
	with open(out_path, 'w') as f:
		json.dump(result, f, indent=2)
	print(f"✅ Evaluation saved to: {out_path}")


if __name__ == '__main__':
	main()
