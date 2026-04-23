from __future__ import annotations

import pandas as pd

from ffbayes.analysis import sampled_player_model as spm


def _make_sampled_history() -> pd.DataFrame:
    rows: list[dict] = []
    season_profiles = {
        'Alpha QB': {
            'position': 'QB',
            'team': 'NYG',
            'weekly': {
                2019: [19.0, 20.0, 18.5, 21.0, 20.5, 19.5, 21.5, 22.0],
                2020: [20.0, 21.5, 20.5, 22.0, 21.0, 20.5, 22.5, 23.0],
                2021: [21.0, 22.0, 21.5, 22.5, 23.0, 22.0, 23.5, 24.0],
                2022: [22.0, 23.0, 22.5, 24.0, 23.5, 23.0, 24.5, 25.0],
            },
        },
        'Beta RB': {
            'position': 'RB',
            'team': 'DAL',
            'weekly': {
                2019: [13.0, 14.0, 12.5, 15.0, 14.5, 13.5, 15.5, 16.0],
                2020: [14.0, 15.0, 14.5, 15.5, 15.0, 14.5, 16.0, 16.5],
                2021: [15.0, 15.5, 16.0, 16.5, 15.5, 16.0, 16.5, 17.0],
                2022: [15.5, 16.0, 16.5, 17.0, 16.5, 17.0, 17.5, 18.0],
            },
        },
        'Gamma WR': {
            'position': 'WR',
            'team': 'PHI',
            'weekly': {
                2019: [11.0, 11.5, 10.5, 12.0, 11.5, 11.0, 12.5, 13.0],
                2020: [11.5, 12.0, 12.5, 13.0, 12.0, 12.5, 13.5, 14.0],
                2021: [12.5, 13.0, 13.5, 14.0, 13.5, 14.0, 14.5, 15.0],
                2022: [13.0, 13.5, 14.0, 14.5, 14.0, 14.5, 15.0, 15.5],
            },
        },
        'Delta TE': {
            'position': 'TE',
            'team': 'BUF',
            'weekly': {
                2019: [8.0, 8.5, 8.0, 9.0, 8.5, 8.0, 9.5, 10.0],
                2020: [8.5, 9.0, 8.5, 9.5, 9.0, 8.5, 10.0, 10.5],
                2021: [9.0, 9.5, 9.0, 10.0, 9.5, 9.0, 10.5, 11.0],
                2022: [9.5, 10.0, 9.5, 10.5, 10.0, 9.5, 11.0, 11.5],
            },
        },
    }
    for name, profile in season_profiles.items():
        for season, weekly_points in profile['weekly'].items():
            for week, points in enumerate(weekly_points, start=1):
                rows.append(
                    {
                        'Season': season,
                        'Name': name,
                        'Position': profile['position'],
                        'FantPt': points,
                        'FantPtPPR': points + 1.0,
                        'G#': week,
                        'Tm': profile['team'],
                        'Opp': 'MIA' if week % 2 else 'NE',
                        'Age': 23 + (season - 2019),
                        'adp': 20 + week,
                        'adp_rank': 10 + week,
                        'vor_value': points * 0.8,
                        'consistency_score_latest': 0.2,
                        'floor_ceiling_spread_latest': 4.0,
                        'team_usage_pct_latest': 0.25,
                        'recent_form_latest': 0.1,
                        'season_trend_latest': 0.1,
                        'role_strength_z': 0.2,
                        'RAV': 5.0,
                        'tier_cliff_distance': 2.0,
                        'site_disagreement': 0.1,
                        'depth_chart_rank': 1,
                    }
                )
    return pd.DataFrame(rows)


def test_build_sampled_bayes_search_space_matches_bounded_contract():
    search_space = spm.build_sampled_bayes_search_space()

    assert len(search_space) == 72
    assert {config.parameterization for config in search_space} == {
        'centered',
        'non_centered',
    }
    assert {config.prior_scale_family for config in search_space} == {
        'conservative',
        'medium',
        'weaker',
    }
    assert {config.target_accept for config in search_space} == {0.9, 0.95}
    assert {(config.tune, config.draws) for config in search_space} == {
        (1000, 1000),
        (1500, 1500),
    }
    assert {config.structural_variant for config in search_space} == {
        'base',
        'team_season',
        'team_season_rookie_priors',
    }


def test_select_sampled_bayes_winner_rejects_pathological_configs():
    search_space = spm.build_sampled_bayes_search_space()[:2]
    empirical_baseline = {
        'overall_forecast': {
            'mae': 10.0,
            'rmse': 12.0,
            'calibration': {'interval_80_coverage': 0.80},
        }
    }
    good_result = {
        'status': 'completed',
        'config': search_space[0].to_dict(),
        'metrics': {
            'mae': 9.7,
            'rmse': 11.7,
            'calibration': {'interval_80_coverage': 0.79},
        },
        'runtime_seconds': 30.0,
    }
    rejected_result = {
        'status': 'rejected',
        'config': search_space[1].to_dict(),
        'reason': 'sampled_component_convergence_failed',
    }

    selection = spm.select_sampled_bayes_winner(
        [good_result, rejected_result],
        empirical_baseline=empirical_baseline,
        search_space=search_space,
    )

    assert selection['preferred_config']['config_id'] == search_space[0].config_id
    assert selection['promoted'] is False
    assert selection['production_estimator'] == 'hierarchical_empirical_bayes'
    assert selection['rejected_config_ids'] == [search_space[1].config_id]


def test_run_sampled_bayes_search_keeps_empirical_default_when_sampled_loses(
    monkeypatch,
):
    history = _make_sampled_history()
    search_space = spm.build_sampled_bayes_search_space()[:2]

    monkeypatch.setattr(
        spm,
        'build_player_forecast_validation_summary',
        lambda *args, **kwargs: {
            'overall_forecast': {
                'mae': 8.0,
                'rmse': 10.0,
                'calibration': {'interval_80_coverage': 0.80},
            }
        },
    )
    monkeypatch.setattr(
        spm,
        'evaluate_sampled_hierarchical_config',
        lambda history, config, **kwargs: {
            'status': 'completed',
            'config': config.to_dict(),
            'metrics': {
                'mae': 8.3,
                'rmse': 10.4,
                'calibration': {'interval_80_coverage': 0.79},
            },
            'runtime_seconds': 15.0,
        },
    )

    report = spm.run_sampled_bayes_search(
        history,
        holdout_years=[2022],
        candidate_configs=search_space,
    )

    assert report['production_estimator'] == 'hierarchical_empirical_bayes'
    assert report['promotion_decision'] == 'keep_empirical_bayes'
    assert report['promoted'] is False
    assert report['selection']['evaluated_config_count'] == len(search_space)


def test_evaluate_sampled_hierarchical_config_emits_diagnostics(monkeypatch):
    history = _make_sampled_history()
    config = spm.SampledBayesConfig(
        parameterization='non_centered',
        prior_scale_family='medium',
        target_accept=0.95,
        chains=2,
        tune=60,
        draws=60,
        structural_variant='base',
    )
    monkeypatch.setitem(spm.SAMPLED_PROMOTION_CRITERIA, 'min_ess_bulk', 10.0)
    monkeypatch.setitem(spm.SAMPLED_PROMOTION_CRITERIA, 'max_rhat', 1.2)
    monkeypatch.setitem(spm.SAMPLED_PROMOTION_CRITERIA, 'max_divergences', 10)

    result = spm.evaluate_sampled_hierarchical_config(
        history,
        config=config,
        holdout_years=[2022],
        random_seed=123,
    )

    assert result['status'] in {'completed', 'rejected'}
    assert result['config']['config_id'] == config.config_id
    assert result['holdout_years'] == [2022]
    if result['status'] == 'completed':
        assert result['metrics']['player_count'] > 0
        assert 'mae' in result['metrics']
        assert 'calibration' in result['metrics']
        assert result['component_diagnostics'][0]['rate']['converged'] is True
        assert result['component_diagnostics'][0]['availability']['converged'] is True
    else:
        assert result['reason'] == 'sampled_component_convergence_failed'
        assert 'rate' in result['component_diagnostics']
        assert 'availability' in result['component_diagnostics']
