from __future__ import annotations

import json
from pathlib import Path

import ffbayes.run_pipeline_split as run_pipeline_split


def _load_step_map(config_name: str) -> dict[str, dict]:
    config_path = Path(__file__).resolve().parents[1] / 'config' / config_name
    config = json.loads(config_path.read_text(encoding='utf-8'))
    return {step['name']: step for step in config['steps']}


def test_pipeline_configs_explicitly_allow_stale_season():
    pre_steps = _load_step_map('pipeline_pre_draft.json')

    assert pre_steps['data_collection']['args'] == '--allow-stale-season'
    assert (
        pre_steps['data_collection']['env']['FFBAYES_PROCESS_DATASET_PROGRESS']
        == 'summary'
    )
    assert pre_steps['data_validation']['env']['FFBAYES_ALLOW_STALE_SEASON'] == 'true'
    assert pre_steps['data_preprocessing']['env']['FFBAYES_ALLOW_STALE_SEASON'] == 'true'
    assert (
        pre_steps['create_unified_dataset']['env']['FFBAYES_ALLOW_STALE_SEASON']
        == 'true'
    )
    assert (
        pre_steps['draft_strategy_comparison']['env']['FFBAYES_ALLOW_STALE_SEASON']
        == 'true'
    )

    config_root = Path(__file__).resolve().parents[1] / 'config'
    assert not (config_root / 'pipeline_post_draft.json').exists()


def test_split_runner_forwards_step_env_and_args(monkeypatch):
    runner = run_pipeline_split.SplitPipelineRunner.__new__(
        run_pipeline_split.SplitPipelineRunner
    )
    runner.pipeline_type = 'pre_draft'
    runner.created_dirs = set()
    runner.create_output_directories = lambda step_name: None
    runner.organize_step_outputs = lambda step_name: None
    runner._log = lambda *args, **kwargs: None

    captured: dict[str, object] = {}

    def fake_run(cmd, capture_output, text, timeout, env):
        captured['cmd'] = cmd
        captured['env'] = env

        class Result:
            returncode = 0
            stdout = ''
            stderr = ''

        return Result()

    monkeypatch.setattr(run_pipeline_split.subprocess, 'run', fake_run)

    success = run_pipeline_split.SplitPipelineRunner.run_step(
        runner,
        {
            'name': 'data_collection',
            'script': 'ffbayes.data_pipeline.collect_data',
            'args': '--allow-stale-season --years "2024,2025"',
            'env': {
                'FFBAYES_ALLOW_STALE_SEASON': 'true',
                'FFBAYES_PROCESS_DATASET_PROGRESS': 'summary',
            },
        },
    )

    assert success is True
    assert captured['cmd'][:3] == [
        run_pipeline_split.sys.executable,
        '-m',
        'ffbayes.data_pipeline.collect_data',
    ]
    assert '--allow-stale-season' in captured['cmd']
    assert '--years' in captured['cmd']
    assert captured['env']['FFBAYES_PIPELINE_PHASE'] == 'pre_draft'
    assert captured['env']['FFBAYES_ALLOW_STALE_SEASON'] == 'true'
    assert captured['env']['FFBAYES_PROCESS_DATASET_PROGRESS'] == 'summary'
