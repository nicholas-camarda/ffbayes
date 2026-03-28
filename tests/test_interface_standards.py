from ffbayes.utils.interface_standards import (
    get_env_bool,
    get_env_int,
    get_standard_paths,
    handle_exception,
    setup_logger,
)


def test_env_helpers(monkeypatch):
    monkeypatch.setenv('TEST_BOOL_TRUE', 'true')
    monkeypatch.setenv('TEST_BOOL_ONE', '1')
    monkeypatch.setenv('TEST_BOOL_NO', 'no')
    monkeypatch.setenv('TEST_INT_OK', '42')
    monkeypatch.setenv('TEST_INT_BAD', 'abc')

    assert get_env_bool('TEST_BOOL_TRUE', False) is True
    assert get_env_bool('TEST_BOOL_ONE', False) is True
    assert get_env_bool('TEST_BOOL_NO', True) is False
    assert get_env_bool('TEST_BOOL_MISSING', True) is True

    assert get_env_int('TEST_INT_OK', 0) == 42
    assert get_env_int('TEST_INT_BAD', 7) == 7
    assert get_env_int('TEST_INT_MISSING', 9) == 9


def test_standard_paths_with_explicit_root(tmp_path):
    paths = get_standard_paths(tmp_path)

    assert paths.plots_root == tmp_path / 'plots'
    assert paths.results_root == tmp_path / 'results'
    assert paths.datasets_root == tmp_path / 'datasets'
    assert paths.monte_carlo_results == tmp_path / 'results' / 'montecarlo_results'
    assert paths.bayesian_results == tmp_path / 'results' / 'bayesian-hierarchical-results'
    assert paths.team_aggregation_plots == tmp_path / 'plots' / 'team_aggregation'
    assert paths.test_runs_plots == tmp_path / 'plots' / 'test_runs'


def test_setup_logger_and_exception_formatting(monkeypatch):
    monkeypatch.setenv('LOG_LEVEL', 'DEBUG')

    logger = setup_logger('ffbayes.test.logger')
    assert logger.level == 10
    assert logger.handlers

    message = handle_exception(ValueError('bad value'), context='UnitTest')
    assert message == '[UnitTest] ValueError: bad value'
