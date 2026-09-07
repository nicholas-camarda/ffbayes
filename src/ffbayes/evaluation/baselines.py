"""Fair, legal, reproducible draft baseline simulations.

This harness compares policies under one player universe, one profile, one
snake order, and one legal-pick wrapper.  It is deliberately a decision
baseline harness; it does not fabricate historical as-of inputs or estimate
championship probability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ffbayes.draft_2026.decision.lineup import optimize_lineup
from ffbayes.draft_2026.decision.roster_value import calculate_marginal_roster_value
from ffbayes.draft_2026.engine import build_draft_board, score_projections
from ffbayes.draft_2026.league import LeagueProfile
from ffbayes.draft_2026.rules import (
    RosterRulesError,
    league_can_complete,
    roster_is_feasible,
    rules_for_profile,
    validate_complete_roster,
)
from ffbayes.evaluation import static_v1
from ffbayes.evaluation.rolling import evaluation_manifest, validate_as_of_inputs

BASELINE_EVALUATION_VERSION = 'fair-baselines-v1'


class EvaluationInputError(ValueError):
    """Raised when a fair evaluation input contract is incomplete."""


@dataclass(frozen=True)
class DraftPick:
    overall_pick: int
    team_index: int
    player_id: int
    policy: str


@dataclass(frozen=True)
class DraftSimulation:
    policy: str
    seed: int
    rosters: tuple[tuple[int, tuple[int, ...]], ...]
    picks: tuple[DraftPick, ...]
    metrics: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            'policy': self.policy,
            'seed': self.seed,
            'rosters': {
                str(team): list(player_ids) for team, player_ids in self.rosters
            },
            'picks': [
                {
                    'overall_pick': pick.overall_pick,
                    'team_index': pick.team_index,
                    'player_id': pick.player_id,
                    'policy': pick.policy,
                }
                for pick in self.picks
            ],
            'metrics': dict(self.metrics),
        }


@dataclass(frozen=True)
class EvaluationResult:
    manifest: Mapping[str, Any]
    simulations: tuple[DraftSimulation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            'schema_version': BASELINE_EVALUATION_VERSION,
            'manifest': dict(self.manifest),
            'simulations': [simulation.to_dict() for simulation in self.simulations],
        }


Policy = Callable[[pd.DataFrame, pd.DataFrame, LeagueProfile, int, int], int]


def _validate_players(players: pd.DataFrame) -> pd.DataFrame:
    required = {'espn_id', 'position', 'projected_points', 'adp'}
    missing = sorted(required.difference(players.columns))
    if missing:
        raise EvaluationInputError(f'evaluation players are missing columns: {missing}')
    frame = players.copy()
    ids = pd.to_numeric(frame['espn_id'], errors='coerce')
    if (
        frame['espn_id'].map(lambda x: isinstance(x, (bool, np.bool_))).any()
        or ids.isna().any()
        or (~np.isfinite(ids)).any()
        or ids.duplicated().any()
        or (ids < 1).any()
        or (ids != ids.round()).any()
    ):
        raise EvaluationInputError(
            'evaluation players require unique positive integer espn_id values'
        )
    points = pd.to_numeric(frame['projected_points'], errors='coerce')
    adp = pd.to_numeric(frame['adp'], errors='coerce')
    if points.isna().any() or (~np.isfinite(points)).any() or (points < 0).any():
        raise EvaluationInputError('projected_points must be finite and non-negative')
    if adp.isna().any() or (~np.isfinite(adp)).any() or (adp <= 0).any():
        raise EvaluationInputError('adp must be finite and positive')
    frame['projected_points'] = points.astype(float)
    frame['adp'] = adp.astype(float)
    frame['espn_id'] = ids.astype(int)
    frame['position'] = frame['position'].astype(str).str.upper()
    return frame


def _snake_order(profile: LeagueProfile) -> tuple[int, ...]:
    return tuple(
        team
        for round_number in range(profile.active_roster_size())
        for team in (
            range(profile.team_count)
            if round_number % 2 == 0
            else range(profile.team_count - 1, -1, -1)
        )
    )


def _rank_ids(frame: pd.DataFrame, *, points: bool = False) -> list[int]:
    columns = ['projected_points', 'adp', 'espn_id'] if points else ['adp', 'espn_id']
    ascending = [False, True, True] if points else [True, True]
    return [
        int(value)
        for value in frame.sort_values(columns, ascending=ascending)['espn_id']
    ]


def _legacy_ffbayes_ids(players: pd.DataFrame, profile: LeagueProfile) -> list[int]:
    """Freeze the pre-roster-aware FFBayes rank as a named baseline.

    This intentionally duplicates the old static formula instead of calling
    the mutable live engine.  Its output is therefore a stable comparison
    when the live decision policy evolves.
    """
    if 'projection_stats' not in players:
        raise EvaluationInputError(
            'current FFBayes baseline requires raw projection_stats in the input'
        )
    try:
        scored = static_v1.score_projections(players, profile)
        replacement = static_v1.calculate_replacement_levels(scored, profile)
        frame = scored.copy()
        frame['replacement_level'] = (
            frame['position'].map(replacement['levels']).fillna(0.0)
        )
        frame['vor'] = frame['projected_points'] - frame['replacement_level']
        frame['scarcity'] = static_v1._position_scarcity(frame, replacement)
        value_scale = max(float(frame['vor'].std(ddof=0)), 1.0)
        scarcity_scale = max(float(frame['scarcity'].std(ddof=0)), 1.0)
        frame['value_signal'] = (
            frame['vor'] / value_scale + 0.25 * frame['scarcity'] / scarcity_scale
        )
        frame['model_rank'] = frame['value_signal'].rank(
            ascending=False, method='average'
        )
        frame['market_rank'] = frame['adp'].rank(ascending=True, method='average')
        frame['board_priority'] = 0.6 * frame['model_rank'] + 0.4 * frame['market_rank']
        board = frame.sort_values(['board_priority', 'vor', 'adp', 'espn_id'])
    except Exception as exc:
        raise EvaluationInputError(
            f'current FFBayes baseline could not build: {exc}'
        ) from exc
    return [int(value) for value in board['espn_id'].dropna()]


def _live_ffbayes_ids(
    players: pd.DataFrame,
    profile: LeagueProfile,
    *,
    roster: Sequence[int],
    taken: Sequence[int],
) -> list[int]:
    """Use the integrated marginal-utility board policy."""
    try:
        board = build_draft_board(players, profile, taken_ids=taken, your_ids=roster)
    except Exception as exc:
        raise EvaluationInputError(
            f'live FFBayes baseline could not build: {exc}'
        ) from exc
    ranked = board.loc[board['is_available'] & board.get('is_draftable', True)]
    return [int(value) for value in ranked['espn_id'].dropna()]


def _legal_pick(
    ranked_ids: Iterable[int],
    *,
    team_roster: Sequence[int],
    available: pd.DataFrame,
    profile: LeagueProfile,
    all_rosters: Sequence[Sequence[int]],
    team_index: int,
) -> int:
    rules = rules_for_profile(profile)
    by_id = available.set_index('espn_id')
    # Roster players have already been removed from ``available``.  The caller
    # provides a complete position map through this private attribute.
    all_positions = available.attrs.get('all_positions', {})
    roster_positions = [str(all_positions[player_id]) for player_id in team_roster]
    for raw_id in ranked_ids:
        player_id = int(raw_id)
        if player_id not in by_id.index:
            continue
        candidate_position = str(by_id.loc[player_id, 'position'])
        if candidate_position not in rules.positions:
            continue
        future = by_id.drop(index=player_id)
        future_positions = [str(value) for value in future['position']]
        team_feasible = roster_is_feasible(
            roster_positions + [candidate_position], future_positions, rules
        )
        if not team_feasible:
            continue
        proposed = [list(roster) for roster in all_rosters]
        proposed[team_index].append(player_id)
        global_feasible = league_can_complete(
            [[str(all_positions[value]) for value in roster] for roster in proposed],
            future_positions,
            rules,
        )
        if global_feasible:
            return player_id
    raise EvaluationInputError(
        f'no legal candidate remains for team roster {list(team_roster)}'
    )


def simulate_snake_draft(
    players: pd.DataFrame,
    profile: LeagueProfile,
    *,
    policy: str = 'adp',
    seed: int = 0,
    focal_team: int | None = None,
    opponent_policy: str = 'adp',
) -> DraftSimulation:
    """Run one deterministic legal snake draft for a named baseline policy."""
    frame = _validate_players(players)
    if 'projection_stats' in frame:
        scored = score_projections(frame, profile)
        if not np.allclose(
            frame.projected_points, scored.projected_points, rtol=1e-9, atol=1e-8
        ):
            raise EvaluationInputError(
                'projected_points disagree with profile-scored projection_stats'
            )
        frame = scored
    rules = rules_for_profile(profile)
    if len(frame) < profile.total_draft_picks():
        raise EvaluationInputError('player universe is too small for all roster slots')
    all_positions = {
        int(row['espn_id']): str(row['position']) for _, row in frame.iterrows()
    }
    available = frame.copy()
    available.attrs['all_positions'] = all_positions
    rosters: list[list[int]] = [[] for _ in range(profile.team_count)]
    picks: list[DraftPick] = []
    if policy == 'adp':
        ranked = _rank_ids(frame)
    elif policy == 'provider_projection':
        ranked = _rank_ids(frame, points=True)
    elif policy in {'current_ffbayes', 'legacy_ffbayes'}:
        ranked = _legacy_ffbayes_ids(frame, profile)
    elif policy in {'ffbayes_live', 'roster_aware_vor'}:
        ranked = []
    else:
        raise EvaluationInputError(f'unknown baseline policy: {policy}')
    if opponent_policy not in {
        'adp',
        'provider_projection',
        'current_ffbayes',
        'legacy_ffbayes',
        'ffbayes_live',
        'roster_aware_vor',
    }:
        raise EvaluationInputError(f'unknown opponent policy: {opponent_policy}')
    if focal_team is not None and not 0 <= focal_team < profile.team_count:
        raise EvaluationInputError('focal_team must be within team_count')
    if focal_team is None:
        focal_team = (profile.draft_slot or 1) - 1

    for overall_pick, team_index in enumerate(_snake_order(profile), start=1):
        team_policy = policy if team_index == focal_team else opponent_policy
        if team_policy == 'roster_aware_vor':
            try:
                values = calculate_marginal_roster_value(
                    frame,
                    profile,
                    your_ids=rosters[team_index],
                    taken_ids=[
                        value
                        for other_index, team in enumerate(rosters)
                        if other_index != team_index
                        for value in team
                    ],
                )
            except ValueError as exc:
                raise EvaluationInputError(str(exc)) from exc
            eligible = available['espn_id'].astype(int)
            scored_values = values.loc[eligible].sort_values(
                ['roster_feasible', 'marginal_usable_points'], ascending=[False, False]
            )
            ranked = [int(value) for value in scored_values.index]
        elif team_policy == 'ffbayes_live':
            ranked = _live_ffbayes_ids(
                frame,
                profile,
                roster=rosters[team_index],
                taken=[
                    value
                    for other_index, team in enumerate(rosters)
                    if other_index != team_index
                    for value in team
                ],
            )
        elif team_policy in {'current_ffbayes', 'legacy_ffbayes'}:
            ranked = _legacy_ffbayes_ids(frame, profile)
        elif team_policy == 'adp':
            ranked = _rank_ids(frame)
        elif team_policy == 'provider_projection':
            ranked = _rank_ids(frame, points=True)
        else:
            raise EvaluationInputError(f'unknown baseline policy: {team_policy}')
        player_id = _legal_pick(
            ranked,
            team_roster=rosters[team_index],
            available=available,
            profile=profile,
            all_rosters=rosters,
            team_index=team_index,
        )
        rosters[team_index].append(player_id)
        available = available[available['espn_id'].ne(player_id)].copy()
        available.attrs['all_positions'] = all_positions
        picks.append(DraftPick(overall_pick, team_index, player_id, team_policy))

    for roster in rosters:
        try:
            validate_complete_roster(
                [all_positions[player_id] for player_id in roster], rules
            )
        except RosterRulesError as exc:
            raise EvaluationInputError(
                f'policy {policy} produced an illegal roster: {exc}'
            ) from exc
    full = frame.set_index('espn_id')
    metrics = {
        'total_projected_points': float(
            sum(
                float(full.loc[player_id, 'projected_points'])
                for roster in rosters
                for player_id in roster
            )
        ),
        'total_starter_points': float(
            sum(
                optimize_lineup(full.loc[roster].reset_index(), profile).total_points
                for roster in rosters
            )
        ),
        'mean_starter_points': float(
            np.mean(
                [
                    optimize_lineup(
                        full.loc[roster].reset_index(), profile
                    ).total_points
                    for roster in rosters
                ]
            )
        ),
        'focal_team': float(focal_team),
        'focal_projected_points': float(
            sum(
                float(full.loc[player_id, 'projected_points'])
                for player_id in rosters[focal_team]
            )
        ),
        'focal_starter_points': float(
            optimize_lineup(
                full.loc[rosters[focal_team]].reset_index(), profile
            ).total_points
        ),
        'opponent_mean_starter_points': float(
            np.mean(
                [
                    optimize_lineup(
                        full.loc[roster].reset_index(), profile
                    ).total_points
                    for index, roster in enumerate(rosters)
                    if index != focal_team
                ]
            )
        )
        if profile.team_count > 1
        else 0.0,
    }
    return DraftSimulation(
        policy=policy,
        seed=seed,
        rosters=tuple((index, tuple(roster)) for index, roster in enumerate(rosters)),
        picks=tuple(picks),
        metrics=metrics,
    )


def evaluate_baselines(
    players: pd.DataFrame,
    profile: LeagueProfile,
    *,
    policies: Sequence[str] = (
        'current_ffbayes',
        'ffbayes_live',
        'provider_projection',
        'adp',
        'roster_aware_vor',
    ),
    seed: int = 0,
    cutoff: str | None = None,
    source_manifests: Iterable[Mapping[str, Any]] = (),
    focal_team: int | None = None,
    opponent_policy: str = 'adp',
    synthetic: bool = False,
) -> EvaluationResult:
    """Evaluate named baselines on the same legal draft scenario."""
    frame = _validate_players(players)
    if not synthetic and not cutoff:
        raise EvaluationInputError(
            'as-of evaluation requires cutoff; use explicit synthetic mode for fixtures'
        )
    if synthetic and (cutoff or source_manifests):
        raise EvaluationInputError(
            'synthetic mode cannot claim as-of source provenance'
        )
    if cutoff:
        if (
            'projection_season' not in frame
            or not frame.projection_season.eq(profile.season).all()
        ):
            raise EvaluationInputError(
                'as-of player rows require matching projection_season'
            )
        validate_as_of_inputs(frame.to_dict(orient='records'), cutoff)
    source_values = tuple(dict(value) for value in source_manifests)
    if cutoff and not source_values:
        raise EvaluationInputError(
            'verified as-of evaluation requires source_manifests with timestamps and hashes'
        )
    manifest = evaluation_manifest(
        cutoff=cutoff,
        season=profile.season,
        source_manifests=source_values,
        rows=frame.to_dict(orient='records'),
        profile=profile.to_dict(),
        seed=seed,
        version=BASELINE_EVALUATION_VERSION,
    )
    import hashlib

    code_root = Path(__file__).resolve().parents[1]
    code_digest = hashlib.sha256()
    for code_path in sorted(code_root.rglob('*.py')):
        code_digest.update(str(code_path.relative_to(code_root)).encode())
        code_digest.update(code_path.read_bytes())
    manifest.update(
        {
            'code_sha256': code_digest.hexdigest(),
            'policies': list(policies),
            'focal_team': focal_team
            if focal_team is not None
            else (profile.draft_slot or 1) - 1,
            'opponent_policy': opponent_policy,
            'metric_scope': 'projected_utility_only',
            'tie_break': 'stable_player_id',
            'seed_effect': 'reserved_no_random_sampling',
        }
    )
    simulations = tuple(
        simulate_snake_draft(
            frame,
            profile,
            policy=policy,
            seed=seed,
            focal_team=focal_team,
            opponent_policy=opponent_policy,
        )
        for policy in policies
    )
    return EvaluationResult(manifest=manifest, simulations=simulations)


def _load_json_input(
    path: Path,
) -> tuple[pd.DataFrame, LeagueProfile, list[dict[str, Any]], str | None]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, Mapping):
        raise EvaluationInputError('evaluation input must be a JSON object')
    profile_value = value.get('profile') or value.get('league_profile')
    if not isinstance(profile_value, Mapping):
        raise EvaluationInputError('evaluation input requires profile')
    profile = LeagueProfile.from_mapping(profile_value)
    rows = value.get('players')
    if not isinstance(rows, list) or not rows:
        raise EvaluationInputError('evaluation input requires a non-empty players list')
    manifests = value.get('source_manifests') or []
    if not isinstance(manifests, list):
        raise EvaluationInputError('source_manifests must be a list')
    return (
        pd.DataFrame(rows),
        profile,
        [dict(item) for item in manifests],
        value.get('cutoff'),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run fair legal FFBayes baseline evaluation.'
    )
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Explicitly label mechanics-only input without as-of provenance',
    )
    parser.add_argument('--policy', action='append')
    parser.add_argument('--focal-team', type=int)
    parser.add_argument('--opponent-policy', default='adp')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame, profile, manifests, cutoff = _load_json_input(args.input)
    result = evaluate_baselines(
        frame,
        profile,
        policies=tuple(
            args.policy
            or (
                'current_ffbayes',
                'ffbayes_live',
                'provider_projection',
                'adp',
                'roster_aware_vor',
            )
        ),
        seed=args.seed,
        cutoff=cutoff,
        source_manifests=manifests,
        focal_team=args.focal_team,
        opponent_policy=args.opponent_policy,
        synthetic=args.synthetic,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding='utf-8'
    )
    print(
        json.dumps(
            {
                'status': 'complete',
                'output': str(args.output),
                'version': BASELINE_EVALUATION_VERSION,
            }
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
