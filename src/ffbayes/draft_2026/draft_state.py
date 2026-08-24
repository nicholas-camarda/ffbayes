"""Immutable-style live draft state transitions.

The dashboard keeps one instance of :class:`DraftState` per league.  A state
transition returns a new value rather than mutating the previous one, which
makes it possible for the service to validate and commit a complete action
atomically.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

Disposition = Literal['mine', 'taken']


def _positive_integer(value: object, field: str) -> int:
    """Validate a positive integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{field} must be a positive integer')
    return value


@dataclass(frozen=True, slots=True)
class DraftAction:
    """One recorded selection in the ordered draft history."""

    pick: int
    player_id: int
    disposition: Disposition

    def __post_init__(self) -> None:
        _positive_integer(self.pick, 'pick')
        _positive_integer(self.player_id, 'player_id')
        if self.disposition not in ('mine', 'taken'):
            raise ValueError("disposition must be 'mine' or 'taken'")


@dataclass(frozen=True, slots=True)
class DraftState:
    """Current live-draft clock, selections, and independent queue.

    ``current_pick`` is the next overall pick to record.  It starts at one,
    regardless of the user's draft slot.  The state deliberately does not
    know the league's total pick count; the profile/service boundary validates
    that clock against a particular league.
    """

    draft_slot: int | None
    current_pick: int = 1
    actions: tuple[DraftAction, ...] = ()
    queue_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.draft_slot is not None:
            _positive_integer(self.draft_slot, 'draft_slot')
        _positive_integer(self.current_pick, 'current_pick')

        if not isinstance(self.actions, tuple):
            raise ValueError('actions must be an immutable tuple')
        action_ids: list[int] = []
        for action in self.actions:
            if not isinstance(action, DraftAction):
                raise ValueError('actions must contain DraftAction values')
            if action.player_id in action_ids:
                raise ValueError('actions cannot record a player more than once')
            action_ids.append(action.player_id)

        if not isinstance(self.queue_ids, tuple):
            raise ValueError('queue_ids must be an immutable tuple')
        queue_ids: list[int] = []
        for player_id in self.queue_ids:
            _positive_integer(player_id, 'queue player_id')
            if player_id in queue_ids:
                raise ValueError('queue_ids cannot contain duplicates')
            queue_ids.append(player_id)

    @property
    def taken_ids(self) -> tuple[int, ...]:
        """Stable IDs of selections recorded as unavailable to the user."""
        return tuple(
            action.player_id
            for action in self.actions
            if action.disposition == 'taken'
        )

    @property
    def your_ids(self) -> tuple[int, ...]:
        """Stable IDs of the user's recorded roster selections."""
        return tuple(
            action.player_id
            for action in self.actions
            if action.disposition == 'mine'
        )

    def record(self, player_id: int, disposition: Disposition) -> 'DraftState':
        """Record or correct a player selection.

        A new player consumes exactly the current pick.  A player already in
        the history is a disposition correction and therefore leaves the
        clock unchanged.  Repeating the same disposition is idempotent.
        """
        _positive_integer(player_id, 'player_id')
        if disposition not in ('mine', 'taken'):
            raise ValueError("disposition must be 'mine' or 'taken'")

        for index, action in enumerate(self.actions):
            if action.player_id != player_id:
                continue
            if action.disposition == disposition:
                return self
            corrected = DraftAction(action.pick, player_id, disposition)
            actions = self.actions[:index] + (corrected,) + self.actions[index + 1 :]
            return replace(self, actions=actions)

        action = DraftAction(self.current_pick, player_id, disposition)
        return replace(
            self,
            current_pick=self.current_pick + 1,
            actions=self.actions + (action,),
        )

    def toggle_queue(self, player_id: int) -> 'DraftState':
        """Add/remove a player from the queue without changing draft state."""
        _positive_integer(player_id, 'player_id')
        if player_id in self.queue_ids:
            queue_ids = tuple(value for value in self.queue_ids if value != player_id)
        else:
            queue_ids = self.queue_ids + (player_id,)
        return replace(self, queue_ids=queue_ids)

    def undo(self) -> 'DraftState':
        """Remove the latest recorded selection and restore its consumed pick."""
        if not self.actions:
            return self
        latest = self.actions[-1]
        return replace(
            self,
            current_pick=latest.pick,
            actions=self.actions[:-1],
        )

    def sync_clock(self, current_pick: int) -> 'DraftState':
        """Manually set the clock while preserving all recorded actions."""
        _positive_integer(current_pick, 'current_pick')
        return replace(self, current_pick=current_pick)
