from __future__ import annotations

from ffbayes.draft_2026.draft_state import DraftAction, DraftState


def test_new_state_starts_at_overall_pick_one_independent_of_draft_slot() -> None:
    state = DraftState(draft_slot=10, current_pick=1)

    assert state.current_pick == 1
    assert state.taken_ids == ()
    assert state.your_ids == ()


def test_record_taken_advances_once_and_marks_player_unavailable() -> None:
    state = DraftState(draft_slot=10, current_pick=1)

    updated = state.record(101, 'taken')

    assert updated.current_pick == 2
    assert updated.actions == (DraftAction(1, 101, 'taken'),)
    assert updated.taken_ids == (101,)
    assert updated.your_ids == ()


def test_record_mine_advances_and_derives_my_roster() -> None:
    state = DraftState(draft_slot=3, current_pick=7)

    updated = state.record(202, 'mine')

    assert updated.current_pick == 8
    assert updated.your_ids == (202,)
    assert updated.taken_ids == ()


def test_correcting_existing_disposition_does_not_consume_another_pick() -> None:
    state = DraftState(draft_slot=3, current_pick=1).record(303, 'taken')

    corrected = state.record(303, 'mine')

    assert corrected.current_pick == 2
    assert corrected.actions == (DraftAction(1, 303, 'mine'),)
    assert corrected.your_ids == (303,)
    assert corrected.taken_ids == ()


def test_repeating_same_record_is_idempotent() -> None:
    state = DraftState(draft_slot=3, current_pick=1).record(404, 'taken')

    repeated = state.record(404, 'taken')

    assert repeated == state


def test_queue_is_independent_of_draft_actions() -> None:
    state = DraftState(draft_slot=3, current_pick=4)

    queued = state.toggle_queue(505)
    unqueued = queued.toggle_queue(505)

    assert queued.current_pick == 4
    assert queued.actions == ()
    assert queued.queue_ids == (505,)
    assert unqueued == state


def test_undo_removes_latest_action_restores_clock_and_roster() -> None:
    state = (
        DraftState(draft_slot=10, current_pick=1)
        .record(606, 'taken')
        .record(707, 'mine')
    )

    undone = state.undo()

    assert undone.current_pick == 2
    assert undone.actions == (DraftAction(1, 606, 'taken'),)
    assert undone.your_ids == ()
    assert undone.taken_ids == (606,)

    assert undone.undo().current_pick == 1
    assert undone.undo().actions == ()


def test_manual_clock_sync_does_not_change_action_history() -> None:
    state = DraftState(draft_slot=4, current_pick=1).record(808, 'mine')

    synced = state.sync_clock(19)

    assert synced.current_pick == 19
    assert synced.actions == state.actions
    assert synced.your_ids == state.your_ids
