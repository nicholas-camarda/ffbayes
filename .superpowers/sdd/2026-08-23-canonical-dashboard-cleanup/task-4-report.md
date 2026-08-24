# Task 4 report: hostile-input-safe dashboard row rendering

## Scope

The canonical dashboard row renderer in `src/ffbayes/draft_2026/dashboard_app.py` no longer interpolates validated payload fields through row `innerHTML`. Rows and cells are built with DOM APIs; player-facing values use `textContent`, player IDs and action names use `dataset`, and Taken/Mine/Queue controls are explicit buttons. Status CSS classes are selected from the existing available/taken/mine values. Service payload validation, state controls, league switching, snapshots, and arbitrary configured league behavior were left unchanged.

Focused hostile fixtures are in `tests/serve_draft_2026_hostile_fixture.py` and `tests/test_draft_2026_dashboard_hostile.mjs`. The fixture supplies a player name containing an image `onerror` payload and a recommendation containing an injected button `onclick` payload while keeping the payload board digest consistent.

## TDD and falsification evidence

The focused browser test was added before the production renderer change. The vulnerable renderer was exercised as the RED condition:

```text
$ node tests/test_draft_2026_dashboard_hostile.mjs
file:///Users/ncamarda/Workspaces/ffbayes/worktrees/cleanup-canonical-dashboard/tests/test_draft_2026_dashboard_hostile.mjs:38
    throw new Error('Hostile player name was not rendered as literal text');
          ^

Error: Hostile player name was not rendered as literal text
    at file:///Users/ncamarda/Workspaces/ffbayes/worktrees/cleanup-canonical-dashboard/tests/test_draft_2026_dashboard_hostile.mjs:38:11

Node.js v26.4.0
exit_code=1
```

For the required guard falsification, a temporary uncommitted patch restored the original `tr.innerHTML` interpolation. The same command failed with the output above, demonstrating that the hostile test detects parsing/execution risk. That vulnerable patch was not committed; the safe renderer was restored before verification and commit.

Safe GREEN verification:

```text
$ node tests/test_draft_2026_dashboard_hostile.mjs
2026 hostile dashboard rendering test passed
exit_code=0

$ python -m pytest -q tests/test_draft_2026_dashboard_service.py
.....                                                                    [100%]
5 passed in 0.50s
exit_code=0

$ node tests/test_draft_2026_dashboard_browser.mjs
2026 interactive dashboard browser smoke passed
exit_code=0
```

The hostile test asserts that the injected name and recommendation remain literal row text, that no injected image or `data-action="evil"` button is created, and that the hostile event-handler flag is not set. The canonical smoke continues to exercise recalculation, Taken state, independent league switching, invalid-slot handling, and snapshot export.

Additional final check:

```text
$ git diff --check
exit_code=0
```
