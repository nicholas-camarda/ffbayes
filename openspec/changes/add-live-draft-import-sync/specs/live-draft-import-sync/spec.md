## ADDED Requirements

### Requirement: ESPN live draft sync SHALL consume normalized userscript pick events
The live dashboard SHALL support an ESPN-specific live sync path that consumes normalized pick events emitted by a desktop-browser userscript running in the ESPN live draft room.

#### Scenario: Supported ESPN userscript event arrives
- **WHEN** the local sync bridge receives a valid `espn_live_pick_v1` event from the repo-owned ESPN userscript
- **THEN** the dashboard SHALL process that event without requiring a page reload

#### Scenario: Unsupported or malformed event is rejected safely
- **WHEN** the local sync bridge receives an unsupported schema version or malformed event payload
- **THEN** the dashboard SHALL reject the event without mutating the existing board state
- **AND** the dashboard SHALL surface a visible warning that the sync input was unsupported

### Requirement: ESPN ownership detection SHALL use local identity matching
The live dashboard SHALL require local ESPN identity configuration so synced picks can be classified as `mine`, `other`, or `unknown`.

#### Scenario: Team label matches local identity
- **WHEN** an incoming ESPN event matches the configured draft-room team label
- **THEN** the dashboard SHALL classify that pick as `mine`

#### Scenario: Team label is unavailable and username matches
- **WHEN** draft-room team label matching is unavailable or unstable but the configured ESPN username matches confidently
- **THEN** the dashboard SHALL classify that pick as `mine`

#### Scenario: Ownership cannot be matched confidently
- **WHEN** neither team label nor username can classify ownership confidently
- **THEN** the dashboard SHALL classify the pick as `unknown`
- **AND** the dashboard SHALL surface an ownership warning rather than silently assuming `mine`

### Requirement: ESPN live sync SHALL reconcile into local board state conservatively
The live dashboard SHALL reconcile ESPN live events into local board state for `taken`, `mine`, and current pick number while preserving local queue and manual controls.

#### Scenario: Opponent pick arrives
- **WHEN** a valid event is classified as `other`
- **THEN** the dashboard SHALL mark the player as `taken`

#### Scenario: Owned pick arrives
- **WHEN** a valid event is classified as `mine`
- **THEN** the dashboard SHALL mark the player as `mine`
- **AND** the player SHALL also be treated as `taken`

#### Scenario: Unknown ownership arrives
- **WHEN** a valid event is classified as `unknown`
- **THEN** the dashboard SHALL mark the player as `taken`
- **AND** the dashboard SHALL preserve a visible ownership warning

#### Scenario: Queue remains local-only
- **WHEN** live sync events are applied
- **THEN** the dashboard SHALL NOT overwrite local queue state in v1

#### Scenario: Current pick advances monotonically
- **WHEN** an incoming event provides a valid pick number greater than or equal to the current synced pick
- **THEN** the dashboard SHALL advance current pick state monotonically
- **AND** it SHALL ignore non-monotone regressions

### Requirement: Manual dashboard operation SHALL remain available during sync failures
The live dashboard SHALL remain fully usable through the current manual workflow even if ESPN sync is missing, stale, disconnected, unsupported, or broken.

#### Scenario: Sync is absent
- **WHEN** ESPN sync has not been configured or started
- **THEN** the dashboard SHALL continue to operate from local browser state and manual actions

#### Scenario: Sync becomes stale or disconnected mid-draft
- **WHEN** the userscript, bridge, parser, or identity detection fails during a live draft
- **THEN** the operator SHALL be able to continue the draft entirely through the existing manual workflow without reload or state loss
- **AND** `Taken`, `Mine`, `Queue`, undo/redo, and finalize SHALL remain available

#### Scenario: Synced updates and manual edits coexist
- **WHEN** synced updates have already been applied and the operator continues manually
- **THEN** the dashboard SHALL preserve the current board state and manual editability rather than locking the session

### Requirement: Sync health and trust boundaries SHALL be visible in the dashboard
The live dashboard SHALL expose source, freshness, and failure state for ESPN live sync so the operator can judge whether to rely on it.

#### Scenario: Live sync is healthy
- **WHEN** ESPN events are arriving normally
- **THEN** the dashboard SHALL label the source as `ESPN userscript`
- **AND** it SHALL show a `live` sync state with last-update timestamp and imported pick count

#### Scenario: Sync is stale or disconnected
- **WHEN** ESPN events stop arriving beyond the configured freshness window or the bridge disconnects
- **THEN** the dashboard SHALL show a `stale` or `disconnected` state prominently
- **AND** it SHALL make manual fallback availability explicit

#### Scenario: ESPN page is unsupported
- **WHEN** the userscript cannot recognize the ESPN draft-room page structure
- **THEN** the dashboard SHALL show an `unsupported page` warning rather than pretending sync is active

#### Scenario: Operator pauses sync
- **WHEN** the operator explicitly pauses or disconnects ESPN sync
- **THEN** the dashboard SHALL stop applying incoming synced updates
- **AND** it SHALL remain fully usable through manual controls

### Requirement: Sync batches SHALL remain undoable
The live dashboard SHALL preserve undo/redo semantics after synced updates.

#### Scenario: Batch of live events is applied
- **WHEN** one or more synced updates are reconciled into the dashboard
- **THEN** the dashboard SHALL record that reconciliation as an undoable transaction

#### Scenario: Operator undoes after sync
- **WHEN** the operator uses undo after a synced batch
- **THEN** the dashboard SHALL restore the pre-batch board state consistently
- **AND** later redo SHALL still behave correctly
