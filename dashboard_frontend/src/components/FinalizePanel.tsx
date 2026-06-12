import type { DashboardPayload } from '../payload/load';
import {
  buildBundle,
  buildFinalizeFilenames,
  downloadTextFile,
  isLocalFinalizeSupported,
  isRosterComplete,
} from '../finalize/buildBundle';
import { buildBoardState } from '../state/buildBoardState';
import type { DraftStore } from '../state/draftState';

export function FinalizePanel(props: { payload: DashboardPayload; store: DraftStore }) {
  const { payload, store } = props;
  const showFinalize = isLocalFinalizeSupported();

  const handleFinalize = () => {
    if (!showFinalize) {
      return;
    }
    const currentState = store.getState();
    const boardState = buildBoardState(payload, currentState);
    if (!isRosterComplete(boardState, currentState)) {
      const confirmed = window.confirm(
        'Your roster is not full yet. Download a finalized draft snapshot anyway?',
      );
      if (!confirmed) {
        return;
      }
    }
    const bundle = buildBundle(currentState, payload);
    const filenames = buildFinalizeFilenames(payload);
    downloadTextFile(filenames.json, JSON.stringify(bundle.json, null, 2), 'application/json');
    downloadTextFile(filenames.snapshotHtml, bundle.snapshotHtml, 'text/html');
    downloadTextFile(filenames.summaryHtml, bundle.summaryHtml, 'text/html');
  };

  return (
    <>
      <div className="toolbar-row">
        <button type="button" id="finalize-button" hidden={!showFinalize} onClick={handleFinalize}>
          Finalize Draft
        </button>
      </div>
      <div className="tiny toolbar-note" id="finalize-note">
        {showFinalize
          ? 'Finalize downloads a locked HTML snapshot, JSON export, and post-draft summary from this local dashboard.'
          : 'Finalize downloads are only supported from the local generated dashboard opened as a file.'}
      </div>
    </>
  );
}
