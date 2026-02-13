# plotly_selection_bridge

Client-side Streamlit component used by `fs_sweep_app_spline.py` for staged `R vs X`
point selection without rerun-per-click.

## Files

- `index.html`: minimal component host
- `listener.js`: click staging, visual highlight, commit handshake

## Input args (from Python)

- `data_id` (string)
- `plot_index` (int): index of target scatter chart among visible Streamlit Plotly charts
- `plot_id` (string): semantic ID (currently passed as `rx`)
- `chart_id` (string): chart namespace key (e.g., `rx:Positive`)
- `commit_token` (int): increments when user clicks `Show list selection`
- `commit_applied` (int): last commit token already applied by Python
- `clear_token` (int): increments when user clicks `Clear list`
- `reset_token` (int): upload/reset signal, forces staged selection clear
- `allowed_cases` (string[]): filter-bounded base case names
- `selected_marker_size` (float)
- `unselected_marker_opacity` (float)

## Behavior

1. Binds to scatter by `plot_index`.
2. On point click:
   - toggles case in staged set
   - persists staged set to localStorage
   - updates selected/unselected marker visuals using `selectedpoints`
3. Prunes staged set to `allowed_cases`.
4. On changed `clear_token` or `reset_token`, clears staged set.
5. On `commit_token > commit_applied` and unseen token, sends to Python:
   - `{commit_token, selected_cases, nonce}`

## localStorage keys

- staged selection: `fsSweepSelBridge:{data_id}:{chart_id}`
- last sent commit token: `fsSweepSelBridgeCommit:{data_id}:{chart_id}`

## Notes

- Staging is client-side; Python state updates only on explicit commit.
- This component intentionally avoids rerun on each click.
