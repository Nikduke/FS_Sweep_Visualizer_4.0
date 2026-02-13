# plotly_relayout_listener

Client-side Streamlit component used by `fs_sweep_app_spline.py` to persist Plotly zoom
without rerun-per-zoom.

## Files

- `index.html`: minimal component host
- `listener.js`: binding + localStorage logic

## Input args (from Python)

- `data_id` (string): dataset identity key
- `plot_count` (int): number of Plotly charts to bind
- `plot_ids` (string[]): logical IDs in display order (`x`, `r`, `xr`, `rx`)
- `debounce_ms` (int): localStorage write debounce
- `nonce` (int): rerender/rebind trigger token
- `reset_token` (int): dataset/reset signal
- `bind_tries` (int), `bind_interval_ms` (int): bind retry loop parameters
- `ignore_autorange_ms` (int): mount autorange suppression window

## Behavior

1. Finds Streamlit plot containers and their `div.js-plotly-plot`.
2. Binds `plotly_relayout` handlers by chart index.
3. Stores x/y ranges in localStorage key:
   - `fsSweepZoom:{data_id}:{plot_id}`
4. On component render with changed `reset_token`, clears all keys for:
   - `fsSweepZoom:{data_id}:*`
5. Applies stored zoom on bind via `Plotly.relayout`.
6. Ignores initial autorange-only relayout events shortly after bind.

## Notes

- This component does not push zoom payloads back to Python.
- Persistence is client-side only.
