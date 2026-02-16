# plotly_selection_bridge

Client-side control bridge for case filtering, selection, color styling, harmonics overlays, and scatter status updates.

## Files

- `index.html`
- `selection_table_module.js` (selection table helpers + selection API bridge)
- `listener.js` (plot restyling, panel rendering, scheduler, event wiring)

Related sibling components (same repo):
- `../plotly_export_button/*` (line-plot full-legend export button UI/logic)
- `../plotly_rx_toolbar/*` (scatter prev/next + selection toolbar row)

## Input args

- `data_id` (string)
- `chart_id` (string)
- `plot_ids` (string[]): visible plot order as rendered in Streamlit (current default starts with `rx`, then line plots)
- `cases_meta` (object[]):
  - `case_id` (string)
  - `display_case` (string)
  - `parts` (string[])
- `part_labels` (string[])
- `color_by_options` (string[])
- `color_maps` (object): `{option -> {case_id -> color_hex}}`
- `base_frequency_options` (number[]): supported base frequencies (current `[50, 60]`)
- `base_frequency_default` (number)
- `auto_color_part_label` (string): case-part label used when `Color=Auto` (for filter color dots)
- `color_by_default` (string)
- `show_only_default` (bool)
- `selected_marker_size` (float)
- `dim_marker_opacity` (float)
- `selected_line_width` (float)
- `dim_line_width` (float)
- `dim_line_opacity` (float)
- `dim_line_color` (string)
- `f_base` (float)
- `n_min` (float)
- `n_max` (float)
- `show_harmonics_default` (bool)
- `bin_width_hz_default` (float)
- `rx_status_dom_id` (string): optional parent DOM id for scatter status text
- `rx_freq_steps` (int): fallback step count for status text
- `reset_token` (int)
- `selection_reset_token` (int)
- `render_nonce` (int)
- `enable_selection` (bool)
- `spline_enabled` (bool): hints JS re-apply schedule (longer tail when spline is enabled)

## Behavior

1. Renders control panel UI inside component iframe.
2. Computes allowed cases from case-part filters.
3. Applies selection style layer:
   - dim mode (default)
   - hide mode (`Show only selected sweeps`)
4. Restyles line plots (`x/r/xr`) without rerun:
   - applies JS base frequency (`x = frequency/base`) from trace `customdata`
   - updates `xaxis.range` for current base
   - `visible`, `showlegend`, line color/width/opacity
   - when selection exists, legend entries are limited to selected cases
5. Applies harmonics overlays on line plots from JS controls.
6. Restyles scatter points (`rx`) without rerun.
   - scatter keeps allowed points visible for picking; `Show only selected sweeps` is line-plot-only.
7. Updates scatter status text in parent DOM (`rx_status_dom_id`) with case-filtered visible count.
8. Handles scatter click selection toggle.
9. Supports selection-table actions (clear/remove/import/csv).
10. Reapplies scatter styling on frequency animation events (`plotly_sliderchange`, `plotly_animated`) and frame-object refreshes to prevent selection flicker.
11. In case-part filters, shows color dots to the right of values for the active `Color` grouping.
    - if `Color=Auto`, uses `auto_color_part_label`.
12. Exposes selection control API at `window.parent.__fsCaseUiApi[{data_id}|{chart_id}]` for scatter-row controls.
13. Apply scheduler uses bounded multi-pass probes and stops early once expected plots are bound/stable.

## State model

- stored in `window.parent.__fsCaseUiStore`
- key: `{data_id}|{chart_id}`
- persists across normal reruns in same page session
- resets on `reset_token` change
- stale entries from older `data_id` values are pruned automatically

## Python roundtrip policy

- Normal interactions do not emit `streamlit:setComponentValue`.
- Updates are applied directly to existing Plotly DOM.
