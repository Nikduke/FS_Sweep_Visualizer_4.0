# FS Sweep Visualizer (Spline) - Current Implementation

This document reflects:
- `fs_sweep_app_spline.py`
- `preselection_shortlist.py`
- `plotly_selection_bridge/listener.js`
- `plotly_selection_bridge/selection_table_module.js`
- `plotly_export_button/listener.js`
- `plotly_rx_toolbar/listener.js`

## Scope

The app renders:
- `X`, `R`, `X/R` line plots
- optional `R vs X` scatter

Core design:
- Streamlit reruns for context changes (data/sequence/location/layout/base frequency).
- JS direct restyling for high-frequency interactions (case filters/selection/color/harmonics/method toggles).
- JS-heavy UI helpers are externalized into dedicated component assets (no large inline JS blocks in Python).
- deterministic preselection metrics are computed server-side once per context and consumed client-side without rerun.
- when enabled, `R vs X` scatter is rendered before line plots in the main area.
- data-scoped Streamlit cache keys are pruned on file/context changes to avoid stale-session growth.
- JS widget state/API stores are pruned to the active `{data_id}|{chart_id}` context.
- selected-location preselection payload is cached in session state by:
  - `preselection_payload:{data_id}:{seq_label}:{location}` (single active base payload, overwritten on base switch).
  - payload sent to JS uses compact array/index format (`compact_v1`) to reduce rerun transfer size.
- on chart-context switches (`base frequency`, `location`, `Positive/Zero` sequence), heavy sequence caches are evicted before rebuild to reduce peak-memory spikes on constrained hosts.
- non-active sequence/location/chart cache entries are pruned proactively in the same data session.
- uploaded workbooks are parsed once, converted to compressed NPZ, and cached in session state as:
  - `uploaded_data:{data_id}:npz`
  - reruns load from NPZ (not XLSX re-parse) for the same upload id.
- raw XLSX bytes are used only for conversion step and are not kept as active runtime data.
- sheet value columns are converted to `float32` during load to reduce in-memory footprint on large datasets.

## Sidebar Layout (current)

1. `Data Source`
2. `Controls`
   - plot height
   - width mode/figure width
   - spline toggle/smoothing
3. `Show plots`
   - checkbox order: `R vs X scatter`, `X`, `R`, `X/R`
   - for enabled line plots (`X`, `R`, `X/R`), an `Export` button appears on the right side of the same row
   - no sidebar legend-width controls (width is internal)
4. `Case Filters & Selection`
   - `Sequence` (`Positive`/`Zero`)
   - `Base frequency` (`50 Hz`/`60 Hz`) (Streamlit-side, rerun)
   - `Location`
   - JS widget panel

## JS Widget (no rerun)

Rendered by `plotly_selection_bridge`:
- case-part filters (excluding location)
  - color dots are shown to the right of case-part values for the active color grouping
  - works for explicit `Case part N` color mode and for `Auto` (uses auto-selected hue part)
- color mode (`Auto`/by case part)
- selection controls:
  - `Clear list` / `Download selected CSV` are shown above the selected-cases table in the widget panel
  - `Show only selected sweeps` stays in the scatter control row when scatter is shown
  - when scatter is hidden: `Show only selected sweeps` is shown in the widget panel
  - add list (paste/import)
  - remove selected rows (`Del`)
  - selected CSV download
- harmonics controls:
  - show harmonic lines
  - bin width (Hz)
  - harmonic/bin guide lines are generated from full baseline harmonic range (not from current zoom window)
- method controls (in scatter toolbar row):
  - `Energinet` toggle + editable `T2/T3/T4` + `Top N`
  - `IEC` toggle + `Top N`
  - `Include collinear boundary (+N)` (IEC-dependent; disabled/off when IEC is off)
    - `+N` shows current additional visible cases vs vertices-only mode
  - method toggles append/remove method-sourced candidates from selection state without rerun
  - `Top N` is method-specific (`0` => all candidates for that method)

## Visibility And Legend Rules

Layer 1 (case-part filters):
- filtered-out cases are hidden in lines and scatter
- filtered-out cases are hidden from legends

Layer 2 (selection):
- default: non-selected visible cases are dimmed
- `Show only selected sweeps`: non-selected visible cases are hidden in line plots
- when selection is non-empty: line legends show selected cases only
- scatter keeps allowed points visible for continued picking

Selection is sticky across case-part changes:
- selected cases hidden by filters remain in selection state
- they reappear when filters include them again

Line legend appearance:
- line traces are rendered in line-only mode, so legend swatches are line-only (no dot marker symbol).

## Scatter Behavior (`R vs X`)

- one scatter trace is rendered; scatter frequency state is controlled client-side.
- in-plot frequency slider remains visible.
  - slider steps are control signals (`method='skip'`), not Plotly animation frames.
  - JS reads `layout.meta.rx_single_trace` and restyles trace `x/y` for selected frequency.
- `Prev frequency` / `Next frequency` buttons:
  - call shared selection API (`stepRxFrequency`) for no-rerun stepping.
  - legacy frame animate path remains as fallback for compatibility.
- direct frequency entry:
  - scatter toolbar includes `Set f (Hz)` field + `Set` button.
  - entered value snaps to the nearest available frequency step without rerun.
- scatter status line:
  - `R vs X points shown: <count> | Frequency steps: <steps>`
  - `<count>` is case-filtered visible point count.
- modebar `Reset axes` returns to location-based baseline bounds.
- click selection toggles by case id and feeds shared JS state.
- line plots (`X`, `R`, `X/R`) also support click selection by case id and feed the same shared JS selection state.
- selected point styling:
  - selected visible points use symbol `diamond`
  - non-selected points keep `circle`
  - dimming behavior is unchanged.
- hover:
  - shows case name, `R`, and `X`.
  - frequency is shown by slider current value and title.

## Zoom Behavior

- no custom zoom persistence bridge is used.
- zoom/pan/reset behavior is native Plotly behavior.
- figure `uirevision` persistence is not forced by app code.
- line-plot x scaling follows Streamlit base-frequency control (`f_base`) on rerun.
- Energinet threshold defaults are base-frequency specific:
  - `50 Hz`: `400/600/2400`
  - `60 Hz`: `450/800/3000`

## Export Behavior

- on-page line-plot legends are rendered below plots (horizontal) with fixed reserved legend area.
- legend column width is auto-calculated internally (not user-configured in sidebar).
- modebar export remains available.
- line full-legend export (buttons in `Show plots`) reads current line visibility/style state.
- hidden cases are excluded from export legend.
