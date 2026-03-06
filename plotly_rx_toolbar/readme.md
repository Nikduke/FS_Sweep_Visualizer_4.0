# plotly_rx_toolbar

Mini Streamlit component used by `fs_sweep_app_spline.py` to render the control row under `R vs X` scatter.

Responsibilities:
- `Prev frequency` / `Next frequency` stepping
  - primary path: shared bridge API `stepRxFrequency(delta)` (single-trace scatter restyle, no rerun)
  - compatibility fallback: Plotly frame animation if legacy frame-based scatter is present
- `Clear list`, `Download selected CSV`
- `Show only selected sweeps` checkbox synchronized with shared JS selection API
- method rows under main control row:
  - `Energinet` toggle with editable `T2/T3/T4` thresholds and `Top N`
  - `IEC` toggle with `Top N`
  - `Include collinear boundary (+N)` (IEC-dependent; disabled/off when IEC is off)
    - `+N` is the current additional visible-case count vs vertices-only mode
  - both operate via selection API and update selection without Streamlit rerun

The component consumes selection state/actions through `window.parent.__fsCaseUiApi`.
