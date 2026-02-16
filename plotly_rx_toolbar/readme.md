# plotly_rx_toolbar

Mini Streamlit component used by `fs_sweep_app_spline.py` to render the control row under `R vs X` scatter.

Responsibilities:
- `Prev frequency` / `Next frequency` frame stepping
- `Clear list`, `Download selected CSV`
- `Show only selected sweeps` checkbox synchronized with shared JS selection API
- Method rows under main control row:
  - `Energinet` toggle with editable `T2/T3/T4` thresholds and `Top N`
  - `IEC` toggle with `Top N`
  - `Include collinear boundary` (IEC-dependent; disabled/off when IEC is off)
  - both operate via selection API and update selection without Streamlit rerun

The component consumes selection state through `window.parent.__fsCaseUiApi`.
