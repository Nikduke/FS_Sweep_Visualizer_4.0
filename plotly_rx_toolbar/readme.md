# plotly_rx_toolbar

Mini Streamlit component used by `fs_sweep_app_spline.py` to render the control row under `R vs X` scatter.

Responsibilities:
- `Prev frequency` / `Next frequency` frame stepping
- `Clear list`, `Download selected CSV`
- `Show only selected sweeps` checkbox synchronized with shared JS selection API

The component consumes selection state through `window.parent.__fsCaseUiApi`.
