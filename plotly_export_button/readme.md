# plotly_export_button

Mini Streamlit component used by `fs_sweep_app_spline.py` to render the sidebar `Export` button for each enabled line plot (`X`, `R`, `X/R`).

Responsibilities:
- render compact button UI
- collect active Plotly trace visibility/style from the target line plot
- generate full-legend PNG in browser with manual legend layout

No Streamlit rerun is triggered by button clicks.

Current status:
- behavior unchanged by recent scatter/single-trace optimizations
- continues to export from the currently rendered line-plot state
