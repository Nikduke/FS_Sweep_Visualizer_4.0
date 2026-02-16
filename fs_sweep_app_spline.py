import os
import hashlib
import json
from typing import Dict, List, Tuple, Optional

# Main app baseline with JS-side interactive case controls.

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType

PLOTLY_LEGEND_SUPPORTS_MAXHEIGHT = "maxheight" in getattr(go.layout.Legend(), "_valid_props", set())


# ---- Page config ----
st.set_page_config(page_title="FS Sweep Visualizer (Spline)", layout="wide")

# =============================================================================
# Settings (single place to tune defaults)
#
# Preference: use named constants grouped by purpose (lowest code churn and most
# readable in a single-file Streamlit app). Keep `STYLE` as a dict because it
# maps directly to Plotly layout fields.
# =============================================================================

# ---- Style (applies to on-page AND exports) ----
STYLE = {
    "font_family": "Open Sans, verdana, arial, sans-serif",
    "font_color": "#000000",
    "base_font_size_px": 14,
    "tick_font_size_px": 14,
    "axis_title_font_size_px": 16,
    "legend_font_size_px": 14,
    "bold_axis_titles": True,
    # Space between tick labels and axis title (px). Set to None to use auto heuristic.
    "xaxis_title_standoff_px": None,
    "yaxis_title_standoff_px": None,
}

# ---- Layout (web view) ----
# NOTE: Keep the bottom legend layout; axis overlap is handled by title standoff + margins.
DEFAULT_FIGURE_WIDTH_PX = 1400
TOP_MARGIN_PX = 40
BOTTOM_AXIS_PX = 60
LEFT_MARGIN_PX = 60
RIGHT_MARGIN_PX = 20

# Layout heuristics (auto margins based on font sizes)
BOTTOM_AXIS_TICK_MULT = 2.4
BOTTOM_AXIS_TITLE_MULT = 1.6
LEFT_MARGIN_TICK_MULT = 4.4
LEFT_MARGIN_TITLE_MULT = 1.6
AXIS_TITLE_STANDOFF_TICK_MULT = 1.1
AXIS_TITLE_STANDOFF_MIN_PX = 10

# ---- Legend sizing (web + export) ----
LEGEND_PADDING_PX = 18  # extra padding used in export legend margin/layout
WEB_LEGEND_EXTRA_PAD_PX = 10  # web-only safety pad to reduce last-row clipping
WEB_LEGEND_VIEWPORT_PX = 500  # fixed visible legend viewport under web line plots

# ---- Performance / computation ----
DEFAULT_SPLINE_SMOOTHING = 1.0
SPLINE_SMOOTHING_MIN = 0.0
SPLINE_SMOOTHING_MAX = 1.3
SPLINE_SMOOTHING_STEP = 0.05
XR_EPS = 1e-9  # treat |R| < XR_EPS as invalid for X/R
XR_EPS_DISPLAY = "1e-9"  # shown in UI text (keep in sync with XR_EPS)

# ---- Export ----
EXPORT_IMAGE_SCALE = 4  # modebar + full-legend export
EXPORT_FALLBACK_COLOR = "#444"

# Full-legend export (JS layout heuristics)
EXPORT_LEGEND_ROW_HEIGHT_FACTOR = 1.25
EXPORT_SAMPLE_LINE_MIN_PX = 18
EXPORT_SAMPLE_LINE_MULT = 1.8
EXPORT_SAMPLE_GAP_MIN_PX = 6
EXPORT_SAMPLE_GAP_MULT = 0.6
EXPORT_TEXT_PAD_MIN_PX = 8
EXPORT_TEXT_PAD_MULT = 0.8
EXPORT_LEGEND_TAIL_FONT_MULT = 0.35
EXPORT_LEGEND_ROW_Y_OFFSET = 0.6
EXPORT_COL_PADDING_MAX_PX = 12
EXPORT_COL_PADDING_FRAC = 0.06

# ---- App behavior ----
UPLOAD_SHA1_PREFIX_LEN = 10

# Session-state cache keys that are scoped by `{data_id}` and can be pruned for old files.
STATE_CACHE_KEY_PREFIXES = (
    "location_select:",
    "line_fig_sig:",
    "line_fig_cache:",
    "line_fig_meta:",
    "rx_filter_sig:",
    "rx_fig_sig:",
    "rx_fig_cache:",
    "rx_fig_steps:",
    "selection_bind_nonce:",
)

# ---- Color shading (clustered color palette) ----
COLOR_FALLBACK_RGB255 = (68, 68, 68)
COLOR_LIGHTEN_MAX_T = 0.40
COLOR_DARKEN_MAX_T = 0.25

# ---- Interactive selection styling ----
SELECTED_LINE_WIDTH = 2.5
DIM_LINE_WIDTH = 1.0
DIM_LINE_OPACITY = 0.35
DIM_LINE_COLOR = "#B8B8B8"
SELECTED_MARKER_SIZE = 10.0
DIM_MARKER_OPACITY = 0.28

# ---- R vs X scatter ----
RX_SCATTER_HEIGHT_FACTOR = 1.5

_plotly_selection_bridge = components.declare_component(
    "plotly_selection_bridge_v15",
    path=str(os.path.join(os.path.dirname(__file__), "plotly_selection_bridge")),
)

_plotly_export_button = components.declare_component(
    "plotly_export_button_v1",
    path=str(os.path.join(os.path.dirname(__file__), "plotly_export_button")),
)

_plotly_rx_toolbar = components.declare_component(
    "plotly_rx_toolbar_v1",
    path=str(os.path.join(os.path.dirname(__file__), "plotly_rx_toolbar")),
)


def plotly_selection_bridge(
    data_id: str,
    chart_id: str,
    plot_ids: List[str],
    cases_meta: List[Dict[str, object]],
    part_labels: List[str],
    color_by_options: List[str],
    color_maps: Dict[str, Dict[str, str]],
    base_frequency_options: List[float],
    base_frequency_default: float,
    auto_color_part_label: str = "",
    color_by_default: str = "Auto",
    show_only_default: bool = False,
    selected_marker_size: float = float(SELECTED_MARKER_SIZE),
    dim_marker_opacity: float = float(DIM_MARKER_OPACITY),
    selected_line_width: float = float(SELECTED_LINE_WIDTH),
    dim_line_width: float = float(DIM_LINE_WIDTH),
    dim_line_opacity: float = float(DIM_LINE_OPACITY),
    dim_line_color: str = str(DIM_LINE_COLOR),
    f_base: float = 50.0,
    n_min: float = 0.0,
    n_max: float = 1.0,
    show_harmonics_default: bool = True,
    bin_width_hz_default: float = 0.0,
    rx_status_dom_id: str = "",
    rx_freq_steps: int = 0,
    reset_token: int = 0,
    selection_reset_token: int = 0,
    render_nonce: int = 0,
    enable_selection: bool = True,
    spline_enabled: bool = False,
) -> None:
    _plotly_selection_bridge(  # type: ignore[misc]
        data_id=str(data_id),
        chart_id=str(chart_id),
        plot_ids=list(plot_ids or []),
        cases_meta=list(cases_meta or []),
        part_labels=list(part_labels or []),
        color_by_options=list(color_by_options or []),
        color_maps=dict(color_maps or {}),
        base_frequency_options=[float(v) for v in list(base_frequency_options or [])],
        base_frequency_default=float(base_frequency_default),
        auto_color_part_label=str(auto_color_part_label or ""),
        color_by_default=str(color_by_default),
        show_only_default=bool(show_only_default),
        selected_marker_size=float(selected_marker_size),
        dim_marker_opacity=float(dim_marker_opacity),
        selected_line_width=float(selected_line_width),
        dim_line_width=float(dim_line_width),
        dim_line_opacity=float(dim_line_opacity),
        dim_line_color=str(dim_line_color),
        f_base=float(f_base),
        n_min=float(n_min),
        n_max=float(n_max),
        show_harmonics_default=bool(show_harmonics_default),
        bin_width_hz_default=float(bin_width_hz_default),
        rx_status_dom_id=str(rx_status_dom_id),
        rx_freq_steps=int(rx_freq_steps),
        reset_token=int(reset_token),
        selection_reset_token=int(selection_reset_token),
        render_nonce=int(render_nonce),
        enable_selection=bool(enable_selection),
        spline_enabled=bool(spline_enabled),
        key=f"plotly_selection_bridge:{data_id}:{chart_id}",
        height=260,
        default=0,
    )


def _note_upload_change() -> None:
    # Called by st.file_uploader(on_change=...): triggers client-state reset on upload actions.
    st.session_state["upload_nonce"] = int(st.session_state.get("upload_nonce", 0)) + 1
    up = st.session_state.get("xlsx_uploader")
    if up is None:
        st.session_state.pop("uploaded_file_sha1_10", None)
        st.session_state.pop("uploaded_file_name", None)
        return
    try:
        st.session_state["uploaded_file_sha1_10"] = hashlib.sha1(up.getvalue()).hexdigest()[: int(UPLOAD_SHA1_PREFIX_LEN)]
        st.session_state["uploaded_file_name"] = getattr(up, "name", None)
    except Exception:
        st.session_state.pop("uploaded_file_sha1_10", None)
        st.session_state.pop("uploaded_file_name", None)


def _prune_data_scoped_session_state(current_data_id: str) -> None:
    """
    Remove stale session-state entries for previous data files.

    Keys listed in `STATE_CACHE_KEY_PREFIXES` are expected to be shaped as:
      `{prefix}{data_id}:...`
    """
    did = str(current_data_id or "")
    if not did:
        return

    for key in list(st.session_state.keys()):
        k = str(key)
        for prefix in STATE_CACHE_KEY_PREFIXES:
            if not k.startswith(prefix):
                continue
            if not k.startswith(f"{prefix}{did}:"):
                st.session_state.pop(k, None)
            break


def _clamp_int(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(val)))


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = (int(_clamp_int(c, 0, 255)) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _mix_rgb(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    tt = float(max(0.0, min(1.0, t)))
    return (
        int(round(a[0] + (b[0] - a[0]) * tt)),
        int(round(a[1] + (b[1] - a[1]) * tt)),
        int(round(a[2] + (b[2] - a[2]) * tt)),
    )


def _parse_color_to_rgb255(color: str) -> Tuple[int, int, int]:
    """
    Accept Plotly palette entries in either hex ("#rrggbb") or "rgb(...)" / "rgba(...)" form.
    """
    c = str(color).strip()
    if not c:
        return tuple(int(v) for v in COLOR_FALLBACK_RGB255)
    if c.startswith("#"):
        return tuple(int(v) for v in pc.hex_to_rgb(c))
    if c.lower().startswith("rgb"):
        tup = pc.unlabel_rgb(c)
        if len(tup) >= 3:
            return (int(round(tup[0])), int(round(tup[1])), int(round(tup[2])))
    # handle hex without '#'
    c2 = c.lstrip().lower()
    if len(c2) in (3, 6) and all(ch in "0123456789abcdef" for ch in c2):
        if len(c2) == 3:
            c2 = "".join([ch * 2 for ch in c2])
        return tuple(int(v) for v in pc.hex_to_rgb(f"#{c2}"))
    return tuple(int(v) for v in COLOR_FALLBACK_RGB255)


def _shade_hex(base_hex: str, position: float) -> str:
    """
    Create a shade variant of a base color.

    `position` in [-1..1]:
      - negative => darken toward black
      - positive => lighten toward white
    """
    base_rgb = _parse_color_to_rgb255(base_hex)
    p = float(max(-1.0, min(1.0, position)))
    if p >= 0:
        # Lighten
        return _rgb_to_hex(_mix_rgb(base_rgb, (255, 255, 255), t=p * float(COLOR_LIGHTEN_MAX_T)))
    # Darken
    return _rgb_to_hex(_mix_rgb(base_rgb, (0, 0, 0), t=(-p) * float(COLOR_DARKEN_MAX_T)))


def build_clustered_case_colors(cases: List[str], hue_part_override: Optional[int] = None) -> Dict[str, str]:
    """
    Assign colors so related cases cluster by hue, with lighter/darker shades inside each cluster.

    Location suffix (after `__`) is ignored for grouping.
    """
    if not cases:
        return {}

    bases = [split_case_location(c)[0] for c in cases]
    split_parts = [str(b).split("_") for b in bases]
    max_parts = max((len(p) for p in split_parts), default=0)
    if max_parts <= 0:
        # Fallback to simple palette
        palette = pc.qualitative.Safe or pc.qualitative.Plotly or ["#1f77b4"]
        return {
            c: palette[i % len(palette)]
            for i, c in enumerate(sorted(cases))
        }

    # Normalize parts (pad with "")
    parts_norm = [p + [""] * (max_parts - len(p)) for p in split_parts]

    # Pick "hue part":
    # - If hue_part_override is provided and valid, use it.
    # - Otherwise, use the most varying part (ties => earlier part).
    uniq_counts = [len(set(row[i] for row in parts_norm)) for i in range(max_parts)]
    if hue_part_override is not None and 0 <= int(hue_part_override) < int(max_parts):
        hue_part = int(hue_part_override)
        varying = [i for i, n in enumerate(uniq_counts) if n > 1]
        rest = [i for i in varying if i != hue_part]
        shade_part = sorted(rest, key=lambda i: (-uniq_counts[i], i))[0] if rest else None
    else:
        varying = [i for i, n in enumerate(uniq_counts) if n > 1]
        if not varying:
            hue_part = 0
            shade_part = None
        else:
            hue_part = sorted(varying, key=lambda i: (-uniq_counts[i], i))[0]
            rest = [i for i in varying if i != hue_part]
            shade_part = sorted(rest, key=lambda i: (-uniq_counts[i], i))[0] if rest else None

    # Use a combined palette so we have enough distinct hues if there are many groups.
    palette = []
    for pal in (
        getattr(pc.qualitative, "Safe", None),
        getattr(pc.qualitative, "D3", None),
        getattr(pc.qualitative, "Plotly", None),
        getattr(pc.qualitative, "Dark24", None),
        getattr(pc.qualitative, "Light24", None),
    ):
        if pal:
            palette.extend(list(pal))
    if not palette:
        palette = ["#1f77b4"]

    # Group cases
    rows = []
    for case, parts in zip(cases, parts_norm):
        group = parts[hue_part]
        shade_key = parts[shade_part] if shade_part is not None else ""
        rows.append((str(group), str(shade_key), str(case)))

    groups = sorted(set(r[0] for r in rows))
    group_color = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    case_colors: Dict[str, str] = {}
    for g in groups:
        group_rows = [r for r in rows if r[0] == g]
        group_rows_sorted = sorted(group_rows, key=lambda r: (r[1], r[2]))
        k = len(group_rows_sorted)
        # Spread shades from darker to lighter.
        positions = np.linspace(-1.0, 1.0, k) if k > 1 else np.array([0.0])
        for (row, pos) in zip(group_rows_sorted, positions):
            _group, _shade_key, case = row
            case_colors[case] = _shade_hex(group_color[g], float(pos))

    return case_colors


@st.cache_data(show_spinner=False)
def cached_clustered_case_colors(cases: Tuple[str, ...], hue_part_override: int) -> Dict[str, str]:
    # hue_part_override: -1 => auto; otherwise 0-based case-part index.
    return build_clustered_case_colors(list(cases), None if int(hue_part_override) < 0 else int(hue_part_override))


# ---- Data loading ----
@st.cache_data(show_spinner=False)
def load_fs_sweep_xlsx(path_or_buf) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    xls = pd.ExcelFile(path_or_buf)
    for name in ["R1", "X1", "R0", "X0"]:
        if name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=name)
            # Frequency column normalization
            freq_col = None
            for c in df.columns:
                c_norm = str(c).strip().lower().replace(" ", "")
                if c_norm in ["frequency(hz)", "frequencyhz", "frequency_"]:
                    freq_col = c
                    break
                if str(c).strip().lower() in ["frequency (hz)", "frequency"]:
                    freq_col = c
                    break
            if freq_col is None:
                if "Frequency (Hz)" in df.columns:
                    freq_col = "Frequency (Hz)"
                else:
                    raise ValueError(f"Sheet '{name}' missing 'Frequency (Hz)' column")
            df = df.rename(columns={freq_col: "Frequency (Hz)"})
            df["Frequency (Hz)"] = pd.to_numeric(df["Frequency (Hz)"], errors="coerce")
            df = df.dropna(subset=["Frequency (Hz)"])
            value_cols = [c for c in df.columns if c != "Frequency (Hz)"]
            if value_cols:
                df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")

            # Prepare numpy arrays once per file load; reused during reruns to speed trace building.
            freq_hz = df["Frequency (Hz)"].to_numpy(copy=False)
            series_map: Dict[object, np.ndarray] = {}
            for c in df.columns:
                if c == "Frequency (Hz)":
                    continue
                series_map[c] = df[c].to_numpy(copy=False)
            df.attrs["__prepared_arrays__"] = (freq_hz, series_map)
            dfs[name] = df
    return dfs


def list_case_columns(df: Optional[pd.DataFrame]) -> List[str]:
    if df is None:
        return []
    return [c for c in df.columns if c != "Frequency (Hz)"]


def split_case_location(name: str) -> Tuple[str, Optional[str]]:
    if "__" in str(name):
        base, loc = str(name).split("__", 1)
        loc = loc if loc else None
        return base, loc
    return str(name), None


def display_case_name(name: str) -> str:
    base, _ = split_case_location(name)
    return base


def prepare_sheet_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[object, np.ndarray]]:
    cached = getattr(df, "attrs", {}).get("__prepared_arrays__")
    if cached is not None:
        return cached

    freq_hz = df["Frequency (Hz)"].to_numpy(copy=False)
    series_map: Dict[object, np.ndarray] = {}
    for c in df.columns:
        if c == "Frequency (Hz)":
            continue
        series_map[c] = df[c].to_numpy(copy=False)
    return freq_hz, series_map


@st.cache_data(show_spinner=False)
def split_case_parts(cases: List[str]) -> Tuple[List[List[str]], List[str]]:
    if not cases:
        return [], []
    temp_parts: List[Tuple[List[str], str]] = []
    max_parts = 0
    for name in cases:
        base_name, location = split_case_location(name)
        base_parts = str(base_name).split("_")
        max_parts = max(max_parts, len(base_parts))
        temp_parts.append((base_parts, location or ""))

    normalized: List[List[str]] = []
    for base_parts, location in temp_parts:
        padded = list(base_parts)
        if len(padded) < max_parts:
            padded.extend([""] * (max_parts - len(padded)))
        padded.append(location or "")
        normalized.append(padded)

    labels = [f"Case part {i+1}" for i in range(max_parts)] + ["Location"]
    return normalized, labels


@st.cache_data(show_spinner=False)
def build_js_case_metadata(cases: Tuple[str, ...]) -> Tuple[List[Dict[str, object]], List[str]]:
    if not cases:
        return [], []
    parts_matrix, part_labels = split_case_parts(list(cases))
    if not part_labels:
        return [], []

    labels_no_loc = list(part_labels[:-1]) if part_labels[-1] == "Location" else list(part_labels)
    part_width = len(labels_no_loc)
    out: List[Dict[str, object]] = []
    for case, row in zip(cases, parts_matrix):
        parts = [str(v) for v in row[:part_width]]
        out.append(
            {
                "case_id": str(case),
                "display_case": str(display_case_name(case)),
                "parts": parts,
            }
        )
    return out, labels_no_loc


@st.cache_data(show_spinner=False)
def _infer_auto_hue_part_label(cases: Tuple[str, ...], part_count: int) -> str:
    # Mirrors Auto hue-part choice used by build_clustered_case_colors(..., hue_part_override=None).
    if not cases or int(part_count) <= 0:
        return ""

    bases = [split_case_location(c)[0] for c in cases]
    split_parts = [str(b).split("_") for b in bases]
    max_parts = max((len(p) for p in split_parts), default=0)
    if max_parts <= 0:
        return ""

    parts_norm = [p + [""] * (max_parts - len(p)) for p in split_parts]
    uniq_counts = [len(set(row[i] for row in parts_norm)) for i in range(max_parts)]
    varying = [i for i, n in enumerate(uniq_counts) if n > 1]
    hue_part = 0 if not varying else sorted(varying, key=lambda i: (-uniq_counts[i], i))[0]
    hue_part = max(0, min(int(part_count) - 1, int(hue_part)))
    return f"Case part {int(hue_part) + 1}"


@st.cache_data(show_spinner=False)
def build_js_color_maps(cases: Tuple[str, ...], part_count: int) -> Tuple[List[str], Dict[str, Dict[str, str]], str]:
    options = ["Auto"] + [f"Case part {i}" for i in range(1, int(part_count) + 1)]
    color_maps: Dict[str, Dict[str, str]] = {}
    for idx, label in enumerate(options):
        hue_idx = -1 if idx == 0 else idx - 1
        cmap = cached_clustered_case_colors(cases, int(hue_idx))
        color_maps[label] = {str(k): str(v) for k, v in cmap.items()}
    auto_color_part_label = _infer_auto_hue_part_label(cases, int(part_count))
    return options, color_maps, auto_color_part_label


def list_location_values(cases: List[str]) -> List[str]:
    vals = sorted({str(split_case_location(c)[1] or "") for c in cases})
    return vals if vals else [""]


def filter_cases_by_location(cases: List[str], location_value: str) -> List[str]:
    loc = str(location_value or "")
    return [c for c in cases if str(split_case_location(c)[1] or "") == loc]


def compute_common_n_range(f_series: List[pd.Series], f_base: float) -> Tuple[float, float]:
    vals: List[float] = []
    for s in f_series:
        if s is None:
            continue
        v = s
        if not pd.api.types.is_numeric_dtype(v):
            v = pd.to_numeric(v, errors="coerce")
        v = v.dropna()
        if not v.empty:
            vals.extend([v.min() / f_base, v.max() / f_base])
    if not vals:
        return 0.0, 1.0
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    return (0.0, 1.0) if (not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi) else (lo, hi)


def make_spline_traces(
    df: pd.DataFrame,
    cases: List[str],
    f_base: float,
    y_title: str,
    smooth: float,
    enable_spline: bool,
    strip_location_suffix: bool,
    case_colors: Dict[str, str],
) -> Tuple[List[BaseTraceType], Optional[pd.Series]]:
    if df is None:
        return [], None
    cd, y_map = prepare_sheet_arrays(df)
    n = cd / float(f_base)
    traces: List[BaseTraceType] = []
    TraceCls = go.Scatter if enable_spline else go.Scattergl
    for case in cases:
        y = y_map.get(case)
        if y is None:
            continue
        color = str(case_colors.get(case, "#1f77b4"))
        line_cfg = dict(color=color)
        tr = TraceCls(
            x=n,
            y=y,
            customdata=cd,
            mode="lines",
            name=display_case_name(case) if strip_location_suffix else str(case),
            meta={
                "kind": "line",
                "case_id": str(case),
                "display_case": str(display_case_name(case)),
                "legend_color": color,
            },
            line=line_cfg,
            opacity=1.0,
            showlegend=True,
            hovertemplate=(
                "Case=%{fullData.name}<br>f=%{customdata:.1f} Hz" + f"<br>{y_title}=%{{y}}<extra></extra>"
            ),
        )
        if enable_spline and isinstance(tr, go.Scatter):
            spline_line = dict(
                shape="spline",
                smoothing=float(smooth),
                simplify=False,
                color=color,
            )
            tr.update(line=spline_line)
        traces.append(tr)
    return traces, df["Frequency (Hz)"]


def apply_common_layout(
    fig: go.Figure,
    plot_height: int,
    y_title: str,
    legend_entrywidth: int,
    use_auto_width: bool,
    figure_width_px: int,
):
    font_base = dict(family=STYLE["font_family"], color=STYLE["font_color"])
    # Keep legend under the plot with a fixed viewport region.
    # Plot-area height semantics from `Plot area height (px)` are preserved.
    bottom_axis_px = int(
        max(
            int(BOTTOM_AXIS_PX),
            int(
                round(
                    float(STYLE["tick_font_size_px"]) * float(BOTTOM_AXIS_TICK_MULT)
                    + float(STYLE["axis_title_font_size_px"]) * float(BOTTOM_AXIS_TITLE_MULT)
                )
            ),
        )
    )
    legend_viewport_px = int(max(120, int(WEB_LEGEND_VIEWPORT_PX)))
    legend_reserve_px = int(legend_viewport_px + int(WEB_LEGEND_EXTRA_PAD_PX))
    total_height = int(plot_height) + int(TOP_MARGIN_PX) + int(bottom_axis_px) + int(legend_reserve_px)
    legend_y = -float(bottom_axis_px + 6) / float(max(1, int(plot_height)))

    # Y-axis overlap fix: grow left margin with font sizes.
    left_margin_px = int(
        max(
            int(LEFT_MARGIN_PX),
            int(
                round(
                    float(STYLE["tick_font_size_px"]) * float(LEFT_MARGIN_TICK_MULT)
                    + float(STYLE["axis_title_font_size_px"]) * float(LEFT_MARGIN_TITLE_MULT)
                )
            ),
        )
    )

    legend_cfg = dict(
        orientation="h",
        yanchor="top",
        y=legend_y,
        xanchor="center",
        x=0.5,
        entrywidth=int(legend_entrywidth),
        entrywidthmode="pixels",
        traceorder="normal",
        font=dict(**font_base, size=int(STYLE["legend_font_size_px"])),
    )
    if PLOTLY_LEGEND_SUPPORTS_MAXHEIGHT:
        legend_cfg["maxheight"] = int(legend_viewport_px)

    fig.update_layout(
        autosize=bool(use_auto_width),
        height=total_height,
        showlegend=True,
        # Preserve Plotly UI state (zoom/pan) across Streamlit reruns.
        uirevision="keep",
        font=dict(
            **font_base,
            size=int(STYLE["base_font_size_px"]),
        ),
        margin=dict(
            l=left_margin_px,
            r=RIGHT_MARGIN_PX,
            t=TOP_MARGIN_PX,
            b=int(bottom_axis_px) + int(legend_reserve_px),
        ),
        margin_autoexpand=False,
        legend=legend_cfg,
    )
    if not use_auto_width:
        fig.update_layout(width=int(figure_width_px), autosize=False)

    x_title = "Harmonic number n = f / f_base"
    y_title_txt = str(y_title)
    if bool(STYLE.get("bold_axis_titles", True)):
        x_title = f"<b>{x_title}</b>"
        y_title_txt = f"<b>{y_title_txt}</b>"

    axis_title_font = dict(**font_base, size=int(STYLE["axis_title_font_size_px"]))
    tick_font = dict(**font_base, size=int(STYLE["tick_font_size_px"]))

    x_title_standoff = STYLE.get("xaxis_title_standoff_px")
    if x_title_standoff is None:
        x_title_standoff = int(
            max(
                int(AXIS_TITLE_STANDOFF_MIN_PX),
                round(float(STYLE["tick_font_size_px"]) * float(AXIS_TITLE_STANDOFF_TICK_MULT)),
            )
        )
    else:
        x_title_standoff = int(x_title_standoff)

    y_title_standoff = STYLE.get("yaxis_title_standoff_px")
    if y_title_standoff is None:
        y_title_standoff = int(
            max(
                int(AXIS_TITLE_STANDOFF_MIN_PX),
                round(float(STYLE["tick_font_size_px"]) * float(AXIS_TITLE_STANDOFF_TICK_MULT)),
            )
        )
    else:
        y_title_standoff = int(y_title_standoff)
    fig.update_xaxes(
        title_text=x_title,
        tick0=1,
        dtick=1,
        title_font=axis_title_font,
        tickfont=tick_font,
        automargin=True,
        title_standoff=x_title_standoff,
    )
    fig.update_yaxes(
        title_text=y_title_txt,
        title_font=axis_title_font,
        tickfont=tick_font,
        automargin=True,
        title_standoff=y_title_standoff,
    )


def build_plot_spline(df: Optional[pd.DataFrame], cases: List[str], f_base: float, plot_height: int, y_title: str,
                      smooth: float, enable_spline: bool, legend_entrywidth: int, strip_location_suffix: bool,
                      use_auto_width: bool, figure_width_px: int, case_colors: Dict[str, str],
                      ) -> Tuple[go.Figure, Optional[pd.Series]]:
    traces, f_series = make_spline_traces(
        df,
        cases,
        f_base,
        y_title,
        smooth,
        enable_spline,
        strip_location_suffix,
        case_colors,
    )
    fig = go.Figure(data=traces)
    apply_common_layout(fig, plot_height, y_title, legend_entrywidth, use_auto_width, figure_width_px)
    return fig, f_series


def build_x_over_r_spline(df_r: Optional[pd.DataFrame], df_x: Optional[pd.DataFrame], cases: List[str], f_base: float,
                          plot_height: int, seq_label: str, smooth: float, legend_entrywidth: int,
                          enable_spline: bool,
                          strip_location_suffix: bool, use_auto_width: bool, figure_width_px: int,
                          case_colors: Dict[str, str],
                          ) -> Tuple[go.Figure, Optional[pd.Series], int, int]:
    xr_dropped = 0
    xr_total = 0
    f_series = None
    eps = float(XR_EPS)
    TraceCls = go.Scatter if enable_spline else go.Scattergl
    traces: List[BaseTraceType] = []
    if df_r is not None and df_x is not None:
        cd, r_map = prepare_sheet_arrays(df_r)
        _cd2, x_map = prepare_sheet_arrays(df_x)
        n = cd / float(f_base)
        both = [c for c in cases if (c in r_map and c in x_map)]
        f_series = df_r["Frequency (Hz)"]
        for case in both:
            r = r_map[case]
            x = x_map[case]
            denom_ok = np.abs(r) >= eps
            bad = (~denom_ok) | np.isnan(r) | np.isnan(x)
            y = np.where(denom_ok, x / r, np.nan)
            xr_dropped += int(np.count_nonzero(bad))
            xr_total += int(r.size)
            color = str(case_colors.get(case, "#1f77b4"))
            line_cfg = dict(color=color)
            tr = TraceCls(
                x=n,
                y=y,
                customdata=cd,
                mode="lines",
                name=display_case_name(case) if strip_location_suffix else str(case),
                meta={
                    "kind": "line",
                    "case_id": str(case),
                    "display_case": str(display_case_name(case)),
                    "legend_color": color,
                },
                line=line_cfg,
                opacity=1.0,
                showlegend=True,
                hovertemplate=(
                    "Case=%{fullData.name}<br>f=%{customdata:.1f} Hz<br>X/R=%{y}<extra></extra>"
                ),
            )
            if enable_spline and isinstance(tr, go.Scatter):
                spline_line = dict(
                    shape="spline",
                    smoothing=float(smooth),
                    simplify=False,
                    color=color,
                )
                tr.update(line=spline_line)
            traces.append(tr)
    fig = go.Figure(data=traces)
    y_title = "X1/R1 (unitless)" if seq_label == "Positive" else "X0/R0 (unitless)"
    apply_common_layout(fig, plot_height, y_title, legend_entrywidth, use_auto_width, figure_width_px)
    return fig, f_series, xr_dropped, xr_total


def build_rx_scatter_animated(
    df_r: Optional[pd.DataFrame],
    df_x: Optional[pd.DataFrame],
    cases: List[str],
    seq_label: str,
    case_colors: Dict[str, str],
    plot_height: int,
    axis_cases: Optional[List[str]] = None,
) -> Tuple[go.Figure, int]:
    fig = go.Figure()
    if df_r is None or df_x is None or not cases:
        fig.update_layout(height=500)
        return fig, 0

    fr, r_map = prepare_sheet_arrays(df_r)
    fx, x_map = prepare_sheet_arrays(df_x)
    if fr.size == 0 or fx.size == 0:
        fig.update_layout(height=500)
        return fig, 0

    freq_candidates = sorted(
        {
            float(v)
            for v in np.concatenate([fr[np.isfinite(fr)], fx[np.isfinite(fx)]], axis=0)
            if np.isfinite(v)
        }
    )
    if not freq_candidates:
        fig.update_layout(height=500)
        return fig, 0
    init_idx = int(min(len(freq_candidates) - 1, max(0, len(freq_candidates) // 2)))
    freq_candidates_arr = np.asarray(freq_candidates, dtype=float)

    r_global_min: Optional[float] = None
    r_global_max: Optional[float] = None
    x_global_min: Optional[float] = None
    x_global_max: Optional[float] = None
    case_arrays: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for case in cases:
        r_arr = r_map.get(case)
        x_arr = x_map.get(case)
        if r_arr is None or x_arr is None:
            continue
        case_arrays.append((str(case), r_arr, x_arr))

    axis_case_list = list(axis_cases) if axis_cases is not None else list(cases)
    for case in axis_case_list:
        r_arr = r_map.get(case)
        x_arr = x_map.get(case)
        if r_arr is None or x_arr is None:
            continue
        r_finite = r_arr[np.isfinite(r_arr)]
        x_finite = x_arr[np.isfinite(x_arr)]
        if r_finite.size > 0:
            r_min_i = float(np.min(r_finite))
            r_max_i = float(np.max(r_finite))
            r_global_min = r_min_i if r_global_min is None else min(r_global_min, r_min_i)
            r_global_max = r_max_i if r_global_max is None else max(r_global_max, r_max_i)
        if x_finite.size > 0:
            x_min_i = float(np.min(x_finite))
            x_max_i = float(np.max(x_finite))
            x_global_min = x_min_i if x_global_min is None else min(x_global_min, x_min_i)
            x_global_max = x_max_i if x_global_max is None else max(x_global_max, x_max_i)

    # Precompute nearest R/X row indices for each slider frequency once.
    idx_r_for_freq = np.array([int(np.argmin(np.abs(fr - float(f_sel)))) for f_sel in freq_candidates_arr], dtype=int)
    idx_x_for_freq = np.array([int(np.argmin(np.abs(fx - float(f_sel)))) for f_sel in freq_candidates_arr], dtype=int)

    def frame_data_for_freq_idx(fi: int) -> Tuple[dict, int]:
        idx_r = int(idx_r_for_freq[int(fi)])
        idx_x = int(idx_x_for_freq[int(fi)])
        f_used_r = float(fr[idx_r])
        f_used_x = float(fx[idx_x])
        f_used = 0.5 * (f_used_r + f_used_x)

        xs: List[float] = []
        ys: List[float] = []
        cds: List[List[object]] = []
        colors: List[str] = []
        ids: List[str] = []

        for case, r_arr, x_arr in case_arrays:
            if idx_r >= int(r_arr.size) or idx_x >= int(x_arr.size):
                continue
            r_v = r_arr[idx_r]
            x_v = x_arr[idx_x]
            if not np.isfinite(r_v) or not np.isfinite(x_v):
                continue
            xs.append(float(r_v))
            ys.append(float(x_v))
            cds.append([str(case), str(display_case_name(case)), float(f_used)])
            colors.append(str(case_colors.get(case, "#1f77b4")))
            ids.append(str(case))

        trace = dict(
            type="scatter",
            x=xs,
            y=ys,
            mode="markers",
            name="Cases",
            customdata=cds,
            ids=ids,
            hovertemplate="Case=%{customdata[1]}<br>f=%{customdata[2]:.1f} Hz<br>R=%{x}<br>X=%{y}<extra></extra>",
            marker=dict(
                color=colors,
                size=float(SELECTED_MARKER_SIZE),
                opacity=1.0,
                line=dict(width=0),
            ),
            showlegend=False,
            meta={"kind": "points"},
        )
        return trace, len(xs)

    f0 = float(freq_candidates[init_idx])
    tr0, _ = frame_data_for_freq_idx(init_idx)
    fig.add_trace(go.Scatter(**tr0))

    frames: List[go.Frame] = []
    for i, f_sel in enumerate(freq_candidates):
        tr_i, _ = frame_data_for_freq_idx(i)
        frames.append(
            go.Frame(
                name=f"{float(f_sel):.6g}",
                data=[go.Scatter(**tr_i)],
                traces=[0],
                layout=go.Layout(title=f"R vs X at f ~ {float(f_sel):.1f} Hz ({seq_label})"),
            )
        )
    fig.frames = frames
    slider_steps = [
        dict(
            method="animate",
            args=[
                [f"{float(f_sel):.6g}"],
                dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0)),
            ],
            label=f"{float(f_sel):.1f}",
        )
        for f_sel in freq_candidates
    ]

    shapes: List[dict] = [
        dict(type="line", xref="x", yref="paper", x0=0, x1=0, y0=0, y1=1, line=dict(color="rgba(0,0,0,0.45)", width=1)),
        dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=0, line=dict(color="rgba(0,0,0,0.45)", width=1)),
    ]
    fig.update_layout(
        title=f"R vs X at f ~ {f0:.1f} Hz ({seq_label})",
        xaxis_title=f"R{1 if seq_label == 'Positive' else 0} (Ohm)",
        yaxis_title=f"X{1 if seq_label == 'Positive' else 0} (Ohm)",
        height=max(420, int(round(float(plot_height) * float(RX_SCATTER_HEIGHT_FACTOR)))),
        dragmode="zoom",
        # Keep click handling deterministic in custom JS stager.
        # Using "event" avoids Plotly's built-in selection-state side effects.
        clickmode="event",
        shapes=shapes,
        sliders=[
            dict(
                active=int(init_idx),
                currentvalue=dict(prefix="Frequency (Hz): "),
                pad=dict(t=20),
                steps=slider_steps,
            )
        ],
    )

    if (
        r_global_min is not None and r_global_max is not None and np.isfinite(r_global_min) and np.isfinite(r_global_max)
        and x_global_min is not None and x_global_max is not None and np.isfinite(x_global_min) and np.isfinite(x_global_max)
    ):
        rx_pad = max(1e-12, 0.04 * max(1e-12, float(r_global_max - r_global_min)))
        xx_pad = max(1e-12, 0.04 * max(1e-12, float(x_global_max - x_global_min)))
        fig.update_xaxes(range=[float(r_global_min - rx_pad), float(r_global_max + rx_pad)])
        fig.update_yaxes(range=[float(x_global_min - xx_pad), float(x_global_max + xx_pad)])
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)
    return fig, len(freq_candidates)


def _make_plot_item(
    kind: str,
    fig: go.Figure,
    f_ref: Optional[pd.Series],
    filename: str,
    button_label: str,
    chart_key: str,
) -> Dict[str, object]:
    return {
        "kind": str(kind),
        "fig": fig,
        "f_ref": f_ref,
        "filename": str(filename),
        "button_label": str(button_label),
        "chart_key": str(chart_key),
    }


def _render_client_png_download(
    filename: str,
    scale: int,
    button_label: str,
    plot_height: int,
    legend_entrywidth: int,
    plot_index: int,
):
    key_payload = f"{filename}|{scale}|{plot_height}|{legend_entrywidth}|{plot_index}"
    key_hash = hashlib.sha1(key_payload.encode("utf-8")).hexdigest()[:10]
    _plotly_export_button(  # type: ignore[misc]
        filename=str(filename),
        scale=int(scale),
        button_label=str(button_label),
        plot_height=int(plot_height),
        legend_entrywidth=int(legend_entrywidth),
        plot_index=int(plot_index),
        top_margin_px=int(TOP_MARGIN_PX),
        bottom_axis_px=int(BOTTOM_AXIS_PX),
        legend_padding_px=int(LEGEND_PADDING_PX),
        fallback_legend_font_size=int(STYLE["legend_font_size_px"]),
        legend_font_family=str(STYLE["font_family"]),
        legend_font_color=str(STYLE["font_color"]),
        left_margin_px=int(LEFT_MARGIN_PX),
        right_margin_px=int(RIGHT_MARGIN_PX),
        tick_font_size_px=int(STYLE["tick_font_size_px"]),
        axis_title_font_size_px=int(STYLE["axis_title_font_size_px"]),
        left_margin_tick_mult=float(LEFT_MARGIN_TICK_MULT),
        left_margin_title_mult=float(LEFT_MARGIN_TITLE_MULT),
        export_legend_row_height_factor=float(EXPORT_LEGEND_ROW_HEIGHT_FACTOR),
        export_sample_line_min_px=int(EXPORT_SAMPLE_LINE_MIN_PX),
        export_sample_line_mult=float(EXPORT_SAMPLE_LINE_MULT),
        export_sample_gap_min_px=int(EXPORT_SAMPLE_GAP_MIN_PX),
        export_sample_gap_mult=float(EXPORT_SAMPLE_GAP_MULT),
        export_text_pad_min_px=int(EXPORT_TEXT_PAD_MIN_PX),
        export_text_pad_mult=float(EXPORT_TEXT_PAD_MULT),
        export_legend_tail_font_mult=float(EXPORT_LEGEND_TAIL_FONT_MULT),
        export_legend_row_y_offset=float(EXPORT_LEGEND_ROW_Y_OFFSET),
        export_col_padding_max_px=int(EXPORT_COL_PADDING_MAX_PX),
        export_col_padding_frac=float(EXPORT_COL_PADDING_FRAC),
        export_fallback_color=str(EXPORT_FALLBACK_COLOR),
        key=f"plotly_export_button:{key_hash}",
        height=30,
        default=0,
    )


def _render_rx_client_step_buttons(plot_index: int, data_id: str, chart_id: str) -> None:
    key_payload = f"{plot_index}|{data_id}|{chart_id}"
    key_hash = hashlib.sha1(key_payload.encode("utf-8")).hexdigest()[:10]
    _plotly_rx_toolbar(  # type: ignore[misc]
        plot_index=int(plot_index),
        data_id=str(data_id),
        chart_id=str(chart_id),
        key=f"plotly_rx_toolbar:{key_hash}",
        height=86,
        default=0,
    )


def _load_data_source(default_path: str) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Resolve active workbook source and a stable data_id for cache scoping.
    """
    up = st.sidebar.file_uploader(
        "Upload Excel",
        type=["xlsx"],
        key="xlsx_uploader",
        on_change=_note_upload_change,
        help="If empty, loads 'FS_sweep.xlsx' from this folder.",
    )
    st.sidebar.markdown("---")

    data_id = "unknown"
    if up is not None:
        data = load_fs_sweep_xlsx(up)
        try:
            cached = st.session_state.get("uploaded_file_sha1_10")
            data_id = str(cached) if cached else hashlib.sha1(up.getvalue()).hexdigest()[:10]
        except Exception:
            data_id = f"upload:{getattr(up, 'name', 'file')}"
        return data, data_id

    if os.path.exists(default_path):
        data = load_fs_sweep_xlsx(default_path)
        try:
            data_id = f"local:{int(os.path.getmtime(default_path))}"
        except Exception:
            data_id = "local"
        st.sidebar.info(f"Loaded local file: {default_path}")
        return data, data_id

    st.warning("Upload an Excel file or place 'FS_sweep.xlsx' here.")
    st.stop()
    return {}, data_id


def _render_global_controls(seq_key: str) -> Dict[str, object]:
    """
    Render Streamlit-side controls that intentionally trigger reruns.
    """
    if st.session_state.get(seq_key) not in ("Positive", "Zero"):
        st.session_state[seq_key] = "Positive"

    st.sidebar.header("Controls")
    seq_label = str(st.session_state.get(seq_key, "Positive"))
    seq = ("R1", "X1") if seq_label == "Positive" else ("R0", "X0")

    # Base frequency is JS-side (50/60 switch) to avoid Streamlit reruns.
    # Server traces are built with 50 Hz baseline and restyled client-side when switched.
    f_base = 50.0
    plot_height = st.sidebar.slider("Plot area height (px)", min_value=100, max_value=1000, value=400, step=25)
    use_auto_width = st.sidebar.checkbox("Auto width (fit container)", value=True)
    figure_width_px = DEFAULT_FIGURE_WIDTH_PX
    if not use_auto_width:
        figure_width_px = st.sidebar.slider("Figure width (px)", min_value=800, max_value=2200, value=DEFAULT_FIGURE_WIDTH_PX, step=50)

    enable_spline = st.sidebar.checkbox("Spline (slow)", value=False)
    spline_selection_reset_key = "selection_reset_nonce:spline_toggle"
    spline_prev_state_key = "selection_reset_prev_spline"
    prev_spline_state = st.session_state.get(spline_prev_state_key, None)
    if prev_spline_state is None:
        st.session_state[spline_prev_state_key] = bool(enable_spline)
    elif bool(prev_spline_state) != bool(enable_spline):
        st.session_state[spline_selection_reset_key] = int(st.session_state.get(spline_selection_reset_key, 0)) + 1
        st.session_state[spline_prev_state_key] = bool(enable_spline)
    selection_reset_token = int(st.session_state.get(spline_selection_reset_key, 0))

    smooth = float(DEFAULT_SPLINE_SMOOTHING)
    if enable_spline:
        prev_smooth = st.session_state.get("spline_smoothing", float(DEFAULT_SPLINE_SMOOTHING))
        try:
            prev_smooth_f = float(prev_smooth)
        except Exception:
            prev_smooth_f = float(DEFAULT_SPLINE_SMOOTHING)
        prev_smooth_f = max(float(SPLINE_SMOOTHING_MIN), min(float(SPLINE_SMOOTHING_MAX), prev_smooth_f))
        smooth = st.sidebar.slider(
            "Spline smoothing",
            min_value=float(SPLINE_SMOOTHING_MIN),
            max_value=float(SPLINE_SMOOTHING_MAX),
            value=prev_smooth_f,
            step=float(SPLINE_SMOOTHING_STEP),
            key="spline_smoothing",
        )
    st.sidebar.markdown("---")

    return {
        "seq_label": seq_label,
        "seq": seq,
        "f_base": float(f_base),
        "plot_height": int(plot_height),
        "use_auto_width": bool(use_auto_width),
        "figure_width_px": int(figure_width_px),
        "enable_spline": bool(enable_spline),
        "smooth": float(smooth),
        "selection_reset_token": int(selection_reset_token),
    }


def _render_show_plot_row(label: str, key: str, default: bool) -> Tuple[bool, object]:
    row = st.sidebar.columns([0.40, 0.60], gap="small", vertical_alignment="center")
    with row[0]:
        enabled = st.checkbox(label, value=bool(default), key=key)
    with row[1]:
        export_slot = st.empty()
    return bool(enabled), export_slot


def _render_show_plots_controls() -> Tuple[bool, bool, bool, bool, Dict[str, object]]:
    st.sidebar.header("Show plots")
    show_plot_rx = st.sidebar.checkbox("R vs X scatter", value=True, key="show_plot_rx")
    show_plot_x, export_slot_x = _render_show_plot_row("X", "show_plot_x", True)
    show_plot_r, export_slot_r = _render_show_plot_row("R", "show_plot_r", False)
    show_plot_xr, export_slot_xr = _render_show_plot_row("X/R", "show_plot_xr", False)
    return show_plot_rx, show_plot_x, show_plot_r, show_plot_xr, {
        "x": export_slot_x,
        "r": export_slot_r,
        "xr": export_slot_xr,
    }


def _render_line_export_buttons(
    export_slots: Dict[str, object],
    plot_items_by_kind: Dict[str, Dict[str, object]],
    show_plot_x: bool,
    show_plot_r: bool,
    show_plot_xr: bool,
    line_plot_index_map: Dict[str, int],
    line_plot_base_index: int,
    export_scale: int,
    plot_height: int,
    legend_entrywidth: int,
) -> None:
    show_flags = {
        "x": bool(show_plot_x),
        "r": bool(show_plot_r),
        "xr": bool(show_plot_xr),
    }
    for kind in ("x", "r", "xr"):
        slot = export_slots.get(kind)
        if slot is None:
            continue
        if show_flags.get(kind) and kind in plot_items_by_kind:
            item = plot_items_by_kind[kind]
            with slot.container():
                _render_client_png_download(
                    filename=str(item["filename"]),
                    scale=int(export_scale),
                    button_label=str(item["button_label"]),
                    plot_height=int(plot_height),
                    legend_entrywidth=int(legend_entrywidth),
                    plot_index=int(line_plot_index_map.get(kind, line_plot_base_index)),
                )
        else:
            slot.empty()


def main():
    st.title("FS Sweep Visualizer (Spline)")

    # Data source
    default_path = "FS_sweep.xlsx"
    st.sidebar.header("Data Source")
    try:
        data, data_id = _load_data_source(default_path)
    except Exception as e:
        st.error(f"Failed to load Excel: {e}")
        st.stop()

    _prune_data_scoped_session_state(data_id)

    upload_nonce = int(st.session_state.get("upload_nonce", 0))
    seq_key = "seq_label_control"
    controls = _render_global_controls(seq_key)
    seq_label = str(controls["seq_label"])
    seq = controls["seq"]
    f_base = float(controls["f_base"])
    plot_height = int(controls["plot_height"])
    use_auto_width = bool(controls["use_auto_width"])
    figure_width_px = int(controls["figure_width_px"])
    enable_spline = bool(controls["enable_spline"])
    smooth = float(controls["smooth"])
    selection_reset_token = int(controls["selection_reset_token"])

    # Prepare sequence sheets and full case list.
    df_r = data.get(seq[0])
    df_x = data.get(seq[1])
    if df_r is None and df_x is None:
        st.error(f"Missing sheets for sequence '{seq_label}' ({seq[0]}/{seq[1]}).")
        st.stop()

    all_cases = sorted(list({*list_case_columns(df_r), *list_case_columns(df_x)}))
    if not all_cases:
        st.warning("No case columns found in the selected sequence sheets.")
        st.stop()

    location_values = list_location_values(all_cases)
    location_labels = [("<empty>" if str(v) == "" else str(v)) for v in location_values]
    location_label_to_value = {lbl: val for lbl, val in zip(location_labels, location_values)}
    location_key = f"location_select:{data_id}:{seq_label}"
    if location_key not in st.session_state or st.session_state.get(location_key) not in location_labels:
        st.session_state[location_key] = location_labels[0]
    # Show-plots controls (R vs X default on).
    show_plot_rx, show_plot_x, show_plot_r, show_plot_xr, export_slots = _render_show_plots_controls()
    if not (show_plot_x or show_plot_r or show_plot_xr or show_plot_rx):
        st.warning("Select at least one plot to display.")
        st.stop()
    st.sidebar.markdown("---")

    st.sidebar.header("Case Filters & Selection")
    st.sidebar.radio("Sequence", ["Positive", "Zero"], key=seq_key)
    selected_location_label = st.sidebar.radio("Location", options=location_labels, key=location_key)
    selected_location = str(location_label_to_value.get(str(selected_location_label), ""))
    interactive_controls_area = st.sidebar.container()
    st.sidebar.markdown("---")

    # Validate required sheets for enabled plots.
    if (show_plot_r or show_plot_xr or show_plot_rx) and df_r is None:
        st.error(f"Sheet '{seq[0]}' is missing, but R, X/R and/or R vs X scatter is enabled.")
        st.stop()
    if (show_plot_x or show_plot_xr or show_plot_rx) and df_x is None:
        st.error(f"Sheet '{seq[1]}' is missing, but X, X/R and/or R vs X scatter is enabled.")
        st.stop()

    location_cases = filter_cases_by_location(all_cases, selected_location)
    if not location_cases:
        st.warning("No cases found for the selected location.")
        st.stop()

    cases_tuple = tuple(location_cases)
    cases_meta, part_labels = build_js_case_metadata(cases_tuple)
    color_by_options, color_maps, auto_color_part_label = build_js_color_maps(cases_tuple, len(part_labels))
    default_color_map = dict(color_maps.get("Auto", {}))
    case_colors_line = {c: str(default_color_map.get(c, "#1f77b4")) for c in location_cases}
    case_colors_scatter = {c: str(default_color_map.get(c, "#1f77b4")) for c in location_cases}

    # Plotly modebar image export defaults.
    download_config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": "plot",
            "scale": int(EXPORT_IMAGE_SCALE),
        }
    }

    # Build all cases for selected location on server-side.
    # Case-part filtering and selection styling are applied client-side in JS.
    cases_for_line = list(location_cases)
    strip_location_suffix = True

    legend_cases = list(cases_for_line)
    display_names = [display_case_name(c) for c in legend_cases]
    max_len = max((len(n) for n in display_names), default=12)
    legend_font_px = int(STYLE["legend_font_size_px"])
    approx_char_px = max(7, int(round(0.60 * float(legend_font_px))))
    base_px = max(44, int(round(3.5 * float(legend_font_px))))  # symbol + padding inside a legend item

    # Internal legend column width (used by on-page layout + full-legend export).
    est_width_px = int(figure_width_px)
    usable_w = max(1, int(est_width_px) - int(LEFT_MARGIN_PX) - int(RIGHT_MARGIN_PX))
    desired = int(max_len) * int(approx_char_px) + int(base_px)
    legend_entrywidth = _clamp_int(desired, 50, min(900, usable_w))
    if legend_entrywidth >= int(usable_w * 0.95):
        legend_entrywidth = usable_w

    # Render order for currently enabled plots.
    plot_order: List[str] = []
    if show_plot_rx:
        plot_order.append("rx")
    if show_plot_x:
        plot_order.append("x")
    if show_plot_r:
        plot_order.append("r")
    if show_plot_xr:
        plot_order.append("xr")

    # Build plots
    r_title = "R1 (\u03A9)" if seq_label == "Positive" else "R0 (\u03A9)"
    x_title = "X1 (\u03A9)" if seq_label == "Positive" else "X0 (\u03A9)"
    plot_items: List[Dict[str, object]] = []
    xr_dropped = 0
    xr_total = 0

    if show_plot_x:
        x_cache_sig_key = f"line_fig_sig:{data_id}:{seq_label}:x"
        x_cache_fig_key = f"line_fig_cache:{data_id}:{seq_label}:x"
        x_sig_payload = {
            "kind": "x",
            "cases": list(cases_for_line),
            "f_base": float(f_base),
            "plot_h": int(plot_height),
            "smooth": float(smooth),
            "spline": bool(enable_spline),
            "legend_w": int(legend_entrywidth),
            "strip_loc": bool(strip_location_suffix),
            "auto_w": bool(use_auto_width),
            "fig_w": int(figure_width_px),
            "colors": [[str(c), str(case_colors_line.get(c, "#1f77b4"))] for c in cases_for_line],
            "title": str(x_title),
        }
        x_sig = hashlib.sha1(json.dumps(x_sig_payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
        if st.session_state.get(x_cache_sig_key) != x_sig or (x_cache_fig_key not in st.session_state):
            fig_x_built, _ = build_plot_spline(
                df_x,
                cases_for_line,
                f_base,
                plot_height,
                x_title,
                smooth,
                enable_spline,
                legend_entrywidth,
                strip_location_suffix,
                use_auto_width,
                figure_width_px,
                case_colors_line,
            )
            st.session_state[x_cache_sig_key] = x_sig
            st.session_state[x_cache_fig_key] = fig_x_built.to_dict()
        fig_x = go.Figure(st.session_state.get(x_cache_fig_key, {}))
        f_x = df_x["Frequency (Hz)"] if df_x is not None else None
        plot_items.append(_make_plot_item("x", fig_x, f_x, "X_full_legend.png", "Export", "plot_x"))

    if show_plot_r:
        r_cache_sig_key = f"line_fig_sig:{data_id}:{seq_label}:r"
        r_cache_fig_key = f"line_fig_cache:{data_id}:{seq_label}:r"
        r_sig_payload = {
            "kind": "r",
            "cases": list(cases_for_line),
            "f_base": float(f_base),
            "plot_h": int(plot_height),
            "smooth": float(smooth),
            "spline": bool(enable_spline),
            "legend_w": int(legend_entrywidth),
            "strip_loc": bool(strip_location_suffix),
            "auto_w": bool(use_auto_width),
            "fig_w": int(figure_width_px),
            "colors": [[str(c), str(case_colors_line.get(c, "#1f77b4"))] for c in cases_for_line],
            "title": str(r_title),
        }
        r_sig = hashlib.sha1(json.dumps(r_sig_payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
        if st.session_state.get(r_cache_sig_key) != r_sig or (r_cache_fig_key not in st.session_state):
            fig_r_built, _ = build_plot_spline(
                df_r,
                cases_for_line,
                f_base,
                plot_height,
                r_title,
                smooth,
                enable_spline,
                legend_entrywidth,
                strip_location_suffix,
                use_auto_width,
                figure_width_px,
                case_colors_line,
            )
            st.session_state[r_cache_sig_key] = r_sig
            st.session_state[r_cache_fig_key] = fig_r_built.to_dict()
        fig_r = go.Figure(st.session_state.get(r_cache_fig_key, {}))
        f_r = df_r["Frequency (Hz)"] if df_r is not None else None
        plot_items.append(_make_plot_item("r", fig_r, f_r, "R_full_legend.png", "Export", "plot_r"))

    if show_plot_xr:
        xr_cache_sig_key = f"line_fig_sig:{data_id}:{seq_label}:xr"
        xr_cache_fig_key = f"line_fig_cache:{data_id}:{seq_label}:xr"
        xr_cache_meta_key = f"line_fig_meta:{data_id}:{seq_label}:xr"
        xr_sig_payload = {
            "kind": "xr",
            "cases": list(cases_for_line),
            "f_base": float(f_base),
            "plot_h": int(plot_height),
            "smooth": float(smooth),
            "spline": bool(enable_spline),
            "legend_w": int(legend_entrywidth),
            "strip_loc": bool(strip_location_suffix),
            "auto_w": bool(use_auto_width),
            "fig_w": int(figure_width_px),
            "colors": [[str(c), str(case_colors_line.get(c, "#1f77b4"))] for c in cases_for_line],
            "title": str(seq_label),
        }
        xr_sig = hashlib.sha1(json.dumps(xr_sig_payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
        if st.session_state.get(xr_cache_sig_key) != xr_sig or (xr_cache_fig_key not in st.session_state):
            fig_xr_built, _, xr_dropped_built, xr_total_built = build_x_over_r_spline(
                df_r,
                df_x,
                cases_for_line,
                f_base,
                plot_height,
                seq_label,
                smooth,
                legend_entrywidth,
                enable_spline,
                strip_location_suffix,
                use_auto_width,
                figure_width_px,
                case_colors_line,
            )
            st.session_state[xr_cache_sig_key] = xr_sig
            st.session_state[xr_cache_fig_key] = fig_xr_built.to_dict()
            st.session_state[xr_cache_meta_key] = {
                "xr_dropped": int(xr_dropped_built),
                "xr_total": int(xr_total_built),
            }
        fig_xr = go.Figure(st.session_state.get(xr_cache_fig_key, {}))
        xr_meta = st.session_state.get(xr_cache_meta_key, {}) if isinstance(st.session_state.get(xr_cache_meta_key, {}), dict) else {}
        xr_dropped = int(xr_meta.get("xr_dropped", 0))
        xr_total = int(xr_meta.get("xr_total", 0))
        f_xr = df_r["Frequency (Hz)"] if df_r is not None else None
        plot_items.append(_make_plot_item("xr", fig_xr, f_xr, "X_over_R_full_legend.png", "Export", "plot_xr"))

    f_refs = [it["f_ref"] for it in plot_items if it.get("f_ref") is not None]
    n_lo, n_hi = compute_common_n_range(f_refs, f_base)
    for it in plot_items:
        fig = it["fig"]
        if isinstance(fig, go.Figure):
            fig.update_xaxes(range=[n_lo, n_hi])

    # Render
    location_caption = selected_location if selected_location else "<empty>"
    st.subheader(f"Sequence: {seq_label} | Location: {location_caption}")
    if show_plot_xr and xr_total > 0 and xr_dropped > 0:
        st.caption(f"X/R: dropped {xr_dropped} of {xr_total} points where |R| < {XR_EPS_DISPLAY} or data missing.")

    export_scale = int(EXPORT_IMAGE_SCALE)
    plot_items_by_kind = {str(it["kind"]): it for it in plot_items}
    line_plot_base_index = 1 if show_plot_rx else 0
    line_kind_order: List[str] = []
    if show_plot_x:
        line_kind_order.append("x")
    if show_plot_r:
        line_kind_order.append("r")
    if show_plot_xr:
        line_kind_order.append("xr")
    line_plot_index_map = {kind: int(line_plot_base_index + idx) for idx, kind in enumerate(line_kind_order)}

    _render_line_export_buttons(
        export_slots=export_slots,
        plot_items_by_kind=plot_items_by_kind,
        show_plot_x=bool(show_plot_x),
        show_plot_r=bool(show_plot_r),
        show_plot_xr=bool(show_plot_xr),
        line_plot_index_map=line_plot_index_map,
        line_plot_base_index=int(line_plot_base_index),
        export_scale=int(export_scale),
        plot_height=int(plot_height),
        legend_entrywidth=int(legend_entrywidth),
    )

    scatter_slot = st.container()
    line_slot = st.container()

    with line_slot:
        for idx, it in enumerate(plot_items):
            fig = it["fig"]
            chart_key = str(it["chart_key"])
            if isinstance(fig, go.Figure):
                st.plotly_chart(fig, use_container_width=bool(use_auto_width), config=download_config, key=chart_key)
            if idx < len(plot_items) - 1:
                st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)

    rx_status_dom_id = ""
    rx_freq_steps_for_bridge = 0
    with scatter_slot:
        if show_plot_rx:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.subheader("R vs X Scatter")

            rx_filter_sig_key = f"rx_filter_sig:{data_id}:{seq_label}"
            rx_fig_sig_key = f"rx_fig_sig:{data_id}:{seq_label}"
            rx_fig_cache_key = f"rx_fig_cache:{data_id}:{seq_label}"
            rx_fig_steps_key = f"rx_fig_steps:{data_id}:{seq_label}"

            filter_sig = hashlib.sha1("|".join(sorted(location_cases)).encode("utf-8")).hexdigest()[:12]
            prev_filter_sig = str(st.session_state.get(rx_filter_sig_key, ""))
            if prev_filter_sig != filter_sig:
                st.session_state[rx_filter_sig_key] = filter_sig
                st.session_state.pop(rx_fig_sig_key, None)
                st.session_state.pop(rx_fig_cache_key, None)
                st.session_state.pop(rx_fig_steps_key, None)

            # Location-based baseline for scatter axis limits.
            location_cases_for_axes = list(location_cases)

            rx_sig_payload = {
                "seq": str(seq_label),
                "plot_h": int(plot_height),
                "cases": list(location_cases),
                "axis_cases": list(location_cases_for_axes),
                "colors": [[str(c), str(case_colors_scatter.get(c, "#1f77b4"))] for c in location_cases],
            }
            rx_sig = hashlib.sha1(json.dumps(rx_sig_payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]

            if st.session_state.get(rx_fig_sig_key) != rx_sig or (rx_fig_cache_key not in st.session_state):
                rx_fig_built, rx_steps_built = build_rx_scatter_animated(
                    df_r=df_r,
                    df_x=df_x,
                    cases=list(location_cases),
                    seq_label=seq_label,
                    case_colors=case_colors_scatter,
                    plot_height=int(plot_height),
                    axis_cases=list(location_cases_for_axes),
                )
                st.session_state[rx_fig_sig_key] = rx_sig
                st.session_state[rx_fig_cache_key] = rx_fig_built.to_dict()
                st.session_state[rx_fig_steps_key] = int(rx_steps_built)

            rx_fig = go.Figure(st.session_state.get(rx_fig_cache_key, {}))
            rx_fig.update_layout(uirevision=f"rx:{seq_label}")
            rx_freq_steps = int(st.session_state.get(rx_fig_steps_key, 0))
            rx_freq_steps_for_bridge = int(rx_freq_steps)
            rx_status_dom_id = f"rx-status-{hashlib.sha1(f'{data_id}:{seq_label}:{selected_location}'.encode('utf-8')).hexdigest()[:10]}"

            st.plotly_chart(rx_fig, use_container_width=bool(use_auto_width), config=download_config, key="plot_rx")
            rx_plot_index = int(plot_order.index("rx")) if "rx" in plot_order else 0
            _render_rx_client_step_buttons(rx_plot_index, data_id=data_id, chart_id=f"{seq_label}:{selected_location}")
            st.markdown(
                (
                    f"<div id=\"{rx_status_dom_id}\" style=\"font-size:0.92rem; color:#666; margin:2px 0 2px 0;\">"
                    f"R vs X points shown: {len(location_cases)} | Frequency steps: {int(rx_freq_steps)}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            st.caption("Point clicks toggle selection. Case-part/color/selection controls are in the sidebar.")

    sel_bind_nonce_key = f"selection_bind_nonce:{data_id}:{seq_label}:{selected_location}"
    st.session_state[sel_bind_nonce_key] = int(st.session_state.get(sel_bind_nonce_key, 0)) + 1

    with interactive_controls_area:
        plotly_selection_bridge(
            data_id=data_id,
            chart_id=f"{seq_label}:{selected_location}",
            plot_ids=list(plot_order),
            cases_meta=list(cases_meta),
            part_labels=list(part_labels),
            color_by_options=list(color_by_options),
            color_maps=dict(color_maps),
            base_frequency_options=[50.0, 60.0],
            base_frequency_default=50.0,
            auto_color_part_label=str(auto_color_part_label),
            color_by_default="Auto",
            show_only_default=False,
            selected_marker_size=float(SELECTED_MARKER_SIZE),
            dim_marker_opacity=float(DIM_MARKER_OPACITY),
            selected_line_width=float(SELECTED_LINE_WIDTH),
            dim_line_width=float(DIM_LINE_WIDTH),
            dim_line_opacity=float(DIM_LINE_OPACITY),
            dim_line_color=str(DIM_LINE_COLOR),
            f_base=float(f_base),
            n_min=float(n_lo),
            n_max=float(n_hi),
            show_harmonics_default=True,
            bin_width_hz_default=0.0,
            rx_status_dom_id=str(rx_status_dom_id),
            rx_freq_steps=int(rx_freq_steps_for_bridge),
            reset_token=int(upload_nonce),
            selection_reset_token=int(selection_reset_token),
            render_nonce=int(st.session_state.get(sel_bind_nonce_key, 0)),
            enable_selection=bool(show_plot_rx),
            spline_enabled=bool(enable_spline),
        )


if __name__ == "__main__":
    main()
