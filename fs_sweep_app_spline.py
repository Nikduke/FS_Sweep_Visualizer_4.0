import os
import hashlib
import json
import re
from typing import Dict, List, Tuple, Optional, Set

# Main app baseline with client-side zoom persistence + staged scatter selection.

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType


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
LEGEND_ROW_HEIGHT_FACTOR = 2.0  # web estimate: row height ~= legend_font_size * factor
LEGEND_PADDING_PX = 18  # extra padding used in legend height estimate and export margin
WEB_LEGEND_EXTRA_PAD_PX = 20  # web-only safety pad to reduce last-row clipping
WEB_LEGEND_MAX_HEIGHT_PX = 1000  # cap reserved web legend height
AUTO_WIDTH_ESTIMATE_PX = 950  # used only when Plotly auto-sizes (legend row estimate)

# ---- Performance / computation ----
DEFAULT_SPLINE_SMOOTHING = 1.0
SPLINE_SMOOTHING_MIN = 0.0
SPLINE_SMOOTHING_MAX = 1.3
SPLINE_SMOOTHING_STEP = 0.05
XR_EPS = 1e-9  # treat |R| < XR_EPS as invalid for X/R
XR_EPS_DISPLAY = "1e-9"  # shown in UI text (keep in sync with XR_EPS)

# ---- Export ----
EXPORT_IMAGE_SCALE = 4  # modebar + full-legend export
EXPORT_DOM_ID_HASH_LEN = 12
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
CHECKBOX_KEY_HASH_LEN = 12

# Zoom listener (JS bind loop and mount-time relayout ignore window)
ZOOM_BIND_TRIES = 80
ZOOM_BIND_INTERVAL_MS = 100
ZOOM_IGNORE_AUTORANGE_MS = 1200

# ---- Color shading (clustered color palette) ----
COLOR_FALLBACK_RGB255 = (68, 68, 68)
COLOR_LIGHTEN_MAX_T = 0.40
COLOR_DARKEN_MAX_T = 0.25

# Debug flag (code-only). When True, prints the latest relayout payload and stored zoom.
DEBUG_ZOOM = False

# ---- Study selection styling ----
STUDY_SELECTED_LINE_WIDTH = 2.5
STUDY_BACKGROUND_LINE_WIDTH = 1
STUDY_BACKGROUND_OPACITY = 0.4
STUDY_BACKGROUND_COLOR = "#B8B8B8"
STUDY_SELECTED_MARKER_SIZE = 10
STUDY_BACKGROUND_MARKER_SIZE = 7

# ---- R vs X scatter ----
RX_SCATTER_HEIGHT_FACTOR = 1.5
RX_TABLE_MARKER_GLYPH = "⬤"
RX_TABLE_MARKER_FONT_SIZE_PX = 36
RX_TABLE_MARKER_COL_WIDTH_PX = 24

_plotly_relayout_listener = components.declare_component(
    "plotly_relayout_listener",
    path=str(os.path.join(os.path.dirname(__file__), "plotly_relayout_listener")),
)
_plotly_selection_bridge = components.declare_component(
    "plotly_selection_bridge",
    path=str(os.path.join(os.path.dirname(__file__), "plotly_selection_bridge")),
)


def plotly_relayout_listener(
    data_id: str,
    plot_count: int = 3,
    plot_ids: Optional[List[str]] = None,
    debounce_ms: int = 120,
    nonce: int = 0,
    reset_token: int = 0,
) -> Optional[Dict[str, object]]:
    # Client-side zoom persistence: binds to Plotly charts and stores axis ranges
    # in browser localStorage. Returns None (no Python roundtrip on zoom).
    return _plotly_relayout_listener(  # type: ignore[misc]
        data_id=str(data_id),
        plot_count=int(plot_count),
        plot_ids=list(plot_ids or []),
        debounce_ms=int(debounce_ms),
        nonce=int(nonce),
        reset_token=int(reset_token),
        bind_tries=int(ZOOM_BIND_TRIES),
        bind_interval_ms=int(ZOOM_BIND_INTERVAL_MS),
        ignore_autorange_ms=int(ZOOM_IGNORE_AUTORANGE_MS),
        key=f"plotly_relayout_listener:{data_id}",
        default=None,
    )


def plotly_selection_bridge(
    data_id: str,
    plot_index: int,
    plot_id: str,
    chart_id: str,
    commit_token: int,
    commit_applied: int,
    clear_token: int,
    reset_token: int = 0,
    selected_marker_size: float = float(STUDY_SELECTED_MARKER_SIZE),
    unselected_marker_opacity: float = 0.30,
) -> Optional[Dict[str, object]]:
    # Lightweight client-side bridge:
    # - stages click selection on an existing Streamlit Plotly chart
    # - emits selected cases only when commit_token changes.
    return _plotly_selection_bridge(  # type: ignore[misc]
        data_id=str(data_id),
        plot_index=int(plot_index),
        plot_id=str(plot_id),
        chart_id=str(chart_id),
        commit_token=int(commit_token),
        commit_applied=int(commit_applied),
        clear_token=int(clear_token),
        reset_token=int(reset_token),
        selected_marker_size=float(selected_marker_size),
        unselected_marker_opacity=float(unselected_marker_opacity),
        key=f"plotly_selection_bridge:{data_id}:{chart_id}:{plot_index}",
        default=None,
    )


def _reset_case_filter_state() -> None:
    # Case filter widgets use deterministic session_state keys; clear them when a new file is loaded
    # so filters start from defaults for the new dataset.
    for k in list(st.session_state.keys()):
        if str(k).startswith("case_part_") or str(k).startswith("case_filters_"):
            try:
                del st.session_state[k]
            except Exception:
                pass


def _note_upload_change() -> None:
    # Called by st.file_uploader(on_change=...): used to trigger filter+zoom reset on any upload action.
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


@st.cache_data(show_spinner=False)
def build_harmonic_shapes(
    n_min: float,
    n_max: float,
    f_base: float,
    show_markers: bool,
    bin_width_hz: float,
) -> Tuple[dict, ...]:
    if not show_markers and (bin_width_hz is None or bin_width_hz <= 0):
        return tuple()
    if not np.isfinite(n_min) or not np.isfinite(n_max) or n_min >= n_max:
        return tuple()
    if not np.isfinite(f_base) or f_base <= 0:
        return tuple()
    shapes: List[dict] = []
    k_start = max(1, int(np.floor(float(n_min))))
    k_end = int(np.ceil(float(n_max)))
    for k in range(k_start, k_end + 1):
        if show_markers:
            shapes.append(
                dict(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=k,
                    x1=k,
                    y0=0,
                    y1=1,
                    line=dict(color="rgba(0,0,0,0.3)", width=1.5),
                )
            )
        if bin_width_hz and bin_width_hz > 0:
            dn = (float(bin_width_hz) / (2.0 * float(f_base)))
            for edge in (k - dn, k + dn):
                shapes.append(
                    dict(
                        type="line",
                        xref="x",
                        yref="paper",
                        x0=edge,
                        x1=edge,
                        y0=0,
                        y1=1,
                        line=dict(color="rgba(0,0,0,0.2)", width=1, dash="dot"),
                    )
                )
    return tuple(shapes)


def _estimate_legend_height_px(n_traces: int, width_px: int, legend_entrywidth: int) -> int:
    usable_w = max(1, int(width_px) - int(LEFT_MARGIN_PX) - int(RIGHT_MARGIN_PX))
    cols = max(1, int(usable_w // max(1, int(legend_entrywidth))))
    rows = int(np.ceil(float(n_traces) / float(cols))) if n_traces > 0 else 0
    row_h = int(np.ceil(float(STYLE["legend_font_size_px"]) * float(LEGEND_ROW_HEIGHT_FACTOR)))
    return int(rows) * int(row_h) + int(LEGEND_PADDING_PX)


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


def parse_study_case_tokens(raw_text: str) -> List[str]:
    """
    Parse pasted case list using mixed separators: tab, space, comma, semicolon, newline.
    Returns deduplicated tokens preserving first-seen order.
    """
    txt = str(raw_text or "").strip()
    if not txt:
        return []
    parts = [p.strip() for p in re.split(r"[\s,;]+", txt) if p and p.strip()]
    seen: Set[str] = set()
    out: List[str] = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _merge_unique_tokens(existing: List[str], incoming: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for tok in list(existing) + list(incoming):
        t = str(tok).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _copy_rx_selection_to_study_text(selected_cases: List[str]) -> None:
    """
    Copy selected scatter cases into the study text area as base-case names.
    Appends uniquely to existing study tokens, preserving order.
    """
    base_tokens: List[str] = []
    seen_base: Set[str] = set()
    for c in selected_cases:
        b = display_case_name(str(c))
        if b and b not in seen_base:
            seen_base.add(b)
            base_tokens.append(b)
    if not base_tokens:
        return
    existing_raw = str(st.session_state.get("study_case_list_text_pending", st.session_state.get("study_case_list_text", "")))
    existing_tokens = parse_study_case_tokens(existing_raw)
    merged = _merge_unique_tokens(existing_tokens, base_tokens)
    st.session_state["study_case_list_text_pending"] = " ".join(merged)


def _clear_study_case_list_text() -> None:
    st.session_state["study_case_list_text"] = ""
    st.session_state.pop("study_case_list_text_pending", None)


def select_study_cases_from_filtered(
    filtered_cases: List[str],
    study_tokens: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Match study tokens (base-case names) against the currently filtered cases.
    Returns:
      - selected full case names (including all matching locations in current filter)
      - unmatched tokens
    """
    if not filtered_cases or not study_tokens:
        return [], []

    base_to_full: Dict[str, List[str]] = {}
    for c in filtered_cases:
        b = display_case_name(c)
        if b not in base_to_full:
            base_to_full[b] = []
        base_to_full[b].append(c)

    selected_set: Set[str] = set()
    unmatched: List[str] = []
    for tok in study_tokens:
        hits = base_to_full.get(tok, [])
        if not hits:
            unmatched.append(tok)
            continue
        for h in hits:
            selected_set.add(h)

    selected = [c for c in filtered_cases if c in selected_set]
    return selected, unmatched


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
def _checkbox_key_map(col_key: str, options_disp: Tuple[str, ...]) -> Dict[str, str]:
    keys: Dict[str, str] = {}
    for o in options_disp:
        h = hashlib.sha1(o.encode("utf-8")).hexdigest()[: int(CHECKBOX_KEY_HASH_LEN)]
        keys[o] = f"{col_key}__opt__{h}"
    return keys


def build_filters_for_case_parts(all_cases: List[str]) -> Tuple[List[str], List[str], int]:
    st.sidebar.header("Case Filters")
    if not all_cases:
        return [], [], -1
    parts_matrix, part_labels = split_case_parts(all_cases)
    if not part_labels:
        return all_cases, [], -1

    reset_all = st.sidebar.button("Reset all filters", key="case_filters_reset_all")

    # Color grouping control lives inside Case Filters (under Reset), and must not reset on new file load.
    # Only case selector states (case_part_*) are reset.
    part_count = max(0, len(part_labels) - 1) if part_labels and part_labels[-1] == "Location" else max(0, len(part_labels))
    color_part_options = ["Auto"] + [f"Case part {i}" for i in range(1, part_count + 1)]
    st.sidebar.markdown("Color by (case part)")
    prev_color_by = st.session_state.get("color_by_case_part", "Auto")
    if isinstance(prev_color_by, str) and prev_color_by not in color_part_options:
        st.session_state["color_by_case_part"] = "Auto"
    color_by_part_label = st.sidebar.selectbox(
        "Color by (case part)",
        options=color_part_options,
        key="color_by_case_part",
        label_visibility="collapsed",
        help="Keeps case colors stable across filters; choose which case part drives the color grouping.",
    )
    st.sidebar.markdown("---")
    hue_part_override = -1 if color_by_part_label == "Auto" else int(color_by_part_label.split()[-1]) - 1

    keep = np.ones(len(all_cases), dtype=bool)
    parts_arr = np.array(parts_matrix, dtype=object)  # shape=(n_cases, n_parts)
    for i, label in enumerate(part_labels):
        col_key = f"case_part_{i+1}_ms"
        options = sorted(set(parts_arr[:, i].tolist()))
        options_disp = [o if o != "" else "<empty>" for o in options]
        is_location_part = bool(label == "Location")

        if is_location_part:
            radio_key = f"{col_key}_single"
            if reset_all or radio_key not in st.session_state or st.session_state.get(radio_key) not in options_disp:
                st.session_state[radio_key] = str(options_disp[0]) if options_disp else "<empty>"
            st.sidebar.markdown(label)
            picked_disp = st.sidebar.radio(
                label="Location",
                options=options_disp,
                key=radio_key,
                label_visibility="collapsed",
            )
            st.session_state[col_key] = [str(picked_disp)]
            selected_raw = ["" if str(picked_disp) == "<empty>" else str(picked_disp)]
            mask_i = np.isin(parts_arr[:, i], selected_raw)
            keep &= mask_i
            if i < len(part_labels) - 1:
                st.sidebar.markdown("---")
            continue

        # init/sanitize
        if reset_all or col_key not in st.session_state:
            st.session_state[col_key] = list(options_disp)
        else:
            st.session_state[col_key] = [v for v in st.session_state[col_key] if v in options_disp]

        st.sidebar.markdown(label)
        c1, _c2 = st.sidebar.columns([1, 1])

        checkbox_keys = _checkbox_key_map(col_key, tuple(options_disp))

        if c1.button("Select all", key=f"{col_key}_all"):
            st.session_state[col_key] = list(options_disp)
            for o in options_disp:
                st.session_state[checkbox_keys[o]] = True

        if _c2.button("Clear all", key=f"{col_key}_none"):
            st.session_state[col_key] = []
            for o in options_disp:
                st.session_state[checkbox_keys[o]] = False

        if reset_all:
            for o in options_disp:
                st.session_state[checkbox_keys[o]] = True

        selected_disp: List[str] = []
        selected_set = set(st.session_state[col_key])
        cols = st.sidebar.columns(2)
        for idx, o in enumerate(options_disp):
            opt_key = checkbox_keys[o]
            if opt_key not in st.session_state:
                st.session_state[opt_key] = o in selected_set
            checked = cols[idx % 2].checkbox(o, key=opt_key)
            if checked:
                selected_disp.append(o)
        st.session_state[col_key] = selected_disp

        if i < len(part_labels) - 1:
            st.sidebar.markdown("---")

        selected_raw = ["" if s == "<empty>" else s for s in selected_disp]
        if 0 < len(selected_raw) < len(options):
            mask_i = np.isin(parts_arr[:, i], selected_raw)
            keep &= mask_i
        if len(selected_raw) == 0:
            keep &= False
    filtered = [c for c, k in zip(all_cases, keep) if k]
    return filtered, part_labels, int(hue_part_override)


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
    selected_cases: Optional[Set[str]] = None,
    show_all_background: bool = False,
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
        study_active = selected_cases is not None
        is_study_selected = (not study_active) or (case in selected_cases)
        dim_case = study_active and bool(show_all_background) and (not is_study_selected)
        color = str(case_colors.get(case, "#1f77b4")) if not dim_case else str(STUDY_BACKGROUND_COLOR)
        line_width = (
            float(STUDY_BACKGROUND_LINE_WIDTH if dim_case else STUDY_SELECTED_LINE_WIDTH)
            if study_active
            else None
        )
        tr_opacity = float(STUDY_BACKGROUND_OPACITY if dim_case else 1.0) if study_active else None
        show_legend = bool(not dim_case)
        line_cfg = dict(color=color)
        if line_width is not None:
            line_cfg["width"] = line_width
        tr = TraceCls(
            x=n,
            y=y,
            customdata=cd,
            mode="lines",
            name=display_case_name(case) if strip_location_suffix else str(case),
            meta={"legend_color": color},
            line=line_cfg,
            opacity=tr_opacity,
            showlegend=show_legend,
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
            if line_width is not None:
                spline_line["width"] = line_width
            tr.update(line=spline_line)
        traces.append(tr)
    return traces, df["Frequency (Hz)"]


def apply_common_layout(
    fig: go.Figure,
    plot_height: int,
    y_title: str,
    legend_entrywidth: int,
    n_traces: int,
    use_auto_width: bool,
    figure_width_px: int,
):
    font_base = dict(family=STYLE["font_family"], color=STYLE["font_color"])
    # Reserve legend space below the plot so the legend stays at the bottom on-page.
    # Increase x-axis reserved space when fonts are larger to reduce title/tick overlap on zoom.
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
    est_width_px = int(figure_width_px) if not use_auto_width else int(AUTO_WIDTH_ESTIMATE_PX)
    legend_h_full = _estimate_legend_height_px(int(n_traces), est_width_px, int(legend_entrywidth))
    legend_h = min(int(WEB_LEGEND_MAX_HEIGHT_PX), int(legend_h_full) + int(WEB_LEGEND_EXTRA_PAD_PX))
    total_height = int(plot_height) + int(TOP_MARGIN_PX) + int(bottom_axis_px) + int(legend_h)
    legend_y = -float(bottom_axis_px) / float(max(1, int(plot_height)))

    # Y-axis overlap fix: keep bottom legend behavior, but grow left margin with font sizes
    # so y tick labels and y title don't collide after zoom.
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

    fig.update_layout(
        autosize=bool(use_auto_width),
        height=total_height,
        # Keep zoom/pan on Streamlit reruns (including when case list changes).
        uirevision="keep",
        font=dict(
            **font_base,
            size=int(STYLE["base_font_size_px"]),
        ),
        margin=dict(
            l=left_margin_px,
            r=RIGHT_MARGIN_PX,
            t=TOP_MARGIN_PX,
            b=int(bottom_axis_px) + int(legend_h),
        ),
        margin_autoexpand=False,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="center",
            x=0.5,
            entrywidth=int(legend_entrywidth),
            entrywidthmode="pixels",
            font=dict(**font_base, size=int(STYLE["legend_font_size_px"])),
        ),
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
                      selected_cases: Optional[Set[str]] = None, show_all_background: bool = False
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
        selected_cases=selected_cases,
        show_all_background=bool(show_all_background),
    )
    fig = go.Figure(data=traces)
    legend_n = sum(1 for t in traces if bool(getattr(t, "showlegend", True)))
    apply_common_layout(fig, plot_height, y_title, legend_entrywidth, legend_n, use_auto_width, figure_width_px)
    return fig, f_series


def build_x_over_r_spline(df_r: Optional[pd.DataFrame], df_x: Optional[pd.DataFrame], cases: List[str], f_base: float,
                          plot_height: int, seq_label: str, smooth: float, legend_entrywidth: int,
                          enable_spline: bool,
                          strip_location_suffix: bool, use_auto_width: bool, figure_width_px: int,
                          case_colors: Dict[str, str], selected_cases: Optional[Set[str]] = None,
                          show_all_background: bool = False
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
            study_active = selected_cases is not None
            is_study_selected = (not study_active) or (case in selected_cases)
            dim_case = study_active and bool(show_all_background) and (not is_study_selected)
            color = str(case_colors.get(case, "#1f77b4")) if not dim_case else str(STUDY_BACKGROUND_COLOR)
            line_width = (
                float(STUDY_BACKGROUND_LINE_WIDTH if dim_case else STUDY_SELECTED_LINE_WIDTH)
                if study_active
                else None
            )
            tr_opacity = float(STUDY_BACKGROUND_OPACITY if dim_case else 1.0) if study_active else None
            show_legend = bool(not dim_case)
            line_cfg = dict(color=color)
            if line_width is not None:
                line_cfg["width"] = line_width
            tr = TraceCls(
                x=n,
                y=y,
                customdata=cd,
                mode="lines",
                name=display_case_name(case) if strip_location_suffix else str(case),
                meta={"legend_color": color},
                line=line_cfg,
                opacity=tr_opacity,
                showlegend=show_legend,
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
                if line_width is not None:
                    spline_line["width"] = line_width
                tr.update(line=spline_line)
            traces.append(tr)
    fig = go.Figure(data=traces)
    y_title = "X1/R1 (unitless)" if seq_label == "Positive" else "X0/R0 (unitless)"
    legend_n = sum(1 for t in traces if bool(getattr(t, "showlegend", True)))
    apply_common_layout(fig, plot_height, y_title, legend_entrywidth, legend_n, use_auto_width, figure_width_px)
    return fig, f_series, xr_dropped, xr_total


def build_rx_scatter_animated(
    df_r: Optional[pd.DataFrame],
    df_x: Optional[pd.DataFrame],
    cases: List[str],
    seq_label: str,
    case_colors: Dict[str, str],
    plot_height: int,
    axis_cases: Optional[List[str]] = None,
) -> Tuple[go.Figure, int, int]:
    fig = go.Figure()
    if df_r is None or df_x is None or not cases:
        fig.update_layout(height=500)
        return fig, 0, 0

    fr, r_map = prepare_sheet_arrays(df_r)
    fx, x_map = prepare_sheet_arrays(df_x)
    if fr.size == 0 or fx.size == 0:
        fig.update_layout(height=500)
        return fig, 0, 0

    freq_candidates = sorted(
        {
            float(v)
            for v in np.concatenate([fr[np.isfinite(fr)], fx[np.isfinite(fx)]], axis=0)
            if np.isfinite(v)
        }
    )
    if not freq_candidates:
        fig.update_layout(height=500)
        return fig, 0, 0
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
            cds.append([display_case_name(case), float(f_used)])
            colors.append(str(case_colors.get(case, "#1f77b4")))
            ids.append(str(display_case_name(case)))

        trace = dict(
            type="scatter",
            x=xs,
            y=ys,
            mode="markers",
            name="Cases",
            customdata=cds,
            ids=ids,
            hovertemplate="Case=%{customdata[0]}<br>f=%{customdata[1]:.1f} Hz<br>R=%{x}<br>X=%{y}<extra></extra>",
            marker=dict(
                color=colors,
                size=float(STUDY_SELECTED_MARKER_SIZE),
                opacity=1.0,
                line=dict(width=0),
            ),
            showlegend=False,
            meta={"kind": "points"},
        )
        return trace, len(xs)

    f0 = float(freq_candidates[init_idx])
    tr0, n0 = frame_data_for_freq_idx(init_idx)
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
    return fig, n0, len(freq_candidates)


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


def _render_rx_selected_cases_table(
    selected_display: List[str],
    filtered_cases: List[str],
    case_colors_scatter: Dict[str, str],
) -> None:
    if not selected_display:
        return
    st.caption(f"Scatter selected cases: {len(selected_display)}")
    base_color_map: Dict[str, str] = {}
    for c in filtered_cases:
        b = display_case_name(c)
        if b not in base_color_map:
            base_color_map[b] = str(case_colors_scatter.get(c, "#1f77b4"))

    marker_col = str(RX_TABLE_MARKER_GLYPH)
    marker_w_px = int(RX_TABLE_MARKER_COL_WIDTH_PX)
    selected_df = pd.DataFrame(
        {
            marker_col: [str(base_color_map.get(name, "#1f77b4")) for name in selected_display],
            "Case": selected_display,
        }
    )
    styled_selected_df = (
        selected_df.style
        .format({marker_col: lambda _: marker_col})
        .map(
            lambda v: (
                f"color: {v}; font-size: {int(RX_TABLE_MARKER_FONT_SIZE_PX)}px; "
                "text-align: center; line-height: 1;"
            ),
            subset=[marker_col],
        )
        .set_table_styles(
            [
                {
                    "selector": "th.col_heading.level0.col0",
                    "props": [
                        ("width", f"{marker_w_px}px"),
                        ("min-width", f"{marker_w_px}px"),
                        ("max-width", f"{marker_w_px}px"),
                    ],
                },
                {
                    "selector": "td.col0",
                    "props": [
                        ("width", f"{marker_w_px}px"),
                        ("min-width", f"{marker_w_px}px"),
                        ("max-width", f"{marker_w_px}px"),
                        ("padding-left", "2px"),
                        ("padding-right", "2px"),
                    ],
                },
            ]
        )
    )
    # Use Styler widths as the single source of truth (avoid conflicting column width systems).
    st.dataframe(styled_selected_df, use_container_width=True, hide_index=True)


def _render_client_png_download(
    filename: str,
    scale: int,
    button_label: str,
    plot_height: int,
    legend_entrywidth: int,
    plot_index: int,
):
    dom_id = hashlib.sha1(f"{filename}|{scale}|{plot_height}|{legend_entrywidth}|{plot_index}".encode("utf-8")).hexdigest()[
        : int(EXPORT_DOM_ID_HASH_LEN)
    ]
    html = f"""
    <div id="exp-{dom_id}">
      <button id="btn-{dom_id}" style="padding:6px 10px; font-size: 0.9rem; cursor:pointer;">
        {button_label}
      </button>
      <div id="plot-{dom_id}" style="width:1px; height:1px; position:absolute; left:-99999px; top:-99999px;"></div>
    </div>
    <script>
      const scale = {int(scale)};
      const plotHeight = {int(plot_height)};
      const topMargin = {int(TOP_MARGIN_PX)};
      const bottomAxis = {int(BOTTOM_AXIS_PX)};
      const legendPad = {int(LEGEND_PADDING_PX)};
      const legendEntryWidth = {int(legend_entrywidth)};
      const plotIndex = {int(plot_index)};
      const filename = {json.dumps(filename)};
      const fallbackLegendFontSize = {int(STYLE["legend_font_size_px"])};

      async function doExport() {{
        try {{
          const Plotly = window.parent?.Plotly;
          if (!Plotly) return;
          const plots = window.parent?.document?.querySelectorAll?.("div.js-plotly-plot");
          if (!plots || plots.length <= plotIndex) return;
          const gd = plots[plotIndex];
          if (!gd) return;

          const r = gd.getBoundingClientRect();
          const widthPx = Math.floor(r.width || 0);
          if (!widthPx) return;

          const legendFontSize =
            gd?._fullLayout?.legend?.font?.size ||
            gd?._fullLayout?.font?.size ||
            fallbackLegendFontSize;
          // Legend row height: keep tight to avoid bottom whitespace.
          const legendRowH = Math.ceil(legendFontSize * {float(EXPORT_LEGEND_ROW_HEIGHT_FACTOR)});
      const legendFontFamily = {json.dumps(STYLE["font_family"])};
      const legendFontColor = {json.dumps(STYLE["font_color"])};

      const leftMarginBase = {int(LEFT_MARGIN_PX)};
      const rightMargin = {int(RIGHT_MARGIN_PX)};
      const tickFontSize = {int(STYLE["tick_font_size_px"])};
      const axisTitleFontSize = {int(STYLE["axis_title_font_size_px"])};
      const leftMarginPx = Math.max(leftMarginBase, Math.round(tickFontSize * {float(LEFT_MARGIN_TICK_MULT)} + axisTitleFontSize * {float(LEFT_MARGIN_TITLE_MULT)}));

          const data = Array.isArray(gd.data) ? gd.data : [];
          const data2 = data.map((tr) => {{
            const t = Object.assign({{}}, tr);
            if (t.type === "scattergl") t.type = "scatter";
            return t;
          }});
          const legendItems = [];
          for (const tr of data2) {{
            if (tr && tr.showlegend === false) continue;
            const name = tr && tr.name ? String(tr.name) : "";
            if (!name) continue;
            // Prefer the actual trace styling (so export always matches on-page plot).
            const color =
              (tr.line && tr.line.color) ? tr.line.color :
              (tr.marker && tr.marker.color) ? tr.marker.color :
              (tr.meta && tr.meta.legend_color) ? tr.meta.legend_color :
              {json.dumps(EXPORT_FALLBACK_COLOR)};
            legendItems.push({{name, color}});
          }}

          const usableW = Math.max(1, widthPx - leftMarginPx - rightMargin);

          // Estimate needed entry width using canvas text measurement to avoid overlap for long names.
          let maxTextW = 0;
          try {{
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            if (ctx) {{
              ctx.font = legendFontSize + "px " + legendFontFamily;
              for (const it of legendItems) {{
                const w = ctx.measureText(it.name).width || 0;
                if (w > maxTextW) maxTextW = w;
              }}
            }}
          }} catch (e) {{}}

          const sampleLinePx = Math.max({int(EXPORT_SAMPLE_LINE_MIN_PX)}, Math.round({float(EXPORT_SAMPLE_LINE_MULT)} * legendFontSize));
          const sampleGapPx = Math.max({int(EXPORT_SAMPLE_GAP_MIN_PX)}, Math.round({float(EXPORT_SAMPLE_GAP_MULT)} * legendFontSize));
          const textPadPx = Math.max({int(EXPORT_TEXT_PAD_MIN_PX)}, Math.round({float(EXPORT_TEXT_PAD_MULT)} * legendFontSize));
          const neededEntryPx = Math.ceil(sampleLinePx + sampleGapPx + maxTextW + textPadPx);
          const entryPx = Math.max(1, Math.max(legendEntryWidth, neededEntryPx));

          const cols = Math.max(1, Math.floor(usableW / entryPx));
          const rows = Math.ceil(legendItems.length / cols);
          // Total legend area in bottom margin.
          // Add a small tail so the last row doesn't look cramped, but avoid large blank space.
          const legendH = (rows * legendRowH) + legendPad + Math.ceil({float(EXPORT_LEGEND_TAIL_FONT_MULT)} * legendFontSize);

          const newHeight = plotHeight + topMargin + bottomAxis + legendH;
          const newMarginB = bottomAxis + legendH;

          const container = document.getElementById("plot-{dom_id}");
          if (!container) return;
          container.style.width = widthPx + "px";
          container.style.height = newHeight + "px";

          const baseLayout = Object.assign({{}}, gd.layout || {{}});
          baseLayout.width = widthPx;
          baseLayout.height = newHeight;
          baseLayout.autosize = false;
          baseLayout.margin = Object.assign({{}}, baseLayout.margin || {{}});
          baseLayout.margin.t = topMargin;
          baseLayout.margin.l = leftMarginPx;
          baseLayout.margin.r = rightMargin;
          baseLayout.margin.b = newMarginB;
          // Disable Plotly legend and draw a manual legend in the bottom margin so it never scrolls/clips.
          baseLayout.showlegend = false;
          for (const tr of data2) {{
            tr.showlegend = false;
          }}

          const ann = Array.isArray(baseLayout.annotations) ? baseLayout.annotations.slice() : [];
          const shp = Array.isArray(baseLayout.shapes) ? baseLayout.shapes.slice() : [];

          // Spread columns across the available width to avoid side whitespace and tight columns.
          const colW = usableW / cols;
          const xPadPx = Math.max(0, Math.min({int(EXPORT_COL_PADDING_MAX_PX)}, Math.floor(colW * {float(EXPORT_COL_PADDING_FRAC)})));

          for (let i = 0; i < legendItems.length; i++) {{
            const row = Math.floor(i / cols);
            const col = i % cols;
            const x0 = (col * colW + xPadPx) / usableW;
            const x1 = x0 + (sampleLinePx / usableW);
            const y = -(bottomAxis + legendPad + (row + {float(EXPORT_LEGEND_ROW_Y_OFFSET)}) * legendRowH) / Math.max(1, plotHeight);

            shp.push({{
              type: "line",
              xref: "paper",
              yref: "paper",
              x0, x1,
              y0: y, y1: y,
              line: {{color: legendItems[i].color, width: 2}}
            }});

            ann.push({{
              xref: "paper",
              yref: "paper",
              x: x1 + (sampleGapPx / usableW),
              y,
              xanchor: "left",
              yanchor: "middle",
              showarrow: false,
              align: "left",
              text: legendItems[i].name,
              font: {{size: legendFontSize, family: legendFontFamily, color: legendFontColor}}
            }});
          }}

          baseLayout.annotations = ann;
          baseLayout.shapes = shp;

          await Plotly.newPlot(container, data2, baseLayout, {{displayModeBar: false, staticPlot: true}});
          const url = await Plotly.toImage(container, {{format: "png", width: widthPx, height: newHeight, scale}});
          const a = document.createElement("a");
          a.href = url;
          a.download = filename;
          document.body.appendChild(a);
          a.click();
          a.remove();
        }} catch (e) {{
        }} finally {{
          try {{
            const container = document.getElementById("plot-{dom_id}");
            if (container) {{
              container.innerHTML = "";
              container.style.width = "1px";
              container.style.height = "1px";
            }}
          }} catch (e) {{}}
        }}
      }}

      document.getElementById("btn-{dom_id}").addEventListener("click", doExport);
    </script>
    """
    components.html(html, height=70)


def _render_rx_client_step_controls(seq_label: str):
    dom_id = hashlib.sha1(f"rx-step:{seq_label}".encode("utf-8")).hexdigest()[: int(EXPORT_DOM_ID_HASH_LEN)]
    seq_js = json.dumps(str(seq_label))
    html = f"""
    <div id="rx-step-{dom_id}" style="display:flex; gap:8px; align-items:center;">
      <button id="rx-prev-{dom_id}" style="padding:6px 10px; font-size: 1rem; line-height:1; cursor:pointer;">&#8592;</button>
      <button id="rx-next-{dom_id}" style="padding:6px 10px; font-size: 1rem; line-height:1; cursor:pointer;">&#8594;</button>
    </div>
    <script>
      const seqLabel = {seq_js};
      function getRxPlot() {{
        try {{
          const plots = window.parent?.document?.querySelectorAll?.("div.js-plotly-plot");
          if (!plots) return null;
          for (const p of plots) {{
            const ui = p?.layout?.uirevision ? String(p.layout.uirevision) : "";
            if (ui === "rx:" + seqLabel || ui.startsWith("rx:")) return p;
          }}
        }} catch (e) {{}}
        return null;
      }}

      async function stepBy(delta) {{
        try {{
          const gd = getRxPlot();
          if (!gd) return;
          const Plotly = window.parent?.Plotly || gd?.ownerDocument?.defaultView?.Plotly;
          if (!Plotly || !Plotly.animate) return;
          const sliders = Array.isArray(gd?.layout?.sliders) ? gd.layout.sliders : [];
          if (!sliders.length) return;
          const active = Number(sliders[0].active || 0);
          const frames =
            (gd?._transitionData && Array.isArray(gd._transitionData._frames) && gd._transitionData._frames) ||
            (Array.isArray(gd?.frames) ? gd.frames : []);
          const n = frames.length;
          if (!Number.isFinite(active) || n <= 0) return;
          const next = Math.max(0, Math.min(n - 1, active + (delta > 0 ? 1 : -1)));
          if (next === active) return;
          const frameObj = frames[next];
          const frameName = frameObj && frameObj.name != null ? String(frameObj.name) : String(next);
          await Plotly.animate(gd, [frameName], {{
            mode: "immediate",
            frame: {{ duration: 0, redraw: false }},
            transition: {{ duration: 0 }}
          }});
        }} catch (e) {{}}
      }}

      document.getElementById("rx-prev-{dom_id}")?.addEventListener("click", () => stepBy(-1));
      document.getElementById("rx-next-{dom_id}")?.addEventListener("click", () => stepBy(1));
    </script>
    """
    components.html(html, height=46)


def main():
    st.title("FS Sweep Visualizer (Spline)")

    # Data source
    default_path = "FS_sweep.xlsx"
    st.sidebar.header("Data Source")
    up = st.sidebar.file_uploader(
        "Upload Excel",
        type=["xlsx"],
        key="xlsx_uploader",
        on_change=_note_upload_change,
        help="If empty, loads 'FS_sweep.xlsx' from this folder.",
    )
    st.sidebar.markdown("---")
    data_id = "unknown"
    try:
        if up is not None:
            data = load_fs_sweep_xlsx(up)
            try:
                cached = st.session_state.get("uploaded_file_sha1_10")
                data_id = str(cached) if cached else hashlib.sha1(up.getvalue()).hexdigest()[:10]
            except Exception:
                data_id = f"upload:{getattr(up, 'name', 'file')}"
        elif os.path.exists(default_path):
            data = load_fs_sweep_xlsx(default_path)
            try:
                data_id = f"local:{int(os.path.getmtime(default_path))}"
            except Exception:
                data_id = "local"
            st.sidebar.info(f"Loaded local file: {default_path}")
        else:
            st.warning("Upload an Excel file or place 'FS_sweep.xlsx' here.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to load Excel: {e}")
        st.stop()

    # Reset case-part/location filters on:
    # - any upload action (even if the same file is uploaded again)
    # - a change of the effective loaded dataset (data_id)
    last_upload_handled = int(st.session_state.get("upload_nonce_handled", 0))
    upload_nonce = int(st.session_state.get("upload_nonce", 0))
    prev_data_id = st.session_state.get("active_data_id")
    if (upload_nonce != last_upload_handled) or (prev_data_id != data_id):
        _reset_case_filter_state()
        st.session_state["upload_nonce_handled"] = upload_nonce
        st.session_state["active_data_id"] = data_id

    # Controls
    st.sidebar.header("Controls")
    seq_label = st.sidebar.radio("Sequence", ["Positive", "Zero"], index=0)
    seq = ("R1", "X1") if seq_label == "Positive" else ("R0", "X0")
    base_label = st.sidebar.radio("Base frequency", ["50 Hz", "60 Hz"], index=0)
    f_base = 50.0 if base_label.startswith("50") else 60.0
    plot_height = st.sidebar.slider("Plot area height (px)", min_value=100, max_value=1000, value=400, step=25)
    use_auto_width = st.sidebar.checkbox("Auto width (fit container)", value=True)
    figure_width_px = DEFAULT_FIGURE_WIDTH_PX
    if not use_auto_width:
        figure_width_px = st.sidebar.slider("Figure width (px)", min_value=800, max_value=2200, value=DEFAULT_FIGURE_WIDTH_PX, step=50)

    enable_spline = st.sidebar.checkbox("Spline (slow)", value=False)
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

    # Prepare cases list early so the Case Filters section can offer a "color by case part" control.
    df_r = data.get(seq[0])
    df_x = data.get(seq[1])
    if df_r is None and df_x is None:
        st.error(f"Missing sheets for sequence '{seq_label}' ({seq[0]}/{seq[1]}).")
        st.stop()

    all_cases = sorted(list({*list_case_columns(df_r), *list_case_columns(df_x)}))

    st.sidebar.header("Show plots")
    show_plot_x = st.sidebar.checkbox("X", value=True)
    show_plot_r = st.sidebar.checkbox("R", value=False)
    show_plot_xr = st.sidebar.checkbox("X/R", value=False)
    show_plot_rx = st.sidebar.checkbox("R vs X scatter", value=False)
    if not (show_plot_x or show_plot_r or show_plot_xr or show_plot_rx):
        st.warning("Select at least one plot to display.")
        st.stop()
    st.sidebar.markdown("---")

    # Legend/Export controls
    st.sidebar.header("Legend & Export")
    auto_legend_entrywidth = st.sidebar.checkbox("Auto legend column width", value=True)
    legend_entrywidth = 180
    if not auto_legend_entrywidth:
        legend_entrywidth = st.sidebar.slider("Legend column width (px)", min_value=50, max_value=300, value=180, step=10)

    # Keep the download buttons visually within the "Legend & Export" section,
    # but fill their contents later once figures are built.
    download_area = st.sidebar.container()
    st.sidebar.markdown("---")

    download_config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": "plot",
            "scale": int(EXPORT_IMAGE_SCALE),
        }
    }

    # Cases / filters
    if (show_plot_r or show_plot_xr or show_plot_rx) and df_r is None:
        st.error(f"Sheet '{seq[0]}' is missing, but R, X/R and/or R vs X scatter is enabled.")
        st.stop()
    if (show_plot_x or show_plot_xr or show_plot_rx) and df_x is None:
        st.error(f"Sheet '{seq[1]}' is missing, but X, X/R and/or R vs X scatter is enabled.")
        st.stop()

    filtered_cases, part_labels, hue_part_override = build_filters_for_case_parts(all_cases)
    if not filtered_cases:
        st.warning("No cases after filtering. Adjust filters.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.header("Study Selection")
    # Apply deferred text updates before the widget is instantiated in this rerun.
    if "study_case_list_text_pending" in st.session_state:
        st.session_state["study_case_list_text"] = str(st.session_state.pop("study_case_list_text_pending", ""))
    # Keep Study list consistent with Case Filters:
    # remove tokens that are known dataset case names but excluded by current case-part/location filters.
    all_base_names = {display_case_name(c) for c in all_cases}
    filtered_base_names = {display_case_name(c) for c in filtered_cases}
    current_study_raw = str(st.session_state.get("study_case_list_text", ""))
    current_study_tokens = parse_study_case_tokens(current_study_raw)
    pruned_study_tokens = [
        t for t in current_study_tokens
        if (t not in all_base_names) or (t in filtered_base_names)
    ]
    if pruned_study_tokens != current_study_tokens:
        st.session_state["study_case_list_text"] = " ".join(pruned_study_tokens)
    st.sidebar.button(
        "Clear study list",
        key="study_clear_list_btn",
        on_click=_clear_study_case_list_text,
    )
    study_raw = st.sidebar.text_area(
        "Study case list",
        value="",
        key="study_case_list_text",
        help="Paste base case names. Separators: tab, space, comma, semicolon, newline.",
        placeholder="case1 case2; case3, case4",
    )
    show_all_background = st.sidebar.checkbox(
        "Show all as background",
        value=True,
        key="study_show_all_background",
        help="If enabled, non-selected filtered cases are drawn as thin gray background lines.",
    )
    study_tokens = parse_study_case_tokens(study_raw)
    selected_study_cases: List[str] = []
    unmatched_study_tokens: List[str] = []
    if study_tokens:
        selected_study_cases, unmatched_study_tokens = select_study_cases_from_filtered(
            filtered_cases,
            study_tokens,
        )
        if unmatched_study_tokens:
            preview = ", ".join(unmatched_study_tokens[:8])
            suffix = "" if len(unmatched_study_tokens) <= 8 else f", +{len(unmatched_study_tokens) - 8} more"
            st.sidebar.warning(f"Unmatched study names: {preview}{suffix}")
        if not selected_study_cases:
            st.warning("Study list provided, but none of the names match the currently filtered cases.")
            st.stop()

    study_mode = bool(study_tokens)
    selected_study_set: Optional[Set[str]] = set(selected_study_cases) if study_mode else None
    line_study_mode = bool(study_mode)
    selected_line_set: Optional[Set[str]] = set(selected_study_cases) if line_study_mode else None
    cases_for_line = list(filtered_cases) if (line_study_mode and show_all_background) else (
        list(selected_study_cases) if line_study_mode else list(filtered_cases)
    )
    strip_location_suffix = True

    if auto_legend_entrywidth:
        legend_cases = list(selected_study_cases) if line_study_mode else list(cases_for_line)
        display_names = [display_case_name(c) for c in legend_cases]
        max_len = max((len(n) for n in display_names), default=12)
        legend_font_px = int(STYLE["legend_font_size_px"])
        approx_char_px = max(7, int(round(0.60 * float(legend_font_px))))
        base_px = max(44, int(round(3.5 * float(legend_font_px))))  # symbol + padding inside a legend item

        # Only used to cap export legend columns; use the configured width when available.
        est_width_px = int(figure_width_px)
        usable_w = max(1, int(est_width_px) - int(LEFT_MARGIN_PX) - int(RIGHT_MARGIN_PX))
        desired = int(max_len) * int(approx_char_px) + int(base_px)
        legend_entrywidth = _clamp_int(desired, 50, min(900, usable_w))
        if legend_entrywidth >= int(usable_w * 0.95):
            legend_entrywidth = usable_w

    # Stable colors: generate per-file mapping from all cases (not just filtered cases),
    # and then look up colors for the currently filtered set.
    all_case_colors = cached_clustered_case_colors(tuple(all_cases), int(hue_part_override))
    case_colors_line = {c: all_case_colors.get(c, "#1f77b4") for c in cases_for_line}
    case_colors_scatter = {c: all_case_colors.get(c, "#1f77b4") for c in filtered_cases}

    # Harmonic decorations
    st.sidebar.markdown("---")
    st.sidebar.header("Harmonics")
    show_harmonics = st.sidebar.checkbox("Show harmonic lines", value=True)
    bin_width_hz = st.sidebar.number_input("Bin width (Hz)", min_value=0.0, value=0.0, step=1.0, help="0 disables tolerance bands")
    st.sidebar.markdown("---")

    # Client-side zoom persistence: bind to shown Streamlit Plotly charts and store axis ranges
    # in the browser (localStorage). No Streamlit rerun is triggered by zooming.
    plot_order: List[str] = []
    if show_plot_x:
        plot_order.append("x")
    if show_plot_r:
        plot_order.append("r")
    if show_plot_xr:
        plot_order.append("xr")
    if show_plot_rx:
        plot_order.append("rx")

    bind_nonce_key = f"zoom_bind_nonce:{data_id}"
    bind_sig_key = f"zoom_bind_sig:{data_id}"
    bind_sig = f"{data_id}|{','.join(plot_order)}|{int(upload_nonce)}"
    if str(st.session_state.get(bind_sig_key, "")) != bind_sig:
        st.session_state[bind_sig_key] = bind_sig
        st.session_state[bind_nonce_key] = int(st.session_state.get(bind_nonce_key, 0)) + 1

    plotly_relayout_listener(
        data_id=data_id,
        plot_count=len(plot_order),
        plot_ids=plot_order,
        debounce_ms=150,
        nonce=int(st.session_state.get(bind_nonce_key, 0)),
        reset_token=int(upload_nonce),
    )

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
            "sel": sorted(list(selected_line_set)) if selected_line_set is not None else None,
            "show_bg": bool(line_study_mode and show_all_background),
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
                selected_cases=selected_line_set,
                show_all_background=bool(line_study_mode and show_all_background),
            )
            st.session_state[x_cache_sig_key] = x_sig
            st.session_state[x_cache_fig_key] = fig_x_built.to_dict()
        fig_x = go.Figure(st.session_state.get(x_cache_fig_key, {}))
        f_x = df_x["Frequency (Hz)"] if df_x is not None else None
        plot_items.append(_make_plot_item("x", fig_x, f_x, "X_full_legend.png", "X\nPNG", "plot_x"))

    if show_plot_r:
        r_cache_sig_key = f"line_fig_sig:{data_id}:{seq_label}:r"
        r_cache_fig_key = f"line_fig_cache:{data_id}:{seq_label}:r"
        r_sig_payload = {
            "kind": "r",
            "cases": list(cases_for_line),
            "sel": sorted(list(selected_line_set)) if selected_line_set is not None else None,
            "show_bg": bool(line_study_mode and show_all_background),
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
                selected_cases=selected_line_set,
                show_all_background=bool(line_study_mode and show_all_background),
            )
            st.session_state[r_cache_sig_key] = r_sig
            st.session_state[r_cache_fig_key] = fig_r_built.to_dict()
        fig_r = go.Figure(st.session_state.get(r_cache_fig_key, {}))
        f_r = df_r["Frequency (Hz)"] if df_r is not None else None
        plot_items.append(_make_plot_item("r", fig_r, f_r, "R_full_legend.png", "R\nPNG", "plot_r"))

    if show_plot_xr:
        xr_cache_sig_key = f"line_fig_sig:{data_id}:{seq_label}:xr"
        xr_cache_fig_key = f"line_fig_cache:{data_id}:{seq_label}:xr"
        xr_cache_meta_key = f"line_fig_meta:{data_id}:{seq_label}:xr"
        xr_sig_payload = {
            "kind": "xr",
            "cases": list(cases_for_line),
            "sel": sorted(list(selected_line_set)) if selected_line_set is not None else None,
            "show_bg": bool(line_study_mode and show_all_background),
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
                selected_cases=selected_line_set,
                show_all_background=bool(line_study_mode and show_all_background),
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
        plot_items.append(_make_plot_item("xr", fig_xr, f_xr, "X_over_R_full_legend.png", "X/R\nPNG", "plot_xr"))

    f_refs = [it["f_ref"] for it in plot_items if it.get("f_ref") is not None]
    n_lo, n_hi = compute_common_n_range(f_refs, f_base)
    harm_shapes = build_harmonic_shapes(n_lo, n_hi, f_base, show_harmonics, bin_width_hz)
    for it in plot_items:
        fig = it["fig"]
        if isinstance(fig, go.Figure):
            fig.update_xaxes(range=[n_lo, n_hi])
            if harm_shapes:
                fig.update_layout(shapes=(fig.layout.shapes + harm_shapes) if fig.layout.shapes else harm_shapes)

    # Render
    st.subheader(f"Sequence: {seq_label} | Base: {int(f_base)} Hz")
    if study_mode:
        matched_total = len(selected_study_cases)
        token_total = len(study_tokens)
        line_msg = "Line plots highlighted."
        scatter_msg = "Background shown." if show_all_background else "Only selected shown."
        st.caption(
            f"Study selection active: {matched_total} matched case(s) from {token_total} token(s). "
            f"{line_msg} Scatter: {scatter_msg}"
        )
    if show_plot_xr and xr_total > 0 and xr_dropped > 0:
        st.caption(f"X/R: dropped {xr_dropped} of {xr_total} points where |R| < {XR_EPS_DISPLAY} or data missing.")

    export_scale = int(EXPORT_IMAGE_SCALE)
    with download_area:
        st.subheader("Download (Full Legend)")
        if plot_items:
            st.caption("Browser PNG download (temporarily expands the on-page chart legend, then downloads).")
            cols = st.columns(len(plot_items))
            for idx, it in enumerate(plot_items):
                with cols[idx]:
                    _render_client_png_download(
                        filename=str(it["filename"]),
                        scale=export_scale,
                        button_label=str(it["button_label"]),
                        plot_height=plot_height,
                        legend_entrywidth=legend_entrywidth,
                        plot_index=int(idx),
                    )
        else:
            st.caption("No X, R, or X/R line plots selected for full-legend download.")

    for idx, it in enumerate(plot_items):
        fig = it["fig"]
        chart_key = str(it["chart_key"])
        if isinstance(fig, go.Figure):
            st.plotly_chart(fig, use_container_width=bool(use_auto_width), config=download_config, key=chart_key)
        if idx < len(plot_items) - 1:
            st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)

    if show_plot_rx:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.subheader("R vs X Scatter")
        st.caption("Use the in-chart frequency slider to animate point motion.")

        rx_state_key = f"rx_selected_cases:{data_id}:{seq_label}"
        rx_clear_token_key = f"rx_clear_token:{data_id}:{seq_label}"
        rx_commit_token_key = f"rx_commit_token:{data_id}:{seq_label}"
        rx_commit_applied_key = f"rx_commit_applied:{data_id}:{seq_label}"
        rx_commit_pending_key = f"rx_commit_pending:{data_id}:{seq_label}"
        rx_commit_retry_left_key = f"rx_commit_retry_left:{data_id}:{seq_label}"
        rx_filter_sig_key = f"rx_filter_sig:{data_id}:{seq_label}"
        rx_fig_sig_key = f"rx_fig_sig:{data_id}:{seq_label}"
        rx_fig_cache_key = f"rx_fig_cache:{data_id}:{seq_label}"
        rx_fig_points_key = f"rx_fig_points:{data_id}:{seq_label}"
        rx_fig_steps_key = f"rx_fig_steps:{data_id}:{seq_label}"
        rx_clear_token = int(st.session_state.get(rx_clear_token_key, 0))
        rx_commit_token = int(st.session_state.get(rx_commit_token_key, 0))
        rx_commit_applied = int(st.session_state.get(rx_commit_applied_key, 0))

        filter_sig = hashlib.sha1("|".join(sorted(filtered_cases)).encode("utf-8")).hexdigest()[:12]
        prev_filter_sig = str(st.session_state.get(rx_filter_sig_key, ""))
        all_base_set = {display_case_name(c) for c in all_cases}
        filtered_base_set = {display_case_name(c) for c in filtered_cases}
        prev_committed = [display_case_name(str(v)) for v in st.session_state.get(rx_state_key, [])]
        committed_clean = sorted({c for c in prev_committed if c in all_base_set})
        st.session_state[rx_state_key] = committed_clean
        if prev_filter_sig != filter_sig:
            st.session_state[rx_filter_sig_key] = filter_sig
            st.session_state.pop(rx_fig_sig_key, None)
            st.session_state.pop(rx_fig_cache_key, None)
            st.session_state.pop(rx_fig_points_key, None)
            st.session_state.pop(rx_fig_steps_key, None)

        committed_before = sorted({display_case_name(str(v)) for v in st.session_state.get(rx_state_key, []) if display_case_name(str(v)) in all_base_set})
        st.session_state[rx_state_key] = committed_before

        # Location-based baseline for scatter axis limits:
        # keep axes stable when case-part filters change within the selected location.
        location_cases_for_axes = list(filtered_cases)
        if part_labels and part_labels[-1] == "Location":
            loc_key = f"case_part_{len(part_labels)}_ms"
            loc_selected_disp = st.session_state.get(loc_key, [])
            loc_selected_raw = ["" if s == "<empty>" else str(s) for s in loc_selected_disp]
            if loc_selected_raw:
                selected_loc = str(loc_selected_raw[0])
                location_cases_for_axes = [
                    c for c in all_cases
                    if str((split_case_location(c)[1] or "")) == selected_loc
                ]

        rx_sig_payload = {
            "seq": str(seq_label),
            "plot_h": int(plot_height),
            "cases": list(filtered_cases),
            "axis_cases": list(location_cases_for_axes),
            "colors": [[str(c), str(case_colors_scatter.get(c, "#1f77b4"))] for c in filtered_cases],
        }
        rx_sig = hashlib.sha1(json.dumps(rx_sig_payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]

        if st.session_state.get(rx_fig_sig_key) != rx_sig or (rx_fig_cache_key not in st.session_state):
            rx_fig_built, rx_points_built, rx_steps_built = build_rx_scatter_animated(
                df_r=df_r,
                df_x=df_x,
                cases=list(filtered_cases),
                seq_label=seq_label,
                case_colors=case_colors_scatter,
                plot_height=int(plot_height),
                axis_cases=list(location_cases_for_axes),
            )
            st.session_state[rx_fig_sig_key] = rx_sig
            st.session_state[rx_fig_cache_key] = rx_fig_built.to_dict()
            st.session_state[rx_fig_points_key] = int(rx_points_built)
            st.session_state[rx_fig_steps_key] = int(rx_steps_built)

        rx_fig = go.Figure(st.session_state.get(rx_fig_cache_key, {}))
        rx_fig.update_layout(uirevision=f"rx:{seq_label}")
        rx_points_count = int(st.session_state.get(rx_fig_points_key, 0))
        rx_freq_steps = int(st.session_state.get(rx_fig_steps_key, 0))

        st.plotly_chart(rx_fig, use_container_width=bool(use_auto_width), config=download_config, key="plot_rx")

        left_pad, cnav, c1, c2, c3, right_pad = st.columns([0.8, 1.6, 1.5, 2.1, 2.4, 0.8])
        with cnav:
            _render_rx_client_step_controls(seq_label)
        with c1:
            rx_clear_selection = st.button("Clear list", key="rx_clear_selection_btn", use_container_width=True)
        with c2:
            rx_show_selection = st.button("Show list selection", key="rx_show_list_selection_btn", use_container_width=True)
        with c3:
            rx_copy_to_study = st.button(
                "Copy selection to study list",
                key="rx_copy_selection_to_study_btn",
                use_container_width=True,
            )

        if rx_clear_selection:
            st.session_state[rx_state_key] = []
            rx_clear_token += 1
            st.session_state[rx_clear_token_key] = int(rx_clear_token)
        if rx_show_selection:
            rx_commit_token += 1
            st.session_state[rx_commit_token_key] = int(rx_commit_token)
            st.session_state[rx_commit_pending_key] = int(rx_commit_token)
            # One automatic retry rerun is enough in practice to absorb initial bind latency.
            st.session_state[rx_commit_retry_left_key] = 1
        if rx_copy_to_study:
            to_copy = sorted({display_case_name(str(v)) for v in st.session_state.get(rx_state_key, [])})
            if to_copy:
                _copy_rx_selection_to_study_text(to_copy)
                # Needed so pending Study list text is applied before sidebar widget creation
                # and line plots are rebuilt with the new Study selection.
                st.rerun()

        bridge_payload = plotly_selection_bridge(
            data_id=data_id,
            plot_index=int(len(plot_items)),
            plot_id="rx",
            chart_id=f"rx:{seq_label}",
            commit_token=int(rx_commit_token),
            commit_applied=int(rx_commit_applied),
            clear_token=int(rx_clear_token),
            reset_token=int(upload_nonce),
            selected_marker_size=float(STUDY_SELECTED_MARKER_SIZE),
            unselected_marker_opacity=0.30,
        )

        if isinstance(bridge_payload, dict):
            try:
                payload_commit_tok = int(bridge_payload.get("commit_token", -1))
            except Exception:
                payload_commit_tok = -1
            payload_cases_raw = bridge_payload.get("selected_cases")
            if (
                payload_commit_tok > 0
                and payload_commit_tok == int(rx_commit_token)
                and payload_commit_tok > int(rx_commit_applied)
                and isinstance(payload_cases_raw, list)
            ):
                st.session_state[rx_commit_applied_key] = int(payload_commit_tok)
                committed_new = sorted(
                    {
                        display_case_name(str(v))
                        for v in payload_cases_raw
                        if str(v).strip() != "" and display_case_name(str(v)) in all_base_set
                    }
                )
                st.session_state[rx_state_key] = committed_new
                if int(st.session_state.get(rx_commit_pending_key, 0)) == int(payload_commit_tok):
                    st.session_state[rx_commit_pending_key] = 0
                    st.session_state[rx_commit_retry_left_key] = 0

        # First-click reliability: if commit is pending and not yet applied, auto-rerun once.
        pending_tok = int(st.session_state.get(rx_commit_pending_key, 0))
        retry_left = int(st.session_state.get(rx_commit_retry_left_key, 0))
        applied_tok = int(st.session_state.get(rx_commit_applied_key, 0))
        if pending_tok > 0 and applied_tok < pending_tok and retry_left > 0:
            st.session_state[rx_commit_retry_left_key] = int(retry_left - 1)
            st.rerun()
        st.caption(f"R vs X points shown (initial frame): {rx_points_count} | Frequency steps: {rx_freq_steps}")
        st.caption("Point clicks toggle selection. Use Show list selection to commit.")
        selected_display_all = sorted(
            {
                display_case_name(str(v))
                for v in st.session_state.get(rx_state_key, [])
                if display_case_name(str(v)) in all_base_set
            }
        )
        st.session_state[rx_state_key] = list(selected_display_all)
        selected_display_visible = [v for v in selected_display_all if v in filtered_base_set]
        hidden_selected_count = max(0, len(selected_display_all) - len(selected_display_visible))
        if hidden_selected_count > 0:
            st.caption(f"Selected but hidden by filters: {hidden_selected_count}")
        _render_rx_selected_cases_table(selected_display_visible, list(filtered_cases), case_colors_scatter)


if __name__ == "__main__":
    main()
