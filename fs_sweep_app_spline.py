import io
import os
import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional, Union

# Main app baseline with JS-side interactive case controls.

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
from preselection_shortlist import (
    ENERGINET_DEFAULT_THRESHOLDS,
    build_preselection_payload_safe,
    default_energinet_thresholds_for_f1,
)

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
SHEET_VALUE_DTYPE = np.float32  # compact in-memory numeric representation for large uploads

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
    "uploaded_data:",
    "preselection_payload:",
    "line_fig_sig:",
    "line_fig_cache:",
    "line_fig_meta:",
    "rx_filter_sig:",
    "rx_fig_sig:",
    "rx_fig_cache:",
    "rx_fig_steps:",
    "selection_bind_nonce:",
    "chart_context_tracker:",
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


@dataclass
class SweepSheet:
    frequency_hz: np.ndarray
    case_ids: Tuple[str, ...]
    values: np.ndarray
    _prepared: Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]] = field(default=None, init=False, repr=False)

    def prepared_arrays(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if self._prepared is None:
            fmap: Dict[str, np.ndarray] = {}
            n_cases = int(self.values.shape[1]) if self.values.ndim == 2 else 0
            for i, cid in enumerate(self.case_ids):
                if i >= n_cases:
                    break
                fmap[str(cid)] = self.values[:, i]
            self._prepared = (self.frequency_hz, fmap)
        return self._prepared


SheetLike = Union[pd.DataFrame, SweepSheet]


def plotly_selection_bridge(
    data_id: str,
    chart_id: str,
    plot_ids: List[str],
    cases_meta: List[Dict[str, object]],
    part_labels: List[str],
    color_by_options: List[str],
    color_maps: Dict[str, Dict[str, str]],
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
    preselection_payload: Optional[Dict[str, object]] = None,
    energinet_t2_default: float = float(ENERGINET_DEFAULT_THRESHOLDS[2]),
    energinet_t3_default: float = float(ENERGINET_DEFAULT_THRESHOLDS[3]),
    energinet_t4_default: float = float(ENERGINET_DEFAULT_THRESHOLDS[4]),
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
        preselection_payload=dict(preselection_payload or {}),
        energinet_t2_default=float(energinet_t2_default),
        energinet_t3_default=float(energinet_t3_default),
        energinet_t4_default=float(energinet_t4_default),
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
        return
    try:
        st.session_state["uploaded_file_sha1_10"] = hashlib.sha1(up.getvalue()).hexdigest()[: int(UPLOAD_SHA1_PREFIX_LEN)]
    except Exception:
        st.session_state.pop("uploaded_file_sha1_10", None)


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


def _preselection_cache_key(data_id: str, seq_label: str, location_value: str) -> str:
    return f"preselection_payload:{data_id}:{seq_label}:{location_value}"


def _uploaded_workbook_cache_key(data_id: str) -> str:
    return f"uploaded_data:{data_id}:npz"


def _selection_bind_nonce_key(data_id: str, chart_id: str) -> str:
    return f"selection_bind_nonce:{data_id}:{chart_id}"


def _chart_context_tracker_key(data_id: str) -> str:
    return f"chart_context_tracker:{data_id}"


def _to_finite_float_or_none(raw: object) -> Optional[float]:
    try:
        v = float(raw)
    except Exception:
        return None
    return float(v) if np.isfinite(v) else None


def _to_nonnegative_int(raw: object) -> int:
    try:
        v = int(round(float(raw)))
    except Exception:
        return 0
    return max(0, int(v))


def _compact_iec_mode_payload(mode_payload: Dict[str, object], case_index: Dict[str, int]) -> Dict[str, object]:
    vertex_orders_raw = mode_payload.get("iec_vertex_orders")
    vertex_orders = vertex_orders_raw if isinstance(vertex_orders_raw, dict) else {}
    ids_raw = mode_payload.get("iec_case_ids")
    if isinstance(ids_raw, list):
        ids = [str(v) for v in ids_raw if str(v) != ""]
    else:
        ids = sorted([str(k) for k in vertex_orders.keys() if str(k) != ""])

    case_idx: List[int] = []
    orders_out: List[List[int]] = []
    seen_case_ids = set()
    for cid in ids:
        if cid in seen_case_ids:
            continue
        seen_case_ids.add(cid)
        idx = case_index.get(str(cid))
        if idx is None:
            continue
        ord_src = vertex_orders.get(str(cid))
        ord_list = ord_src if isinstance(ord_src, list) else []
        ord_clean: List[int] = []
        seen_orders = set()
        for hv_raw in ord_list:
            hv = _to_nonnegative_int(hv_raw)
            if hv < 1 or hv in seen_orders:
                continue
            seen_orders.add(hv)
            ord_clean.append(int(hv))
        case_idx.append(int(idx))
        orders_out.append(ord_clean)

    return {
        "case_idx": case_idx,
        "vertex_orders": orders_out,
        "n_env": int(_to_nonnegative_int(mode_payload.get("n_env", 0))),
    }


def _compact_preselection_payload(payload: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(payload, dict):
        return {
            "available": False,
            "error": "Invalid preselection payload shape.",
            "limitation_note": "",
            "cases_count": 0,
            "format": "compact_v1",
            "by_f1": {},
        }

    if str(payload.get("format", "")) == "compact_v1":
        return dict(payload)

    out: Dict[str, object] = {
        "available": bool(payload.get("available", False)),
        "error": str(payload.get("error", "")),
        "limitation_note": str(payload.get("limitation_note", "")),
        "cases_count": int(_to_nonnegative_int(payload.get("cases_count", 0))),
        "format": "compact_v1",
        "by_f1": {},
    }

    by_f1_raw = payload.get("by_f1")
    by_f1_out: Dict[str, Dict[str, object]] = {}
    if isinstance(by_f1_raw, dict):
        for f1_key, base_node_raw in by_f1_raw.items():
            if not isinstance(base_node_raw, dict):
                continue

            metrics_raw = base_node_raw.get("energinet_metrics")
            metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
            case_ids = sorted([str(cid) for cid in metrics.keys() if str(cid) != ""])

            iec_modes_raw = base_node_raw.get("iec_modes")
            iec_modes = iec_modes_raw if isinstance(iec_modes_raw, dict) else {}
            vertices_src = iec_modes.get("vertices")
            boundary_src = iec_modes.get("boundary")
            vertices_node = vertices_src if isinstance(vertices_src, dict) else base_node_raw
            boundary_node = boundary_src if isinstance(boundary_src, dict) else base_node_raw

            for mode_node in (vertices_node, boundary_node):
                mode_ids = mode_node.get("iec_case_ids") if isinstance(mode_node, dict) else None
                if isinstance(mode_ids, list):
                    for cid in mode_ids:
                        c = str(cid)
                        if c and c not in case_ids:
                            case_ids.append(c)
            case_ids = sorted(case_ids)
            case_index = {cid: i for i, cid in enumerate(case_ids)}

            z2: List[Optional[float]] = []
            z3: List[Optional[float]] = []
            z4: List[Optional[float]] = []
            f2: List[Optional[float]] = []
            f3: List[Optional[float]] = []
            f4: List[Optional[float]] = []
            for cid in case_ids:
                row_raw = metrics.get(cid)
                row = row_raw if isinstance(row_raw, dict) else {}
                z2.append(_to_finite_float_or_none(row.get("zmax_band_2")))
                z3.append(_to_finite_float_or_none(row.get("zmax_band_3")))
                z4.append(_to_finite_float_or_none(row.get("zmax_band_4")))
                f2.append(_to_finite_float_or_none(row.get("f_at_zmax_band_2")))
                f3.append(_to_finite_float_or_none(row.get("f_at_zmax_band_3")))
                f4.append(_to_finite_float_or_none(row.get("f_at_zmax_band_4")))

            band_counts_raw = base_node_raw.get("band_sample_counts")
            band_counts_in = band_counts_raw if isinstance(band_counts_raw, dict) else {}
            band_counts = {
                str(k): int(_to_nonnegative_int(v))
                for k, v in band_counts_in.items()
            }

            by_f1_out[str(f1_key)] = {
                "format": "compact_v1",
                "case_ids": list(case_ids),
                "energinet": {
                    "z2": z2,
                    "z3": z3,
                    "z4": z4,
                    "f2": f2,
                    "f3": f3,
                    "f4": f4,
                },
                "band_sample_counts": band_counts,
                "iec_modes": {
                    "vertices": _compact_iec_mode_payload(vertices_node, case_index),
                    "boundary": _compact_iec_mode_payload(boundary_node, case_index),
                },
            }
    out["by_f1"] = by_f1_out
    return out


def _evict_sequence_render_caches(data_id: str, seq_label: str) -> None:
    did = str(data_id or "")
    seq = str(seq_label or "")
    if not did or not seq:
        return
    seq_prefixes = (
        "line_fig_sig:",
        "line_fig_cache:",
        "line_fig_meta:",
        "rx_filter_sig:",
        "rx_fig_sig:",
        "rx_fig_cache:",
        "rx_fig_steps:",
    )
    for key in list(st.session_state.keys()):
        k = str(key)
        for prefix in seq_prefixes:
            if k.startswith(f"{prefix}{did}:{seq}:"):
                st.session_state.pop(k, None)
                break


def _prune_chart_scoped_session_state(
    data_id: str,
    seq_label: str,
    selected_location: str,
    chart_id: str,
) -> None:
    did = str(data_id or "")
    seq = str(seq_label or "")
    loc = str(selected_location or "")
    cid = str(chart_id or "")
    if not did:
        return

    current_preselection_key = _preselection_cache_key(did, seq, loc)
    current_nonce_key = _selection_bind_nonce_key(did, cid)
    current_chart_context_key = _chart_context_tracker_key(did)
    current_location_select_key = f"location_select:{did}:{seq}"

    seq_prefixes = (
        "line_fig_sig:",
        "line_fig_cache:",
        "line_fig_meta:",
        "rx_filter_sig:",
        "rx_fig_sig:",
        "rx_fig_cache:",
        "rx_fig_steps:",
    )

    for key in list(st.session_state.keys()):
        k = str(key)
        if k.startswith(f"selection_bind_nonce:{did}:") and k != current_nonce_key:
            st.session_state.pop(k, None)
            continue
        if k.startswith(f"preselection_payload:{did}:") and k != current_preselection_key:
            st.session_state.pop(k, None)
            continue
        if k.startswith(f"chart_context_tracker:{did}") and k != current_chart_context_key:
            st.session_state.pop(k, None)
            continue
        if k.startswith(f"location_select:{did}:") and k != current_location_select_key:
            st.session_state.pop(k, None)
            continue

        for prefix in seq_prefixes:
            if not k.startswith(prefix):
                continue
            if not k.startswith(f"{prefix}{did}:{seq}:"):
                st.session_state.pop(k, None)
            break


def _maybe_evict_caches_on_chart_switch(
    data_id: str,
    seq_label: str,
    selected_location: str,
    chart_id: str,
) -> None:
    did = str(data_id or "")
    seq = str(seq_label or "")
    loc = str(selected_location or "")
    if not did or not seq:
        return

    tracker_key = _chart_context_tracker_key(did)
    prev_chart = str(st.session_state.get(tracker_key, "") or "")
    cur_chart = str(chart_id or "")
    if prev_chart and cur_chart and prev_chart != cur_chart:
        _evict_sequence_render_caches(did, seq)
        st.session_state.pop(_preselection_cache_key(did, seq, loc), None)
    st.session_state[tracker_key] = cur_chart


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
def _find_frequency_column(df: pd.DataFrame) -> Optional[object]:
    for c in df.columns:
        c_norm = str(c).strip().lower().replace(" ", "")
        if c_norm in ["frequency(hz)", "frequencyhz", "frequency_"]:
            return c
        if str(c).strip().lower() in ["frequency (hz)", "frequency"]:
            return c
    if "Frequency (Hz)" in df.columns:
        return "Frequency (Hz)"
    return None


def _sheet_from_dataframe(df: pd.DataFrame, sheet_name: str) -> SweepSheet:
    freq_col = _find_frequency_column(df)
    if freq_col is None:
        raise ValueError(f"Sheet '{sheet_name}' missing 'Frequency (Hz)' column")
    src = df.rename(columns={freq_col: "Frequency (Hz)"})
    freq_raw = pd.to_numeric(src["Frequency (Hz)"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    finite_mask = np.isfinite(freq_raw)
    freq = freq_raw[finite_mask]
    if freq.size == 0:
        raise ValueError(f"Sheet '{sheet_name}' has empty or invalid frequency column.")

    case_cols = [str(c) for c in src.columns if str(c) != "Frequency (Hz)"]
    n_rows = int(freq.shape[0])
    n_cases = int(len(case_cols))
    values = np.empty((n_rows, n_cases), dtype=SHEET_VALUE_DTYPE)
    for j, case_name in enumerate(case_cols):
        arr_raw = pd.to_numeric(src[case_name], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        arr = arr_raw[finite_mask]
        if arr.shape[0] != n_rows:
            raise ValueError(f"Sheet '{sheet_name}' case '{case_name}' row count mismatch after frequency filtering.")
        values[:, j] = arr.astype(SHEET_VALUE_DTYPE, copy=False)

    return SweepSheet(
        frequency_hz=freq.astype(np.float64, copy=False),
        case_ids=tuple(case_cols),
        values=values,
    )


def _load_fs_sweep_xlsx_impl(path_or_buf) -> Dict[str, SweepSheet]:
    out: Dict[str, SweepSheet] = {}
    xls = pd.ExcelFile(path_or_buf)
    for name in ["R1", "X1", "R0", "X0"]:
        if name not in xls.sheet_names:
            continue
        df = pd.read_excel(xls, sheet_name=name)
        out[name] = _sheet_from_dataframe(df, name)
    return out


def _serialize_workbook_npz(workbook: Dict[str, SweepSheet]) -> bytes:
    payload: Dict[str, np.ndarray] = {}
    for name, sheet in workbook.items():
        payload[f"{name}__freq"] = np.asarray(sheet.frequency_hz, dtype=np.float64)
        payload[f"{name}__cases"] = np.asarray(list(sheet.case_ids), dtype=np.str_)
        payload[f"{name}__values"] = np.asarray(sheet.values, dtype=SHEET_VALUE_DTYPE)
    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    return bytes(buf.getvalue())


def _deserialize_workbook_npz(raw_bytes: bytes) -> Dict[str, SweepSheet]:
    out: Dict[str, SweepSheet] = {}
    with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as z:
        keys = list(z.keys())
        sheet_names = sorted({str(k).split("__", 1)[0] for k in keys if "__" in str(k)})
        for name in sheet_names:
            k_freq = f"{name}__freq"
            k_cases = f"{name}__cases"
            k_values = f"{name}__values"
            if k_freq not in z or k_cases not in z or k_values not in z:
                continue
            freq = np.asarray(z[k_freq], dtype=np.float64)
            cases_arr = np.asarray(z[k_cases])
            case_ids = tuple(str(v) for v in cases_arr.tolist())
            values = np.asarray(z[k_values], dtype=SHEET_VALUE_DTYPE)
            if values.ndim != 2:
                raise ValueError(f"Invalid NPZ shape for sheet '{name}': values must be 2-D.")
            if values.shape[0] != freq.shape[0]:
                raise ValueError(f"Invalid NPZ data for sheet '{name}': frequency/values row mismatch.")
            if values.shape[1] != len(case_ids):
                raise ValueError(f"Invalid NPZ data for sheet '{name}': case/values column mismatch.")
            out[name] = SweepSheet(
                frequency_hz=freq,
                case_ids=case_ids,
                values=values,
            )
    return out


@st.cache_data(show_spinner=False)
def load_fs_sweep_xlsx_cached(path_or_buf) -> Dict[str, SweepSheet]:
    return _load_fs_sweep_xlsx_impl(path_or_buf)


def list_case_columns(df: Optional[SheetLike]) -> List[str]:
    if df is None:
        return []
    if isinstance(df, SweepSheet):
        return [str(c) for c in df.case_ids]
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


def sheet_frequency_values(sheet: Optional[SheetLike]) -> Optional[np.ndarray]:
    if sheet is None:
        return None
    if isinstance(sheet, SweepSheet):
        return np.asarray(sheet.frequency_hz, dtype=np.float64)
    if "Frequency (Hz)" not in sheet.columns:
        return None
    return pd.to_numeric(sheet["Frequency (Hz)"], errors="coerce").to_numpy(dtype=np.float64, copy=False)


def prepare_sheet_arrays(df: SheetLike) -> Tuple[np.ndarray, Dict[object, np.ndarray]]:
    if isinstance(df, SweepSheet):
        freq_hz, fmap = df.prepared_arrays()
        return np.asarray(freq_hz, dtype=np.float64), {str(k): np.asarray(v) for k, v in fmap.items()}

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


def compute_common_n_range(f_series: List[Optional[np.ndarray]], f_base: float) -> Tuple[float, float]:
    vals: List[float] = []
    for s in f_series:
        if s is None:
            continue
        v = np.asarray(s, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size > 0:
            vals.extend([float(np.min(v)) / f_base, float(np.max(v)) / f_base])
    if not vals:
        return 0.0, 1.0
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    return (0.0, 1.0) if (not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi) else (lo, hi)


def make_spline_traces(
    df: SheetLike,
    cases: List[str],
    f_base: float,
    y_title: str,
    smooth: float,
    enable_spline: bool,
    strip_location_suffix: bool,
    case_colors: Dict[str, str],
) -> Tuple[List[BaseTraceType], Optional[np.ndarray]]:
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
        mode = "lines"
        tr = TraceCls(
            x=n,
            y=y,
            customdata=cd,
            mode=mode,
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
                color=color,
            )
            tr.update(line=spline_line)
        traces.append(tr)
    return traces, np.asarray(cd, dtype=np.float64)


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


def build_plot_spline(df: Optional[SheetLike], cases: List[str], f_base: float, plot_height: int, y_title: str,
                      smooth: float, enable_spline: bool, legend_entrywidth: int, strip_location_suffix: bool,
                      use_auto_width: bool, figure_width_px: int, case_colors: Dict[str, str],
                      ) -> Tuple[go.Figure, Optional[np.ndarray]]:
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


def build_x_over_r_spline(df_r: Optional[SheetLike], df_x: Optional[SheetLike], cases: List[str], f_base: float,
                          plot_height: int, seq_label: str, smooth: float, legend_entrywidth: int,
                          enable_spline: bool,
                          strip_location_suffix: bool, use_auto_width: bool, figure_width_px: int,
                          case_colors: Dict[str, str],
                          ) -> Tuple[go.Figure, Optional[np.ndarray], int, int]:
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
        f_series = sheet_frequency_values(df_r)
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
            mode = "lines"
            tr = TraceCls(
                x=n,
                y=y,
                customdata=cd,
                mode=mode,
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
                    color=color,
                )
                tr.update(line=spline_line)
            traces.append(tr)
    fig = go.Figure(data=traces)
    y_title = "X1/R1 (unitless)" if seq_label == "Positive" else "X0/R0 (unitless)"
    apply_common_layout(fig, plot_height, y_title, legend_entrywidth, use_auto_width, figure_width_px)
    return fig, f_series, xr_dropped, xr_total


def build_rx_scatter_animated(
    df_r: Optional[SheetLike],
    df_x: Optional[SheetLike],
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
    point_case_ids: List[str] = [str(case) for case, _, _ in case_arrays]
    point_display_names: List[str] = [str(display_case_name(case)) for case in point_case_ids]
    point_colors: List[str] = [str(case_colors.get(case, "#1f77b4")) for case in point_case_ids]
    point_customdata: List[List[object]] = [
        [str(case_id), str(display_name)]
        for case_id, display_name in zip(point_case_ids, point_display_names)
    ]

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

    def frame_data_for_freq_idx(fi: int, include_style: bool = True) -> Tuple[dict, int]:
        idx_r = int(idx_r_for_freq[int(fi)])
        idx_x = int(idx_x_for_freq[int(fi)])

        xs: List[float] = []
        ys: List[float] = []

        finite_count = 0
        for _case, r_arr, x_arr in case_arrays:
            if idx_r >= int(r_arr.size) or idx_x >= int(x_arr.size):
                xs.append(None)
                ys.append(None)
                continue
            r_v = r_arr[idx_r]
            x_v = x_arr[idx_x]
            if np.isfinite(r_v) and np.isfinite(x_v):
                xs.append(float(r_v))
                ys.append(float(x_v))
                finite_count += 1
            else:
                # Keep case index stable across frames.
                # Invisible points are represented by null coordinates.
                xs.append(None)
                ys.append(None)

        if include_style:
            trace = dict(
                type="scatter",
                x=xs,
                y=ys,
                mode="markers",
                name="Cases",
                customdata=point_customdata,
                ids=point_case_ids,
                hovertemplate="Case=%{customdata[1]}<br>R=%{x}<br>X=%{y}<extra></extra>",
                marker=dict(
                    color=point_colors,
                    size=float(SELECTED_MARKER_SIZE),
                    opacity=1.0,
                    line=dict(width=0),
                ),
                showlegend=False,
                meta={"kind": "points"},
            )
        else:
            # Keep animation frame payload lean: marker style is inherited from active trace
            # and is updated client-side without rewriting all frame traces.
            trace = dict(
                type="scatter",
                x=xs,
                y=ys,
                ids=point_case_ids,
            )
        return trace, int(finite_count)

    x_by_step: List[List[Optional[float]]] = []
    y_by_step: List[List[Optional[float]]] = []
    for i in range(len(freq_candidates)):
        tr_i, _ = frame_data_for_freq_idx(i, include_style=False)
        x_by_step.append(list(tr_i.get("x", [])))
        y_by_step.append(list(tr_i.get("y", [])))

    f0 = float(freq_candidates[init_idx])
    tr0, _ = frame_data_for_freq_idx(init_idx)
    fig.add_trace(go.Scatter(**tr0))

    slider_steps = [
        dict(
            method="skip",
            args=[int(i)],
            label=f"{float(f_sel):.1f}",
        )
        for i, f_sel in enumerate(freq_candidates)
    ]

    shapes: List[dict] = [
        dict(type="line", xref="x", yref="paper", x0=0, x1=0, y0=0, y1=1, line=dict(color="rgba(0,0,0,0.45)", width=1)),
        dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=0, line=dict(color="rgba(0,0,0,0.45)", width=1)),
    ]
    fig.update_layout(
        title=f"R vs X at f ~ {f0:.1f} Hz ({seq_label})",
        xaxis_title=f"R{1 if seq_label == 'Positive' else 0} (Ohm)",
        yaxis_title=f"X{1 if seq_label == 'Positive' else 0} (Ohm)",
        meta={
            "rx_single_trace": {
                "enabled": True,
                "freq_hz": [float(v) for v in freq_candidates],
                "x_by_step": x_by_step,
                "y_by_step": y_by_step,
                "seq_label": str(seq_label),
            }
        },
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
    f_ref: Optional[np.ndarray],
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
        height=150,
        default=0,
    )


def _load_data_source(default_path: str) -> Tuple[Dict[str, SweepSheet], str]:
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
        raw_bytes: Optional[bytes] = None
        try:
            cached = st.session_state.get("uploaded_file_sha1_10")
            if cached:
                data_id = str(cached)
            else:
                raw_bytes = up.getvalue()
                data_id = hashlib.sha1(raw_bytes).hexdigest()[:10]
        except Exception:
            data_id = f"upload:{getattr(up, 'name', 'file')}"

        upload_cache_key = _uploaded_workbook_cache_key(str(data_id))
        cached_npz = st.session_state.get(upload_cache_key)
        if isinstance(cached_npz, (bytes, bytearray)) and len(cached_npz) > 0:
            try:
                return _deserialize_workbook_npz(bytes(cached_npz)), data_id
            except Exception:
                st.session_state.pop(upload_cache_key, None)

        if raw_bytes is None:
            try:
                raw_bytes = up.getvalue()
            except Exception:
                raw_bytes = None
        if raw_bytes is not None:
            data = _load_fs_sweep_xlsx_impl(io.BytesIO(raw_bytes))
        else:
            data = _load_fs_sweep_xlsx_impl(up)
        try:
            st.session_state[upload_cache_key] = _serialize_workbook_npz(data)
        except Exception:
            st.session_state.pop(upload_cache_key, None)
        return data, data_id

    if os.path.exists(default_path):
        data = load_fs_sweep_xlsx_cached(default_path)
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

    base_freq_key = "base_frequency_hz_control"
    base_freq_val = int(st.session_state.get(base_freq_key, 50))
    if base_freq_val not in (50, 60):
        base_freq_val = 50
        st.session_state[base_freq_key] = 50
    f_base = float(base_freq_val)
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
        "base_freq_key": str(base_freq_key),
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


def _line_fig_cache_keys(data_id: str, seq_label: str, kind: str) -> Tuple[str, str, str]:
    base = f"{data_id}:{seq_label}:{kind}"
    return (
        f"line_fig_sig:{base}",
        f"line_fig_cache:{base}",
        f"line_fig_meta:{base}",
    )


def _get_or_build_cached_line_figure(
    data_id: str,
    seq_label: str,
    kind: str,
    sig_payload: Dict[str, object],
    builder: Callable[[], Tuple[go.Figure, Dict[str, object]]],
) -> Tuple[go.Figure, Dict[str, object]]:
    sig_key, fig_key, meta_key = _line_fig_cache_keys(data_id, seq_label, kind)
    sig = hashlib.sha1(
        json.dumps(sig_payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]

    if st.session_state.get(sig_key) != sig or (fig_key not in st.session_state):
        fig_built, meta_built = builder()
        st.session_state[sig_key] = sig
        st.session_state[fig_key] = fig_built.to_dict()
        st.session_state[meta_key] = dict(meta_built or {})

    fig = go.Figure(st.session_state.get(fig_key, {}))
    raw_meta = st.session_state.get(meta_key, {})
    meta = raw_meta if isinstance(raw_meta, dict) else {}
    return fig, meta


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
    base_freq_key = str(controls["base_freq_key"])
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
    st.sidebar.radio("Base frequency", [50, 60], key=base_freq_key, format_func=lambda v: f"{int(v)} Hz")
    f_base = float(int(st.session_state.get(base_freq_key, int(round(f_base)))))
    selected_location_label = st.sidebar.radio("Location", options=location_labels, key=location_key)
    selected_location = str(location_label_to_value.get(str(selected_location_label), ""))
    chart_id = f"{seq_label}:{selected_location}:f{int(round(f_base))}"
    _prune_chart_scoped_session_state(
        data_id=str(data_id),
        seq_label=str(seq_label),
        selected_location=str(selected_location),
        chart_id=str(chart_id),
    )
    _maybe_evict_caches_on_chart_switch(
        data_id=str(data_id),
        seq_label=str(seq_label),
        selected_location=str(selected_location),
        chart_id=str(chart_id),
    )
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

    preselection_cache_key = _preselection_cache_key(str(data_id), str(seq_label), str(selected_location))
    requested_base_key = str(int(round(float(f_base))))
    raw_preselection_payload = st.session_state.get(preselection_cache_key)
    payload_by_f1 = raw_preselection_payload.get("by_f1") if isinstance(raw_preselection_payload, dict) else {}
    cache_has_requested_base = isinstance(payload_by_f1, dict) and requested_base_key in payload_by_f1
    if not isinstance(raw_preselection_payload, dict) or not cache_has_requested_base:
        built_payload = build_preselection_payload_safe(
            data=data,
            cases=list(location_cases),
            fundamentals_hz=(float(f_base),),
        )
        raw_preselection_payload = _compact_preselection_payload(built_payload)
        st.session_state[preselection_cache_key] = dict(raw_preselection_payload)
    preselection_payload = dict(raw_preselection_payload)

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

        def _build_x_cached() -> Tuple[go.Figure, Dict[str, object]]:
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
            return fig_x_built, {}

        fig_x, _ = _get_or_build_cached_line_figure(
            data_id=data_id,
            seq_label=seq_label,
            kind="x",
            sig_payload=x_sig_payload,
            builder=_build_x_cached,
        )
        f_x = sheet_frequency_values(df_x)
        plot_items.append(_make_plot_item("x", fig_x, f_x, "X_full_legend.png", "Export", "plot_x"))

    if show_plot_r:
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

        def _build_r_cached() -> Tuple[go.Figure, Dict[str, object]]:
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
            return fig_r_built, {}

        fig_r, _ = _get_or_build_cached_line_figure(
            data_id=data_id,
            seq_label=seq_label,
            kind="r",
            sig_payload=r_sig_payload,
            builder=_build_r_cached,
        )
        f_r = sheet_frequency_values(df_r)
        plot_items.append(_make_plot_item("r", fig_r, f_r, "R_full_legend.png", "Export", "plot_r"))

    if show_plot_xr:
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

        def _build_xr_cached() -> Tuple[go.Figure, Dict[str, object]]:
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
            return fig_xr_built, {
                "xr_dropped": int(xr_dropped_built),
                "xr_total": int(xr_total_built),
            }

        fig_xr, xr_meta = _get_or_build_cached_line_figure(
            data_id=data_id,
            seq_label=seq_label,
            kind="xr",
            sig_payload=xr_sig_payload,
            builder=_build_xr_cached,
        )
        xr_dropped = int(xr_meta.get("xr_dropped", 0))
        xr_total = int(xr_meta.get("xr_total", 0))
        f_xr = sheet_frequency_values(df_r)
        plot_items.append(_make_plot_item("xr", fig_xr, f_xr, "X_over_R_full_legend.png", "Export", "plot_xr"))

    f_refs = [it["f_ref"] for it in plot_items if it.get("f_ref") is not None]
    n_lo, n_hi = compute_common_n_range(f_refs, f_base)
    for it in plot_items:
        fig = it["fig"]
        if isinstance(fig, go.Figure):
            fig.update_xaxes(range=[n_lo, n_hi])

    # Render
    location_caption = selected_location if selected_location else "<empty>"
    st.subheader(f"Sequence: {seq_label} | Base frequency: {int(round(f_base))} Hz | Location: {location_caption}")
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
            rx_freq_steps = int(st.session_state.get(rx_fig_steps_key, 0))
            rx_freq_steps_for_bridge = int(rx_freq_steps)
            rx_status_dom_id = f"rx-status-{hashlib.sha1(f'{data_id}:{chart_id}'.encode('utf-8')).hexdigest()[:10]}"

            st.plotly_chart(rx_fig, use_container_width=bool(use_auto_width), config=download_config, key="plot_rx")
            rx_plot_index = int(plot_order.index("rx")) if "rx" in plot_order else 0
            _render_rx_client_step_buttons(rx_plot_index, data_id=data_id, chart_id=chart_id)
            st.markdown(
                (
                    f"<div id=\"{rx_status_dom_id}\" style=\"font-size:0.92rem; color:#666; margin:2px 0 2px 0;\">"
                    f"R vs X points shown: {len(location_cases)} | Frequency steps: {int(rx_freq_steps)}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            st.caption("Point clicks toggle selection. Case-part/color/selection controls are in the sidebar.")

    sel_bind_nonce_key = _selection_bind_nonce_key(str(data_id), str(chart_id))
    st.session_state[sel_bind_nonce_key] = int(st.session_state.get(sel_bind_nonce_key, 0)) + 1

    with interactive_controls_area:
        energinet_defaults = default_energinet_thresholds_for_f1(float(f_base))
        plotly_selection_bridge(
            data_id=data_id,
            chart_id=chart_id,
            plot_ids=list(plot_order),
            cases_meta=list(cases_meta),
            part_labels=list(part_labels),
            color_by_options=list(color_by_options),
            color_maps=dict(color_maps),
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
            preselection_payload=dict(preselection_payload),
            energinet_t2_default=float(energinet_defaults[2]),
            energinet_t3_default=float(energinet_defaults[3]),
            energinet_t4_default=float(energinet_defaults[4]),
            reset_token=int(upload_nonce),
            selection_reset_token=int(selection_reset_token),
            render_nonce=int(st.session_state.get(sel_bind_nonce_key, 0)),
            enable_selection=bool(show_plot_rx),
            spline_enabled=bool(enable_spline),
        )


if __name__ == "__main__":
    main()
