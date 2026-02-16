from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REQUIRED_SHEETS: Tuple[str, str, str, str] = ("R1", "X1", "R0", "X0")
ENERGINET_DEFAULT_THRESHOLDS_BY_F1: Dict[int, Dict[int, float]] = {
    50: {2: 400.0, 3: 600.0, 4: 2400.0},
    60: {2: 450.0, 3: 800.0, 4: 3000.0},
}
# Backward-compatible alias for existing imports (50 Hz defaults).
ENERGINET_DEFAULT_THRESHOLDS: Dict[int, float] = dict(ENERGINET_DEFAULT_THRESHOLDS_BY_F1[50])
BAND_HALF_WIDTH_FACTOR = 0.2

LIMITATION_NOTE = (
    "Low-order resonance / TOV screening only (f1-relative up to the available sweep range). "
    "Not defensible for kHz switching-overvoltage ranking from 6th-harmonic-limited sweeps."
)


def default_energinet_thresholds_for_f1(f1_hz: float) -> Dict[int, float]:
    key = int(round(float(f1_hz)))
    if key not in ENERGINET_DEFAULT_THRESHOLDS_BY_F1:
        key = 50
    return dict(ENERGINET_DEFAULT_THRESHOLDS_BY_F1[key])


def _sheet_case_columns(df: pd.DataFrame) -> List[str]:
    return [str(c) for c in df.columns if str(c) != "Frequency (Hz)"]


def _frequency_vector(df: pd.DataFrame, sheet_name: str) -> np.ndarray:
    if "Frequency (Hz)" not in df.columns:
        raise ValueError(f"Sheet '{sheet_name}' is missing 'Frequency (Hz)' column.")
    freq = pd.to_numeric(df["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float)
    if freq.size == 0:
        raise ValueError(f"Sheet '{sheet_name}' has empty frequency column.")
    if not np.all(np.isfinite(freq)):
        raise ValueError(f"Sheet '{sheet_name}' has non-numeric frequency values.")
    return freq


def _validate_input_tables(data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    missing = [s for s in REQUIRED_SHEETS if s not in data or data[s] is None]
    if missing:
        raise ValueError(f"Missing required sheets: {', '.join(missing)}.")

    row_orders: Dict[str, np.ndarray] = {}
    ref_freq_sorted: Optional[np.ndarray] = None
    ref_cases: Optional[List[str]] = None

    for name in REQUIRED_SHEETS:
        df = data[name]
        freq = _frequency_vector(df, name)
        order = np.argsort(freq, kind="mergesort")
        freq_sorted = freq[order]
        if np.unique(freq_sorted).size != freq_sorted.size:
            raise ValueError(f"Sheet '{name}' has duplicated frequency values; deterministic band selection requires unique frequencies.")
        cases = _sheet_case_columns(df)
        if not cases:
            raise ValueError("No case columns found in required sheets.")

        row_orders[name] = order
        if ref_freq_sorted is None:
            ref_freq_sorted = freq_sorted
            ref_cases = list(cases)
            continue
        if freq_sorted.shape != ref_freq_sorted.shape or not np.allclose(freq_sorted, ref_freq_sorted, rtol=0.0, atol=1e-9):
            raise ValueError("Frequency column mismatch across required sheets (after sorting by frequency).")
        if list(cases) != list(ref_cases):
            raise ValueError("Case columns mismatch across required sheets.")

    if ref_freq_sorted is None or ref_cases is None:
        raise ValueError("Unable to validate required sheets.")
    return ref_freq_sorted, list(ref_cases), row_orders


def _extract_case_arrays(
    df: pd.DataFrame,
    cases: Sequence[str],
    sheet_name: str,
    row_order: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for case in cases:
        if case not in df.columns:
            raise ValueError(f"Case '{case}' not found in sheet '{sheet_name}'.")
        arr = pd.to_numeric(df[case], errors="coerce").to_numpy(dtype=float)
        if row_order is not None:
            if arr.shape[0] != row_order.shape[0]:
                raise ValueError(f"Row-count mismatch while sorting case '{case}' in sheet '{sheet_name}'.")
            arr = arr[row_order]
        out[str(case)] = arr
    return out


def _band_indices(freq: np.ndarray, center_hz: float, half_width_hz: float) -> np.ndarray:
    lo = float(center_hz) - float(half_width_hz)
    hi = float(center_hz) + float(half_width_hz)
    return np.where((freq >= lo) & (freq <= hi))[0]


def _max_mag_index_in_band(
    r_arr: np.ndarray,
    x_arr: np.ndarray,
    freq: np.ndarray,
    band_idx: np.ndarray,
) -> Optional[int]:
    if band_idx.size == 0:
        return None
    r = r_arr[band_idx]
    x = x_arr[band_idx]
    mags = np.sqrt(np.square(r) + np.square(x))
    valid = np.isfinite(mags)
    if not np.any(valid):
        return None
    valid_local = np.where(valid)[0]
    valid_mags = mags[valid_local]
    max_mag = float(np.nanmax(valid_mags))
    tol = 1e-12 * max(1.0, abs(max_mag))
    max_local = valid_local[np.where(np.isclose(valid_mags, max_mag, rtol=0.0, atol=tol))[0]]
    if max_local.size == 0:
        max_local = valid_local[np.where(valid_mags == max_mag)[0]]
    if max_local.size == 0:
        return None
    global_candidates = band_idx[max_local]
    if global_candidates.size == 1:
        return int(global_candidates[0])
    f_candidates = freq[global_candidates]
    return int(global_candidates[int(np.argmin(f_candidates))])


def _cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(pts) <= 1:
        return pts

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _point_on_segment(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
    tol: float,
) -> bool:
    if abs(_cross(a, b, p)) > float(tol):
        return False
    min_x = min(a[0], b[0]) - float(tol)
    max_x = max(a[0], b[0]) + float(tol)
    min_y = min(a[1], b[1]) - float(tol)
    max_y = max(a[1], b[1]) + float(tol)
    return (min_x <= p[0] <= max_x) and (min_y <= p[1] <= max_y)


def _compute_energinet_metrics(
    freq: np.ndarray,
    r_map: Dict[str, np.ndarray],
    x_map: Dict[str, np.ndarray],
    cases: Sequence[str],
    f1_hz: float,
) -> Dict[str, object]:
    band_half = BAND_HALF_WIDTH_FACTOR * float(f1_hz)
    band_indices = {n: _band_indices(freq, float(n) * float(f1_hz), band_half) for n in (2, 3, 4)}

    metrics_by_case: Dict[str, Dict[str, Optional[float]]] = {}
    for case in cases:
        case_id = str(case)
        r_arr = r_map[case_id]
        x_arr = x_map[case_id]
        row: Dict[str, Optional[float]] = {}
        for n in (2, 3, 4):
            idx_star = _max_mag_index_in_band(r_arr, x_arr, freq, band_indices[n])
            if idx_star is None:
                row[f"zmax_band_{n}"] = None
                row[f"f_at_zmax_band_{n}"] = None
            else:
                zmax = float(np.sqrt(float(r_arr[idx_star]) ** 2 + float(x_arr[idx_star]) ** 2))
                row[f"zmax_band_{n}"] = zmax if np.isfinite(zmax) else None
                row[f"f_at_zmax_band_{n}"] = float(freq[idx_star])
        metrics_by_case[case_id] = row

    return {
        "energinet_metrics": metrics_by_case,
        "band_sample_counts": {str(n): int(band_indices[n].size) for n in (2, 3, 4)},
    }


def _compute_iec_vertices(
    freq: np.ndarray,
    r_map: Dict[str, np.ndarray],
    x_map: Dict[str, np.ndarray],
    cases: Sequence[str],
    f1_hz: float,
    include_collinear_boundary: bool = False,
) -> Dict[str, object]:
    freq_max = float(np.nanmax(freq))
    n_env = int(min(6, np.floor(freq_max / float(f1_hz))))
    if n_env < 2:
        return {
            "iec_case_ids": [],
            "iec_first_harmonic": {},
            "iec_vertex_orders": {},
            "n_env": int(max(0, n_env)),
        }

    band_half = BAND_HALF_WIDTH_FACTOR * float(f1_hz)
    vertex_orders: Dict[str, List[int]] = {str(c): [] for c in cases}

    for n in range(2, n_env + 1):
        b_idx = _band_indices(freq, float(n) * float(f1_hz), band_half)
        if b_idx.size == 0:
            continue

        case_points: Dict[str, Tuple[float, float]] = {}
        point_to_cases: Dict[Tuple[float, float], List[str]] = {}
        for case in cases:
            case_id = str(case)
            idx_star = _max_mag_index_in_band(r_map[case_id], x_map[case_id], freq, b_idx)
            if idx_star is None:
                continue
            r_val = float(r_map[case_id][idx_star])
            x_val = float(x_map[case_id][idx_star])
            if not np.isfinite(r_val) or not np.isfinite(x_val):
                continue
            p = (r_val, x_val)
            case_points[case_id] = p
            point_to_cases.setdefault(p, []).append(case_id)

        if not case_points:
            continue

        unique_points = sorted(point_to_cases.keys(), key=lambda p: (p[0], p[1]))
        selected_points: List[Tuple[float, float]] = []
        if len(unique_points) < 3:
            selected_points = list(unique_points)
        else:
            hull_pts = _convex_hull(unique_points)
            if not include_collinear_boundary:
                selected_points = list(hull_pts)
            else:
                if len(hull_pts) <= 2:
                    selected_points = list(unique_points)
                else:
                    max_abs = max(max(abs(px), abs(py)) for (px, py) in unique_points)
                    tol = 1e-10 * max(1.0, float(max_abs))
                    for p in unique_points:
                        for i in range(len(hull_pts)):
                            a = hull_pts[i]
                            b = hull_pts[(i + 1) % len(hull_pts)]
                            if _point_on_segment(p, a, b, tol):
                                selected_points.append(p)
                                break

        selected: List[str] = []
        for sp in selected_points:
            selected.extend(point_to_cases.get(sp, []))
        selected_cases = sorted(set(selected))

        for case_id in selected_cases:
            vertex_orders.setdefault(case_id, []).append(int(n))

    vertex_orders_clean = {
        str(case_id): sorted(set(int(v) for v in h_list))
        for case_id, h_list in vertex_orders.items()
        if h_list
    }
    first_harmonic = {
        str(case_id): int(min(h_list))
        for case_id, h_list in vertex_orders_clean.items()
    }
    selected_case_ids = sorted(first_harmonic.keys())
    return {
        "iec_case_ids": selected_case_ids,
        "iec_first_harmonic": first_harmonic,
        "iec_vertex_orders": vertex_orders_clean,
        "n_env": int(n_env),
    }


def build_preselection_payload(
    data: Dict[str, pd.DataFrame],
    cases: Sequence[str],
    fundamentals_hz: Sequence[float] = (50.0, 60.0),
) -> Dict[str, object]:
    freq, all_case_ids, row_orders = _validate_input_tables(data)
    all_case_set = set(str(c) for c in all_case_ids)

    chosen_cases: List[str] = []
    seen: set[str] = set()
    for case in cases:
        case_id = str(case)
        if case_id in all_case_set and case_id not in seen:
            chosen_cases.append(case_id)
            seen.add(case_id)
    if not chosen_cases:
        raise ValueError("No selected-location cases are available in validated sheets.")

    r1_map = _extract_case_arrays(data["R1"], chosen_cases, "R1", row_order=row_orders.get("R1"))
    x1_map = _extract_case_arrays(data["X1"], chosen_cases, "X1", row_order=row_orders.get("X1"))

    by_f1: Dict[str, Dict[str, object]] = {}
    for f1 in fundamentals_hz:
        f1_val = float(f1)
        if f1_val not in (50.0, 60.0):
            raise ValueError(f"Unsupported fundamental frequency in configuration: {f1_val}.")
        f1_key = str(int(round(f1_val)))
        energinet = _compute_energinet_metrics(freq, r1_map, x1_map, chosen_cases, f1_val)
        iec_vertices = _compute_iec_vertices(
            freq,
            r1_map,
            x1_map,
            chosen_cases,
            f1_val,
            include_collinear_boundary=False,
        )
        iec_boundary = _compute_iec_vertices(
            freq,
            r1_map,
            x1_map,
            chosen_cases,
            f1_val,
            include_collinear_boundary=True,
        )
        by_f1[f1_key] = {
            "energinet_metrics": dict(energinet["energinet_metrics"]),
            "band_sample_counts": dict(energinet["band_sample_counts"]),
            "iec_modes": {
                "vertices": {
                    "iec_case_ids": list(iec_vertices["iec_case_ids"]),
                    "iec_first_harmonic": dict(iec_vertices["iec_first_harmonic"]),
                    "iec_vertex_orders": dict(iec_vertices["iec_vertex_orders"]),
                    "n_env": int(iec_vertices["n_env"]),
                },
                "boundary": {
                    "iec_case_ids": list(iec_boundary["iec_case_ids"]),
                    "iec_first_harmonic": dict(iec_boundary["iec_first_harmonic"]),
                    "iec_vertex_orders": dict(iec_boundary["iec_vertex_orders"]),
                    "n_env": int(iec_boundary["n_env"]),
                },
            },
            # Compatibility alias to preserve existing consumers.
            "iec_case_ids": list(iec_vertices["iec_case_ids"]),
            "iec_first_harmonic": dict(iec_vertices["iec_first_harmonic"]),
            "iec_vertex_orders": dict(iec_vertices["iec_vertex_orders"]),
            "n_env": int(iec_vertices["n_env"]),
        }

    return {
        "available": True,
        "error": "",
        "limitation_note": str(LIMITATION_NOTE),
        "cases_count": int(len(chosen_cases)),
        "by_f1": by_f1,
    }


def build_preselection_payload_safe(
    data: Dict[str, pd.DataFrame],
    cases: Sequence[str],
    fundamentals_hz: Sequence[float] = (50.0, 60.0),
) -> Dict[str, object]:
    try:
        return build_preselection_payload(data, cases, fundamentals_hz=fundamentals_hz)
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "limitation_note": str(LIMITATION_NOTE),
            "cases_count": int(len(cases)),
            "by_f1": {},
        }
