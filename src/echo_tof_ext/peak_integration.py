"""EIC 추출 + 피크 검출 + 적분.

pyOpenMS 기반 mzML 데이터 처리.
원본: LC 분석/20260306_0443/scripts/peak_integration.py
"""
import numpy as np
from typing import List, Optional, Tuple
from .config import logger
from .baseline_correction import correct_baseline


def extract_eic(
    exp,
    target_mz: float,
    mz_tolerance: float = 0.01,
    rt_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """MSExperiment에서 특정 m/z의 EIC를 추출."""
    rts, intensities = [], []
    for spec in exp:
        if spec.getMSLevel() != 1:
            continue
        rt = spec.getRT() / 60.0
        if rt_range and (rt < rt_range[0] or rt > rt_range[1]):
            continue
        mzs, ints = spec.get_peaks()
        mask = np.abs(mzs - target_mz) <= mz_tolerance
        rts.append(rt)
        intensities.append(float(ints[mask].sum()) if mask.any() else 0.0)
    return np.array(rts), np.array(intensities)


def detect_peaks(
    rts: np.ndarray,
    intensities: np.ndarray,
    min_sn: float = 3.0,
    baseline_method: Optional[str] = "snip",
    min_peak_width_min: float = 0.05,
    max_peak_width_min: float = 0.15,
) -> List[dict]:
    """LC-MS EIC 피크 검출: smoothing + scipy.signal.find_peaks 기반."""
    from scipy.signal import find_peaks, savgol_filter

    if len(intensities) < 3:
        return []

    # Baseline 보정
    if baseline_method and len(intensities) >= 10:
        try:
            corrected, _, _ = correct_baseline(rts, intensities, method=baseline_method)
            work_ints = corrected
        except Exception:
            work_ints = intensities.copy()
    else:
        work_ints = intensities.copy()

    dt = float(np.median(np.diff(rts))) if len(rts) >= 2 else 0.003

    # Noise estimation (smoothing 전 raw 신호에서)
    noise = _estimate_noise(work_ints)
    raw_corrected = work_ints.copy()

    # Savitzky-Golay smoothing
    if dt > 0:
        sg_window = max(7, int(0.04 / dt) | 1)
        if sg_window % 2 == 0:
            sg_window += 1
        sg_window = min(sg_window, len(work_ints) - 1)
        if sg_window % 2 == 0:
            sg_window -= 1
        if sg_window >= 5:
            work_ints = savgol_filter(work_ints, sg_window, polyorder=2)
            work_ints = np.maximum(work_ints, 0)

    min_distance = max(5, int(min_peak_width_min / dt)) if dt > 0 else 10
    max_boundary_scans = max(10, int(max_peak_width_min / dt)) if dt > 0 else 30
    height_threshold = noise * min_sn

    peaks, _ = find_peaks(work_ints, height=height_threshold,
                          distance=min_distance, prominence=height_threshold)

    raw_results = []
    for idx in peaks:
        search_lo = max(0, idx - 3)
        search_hi = min(len(raw_corrected), idx + 4)
        apex_raw = float(np.max(raw_corrected[search_lo:search_hi]))
        sn_raw = min(apex_raw / noise, 1e6) if noise > 0 else 0.0
        left, right = _find_peak_boundaries(work_ints, idx, max_width=max_boundary_scans)
        area = _integrate_above_baseline(raw_corrected, rts, left, right)
        raw_results.append({
            "rt": rts[idx], "apex_idx": idx, "apex_intensity": apex_raw,
            "area": area, "sn": sn_raw, "left": left, "right": right,
        })

    results = _merge_overlapping_peaks(raw_results, rts, raw_corrected,
                                       min_gap_min=min_peak_width_min)

    if results:
        max_area = max(p["area"] for p in results)
        results = [p for p in results if p["area"] >= max_area * 0.005]

    return results


def integrate_peaks(
    exp,
    target_mz_list: List[float],
    mz_tolerance: float = 0.01,
    min_sn: float = 3.0,
) -> dict:
    """pyOpenMS를 이용한 피크 검출 + 적분 파이프라인."""
    all_peaks = {}
    for mz in target_mz_list:
        rts, ints = extract_eic(exp, mz, mz_tolerance)
        peaks = detect_peaks(rts, ints, min_sn)
        all_peaks[mz] = peaks
        logger.info(f"m/z={mz:.4f}: {len(peaks)} peaks detected")
    return all_peaks


# ─── 내부 함수 ─────────────────────────────────────────────

def _estimate_noise(intensities: np.ndarray) -> float:
    positive = intensities[intensities > 0]
    if len(positive) > 20:
        q25 = float(np.percentile(positive, 25))
        baseline_region = positive[positive <= q25]
        if len(baseline_region) > 5:
            noise = float(np.std(baseline_region))
            if noise <= 0:
                noise = float(np.percentile(positive, 5))
        else:
            noise = float(np.percentile(positive, 5))
    elif len(positive) > 0:
        noise = float(np.percentile(positive, 25))
    else:
        noise = 1.0
    return max(noise, 1.0)


def _find_peak_boundaries(intensities, apex_idx, max_width=None):
    n = len(intensities)
    apex_int = intensities[apex_idx]
    floor_threshold = apex_int * 0.01
    if max_width is None:
        max_width = min(150, n // 2)

    half_max = apex_int * 0.5
    half_left = apex_idx
    for i in range(apex_idx - 1, max(0, apex_idx - 150) - 1, -1):
        if intensities[i] <= half_max:
            half_left = i
            break
    half_right = apex_idx
    for i in range(apex_idx + 1, min(n, apex_idx + 150)):
        if intensities[i] <= half_max:
            half_right = i
            break
    half_width = max(half_right - half_left, 3)
    search_range = min(max(max_width, 3 * half_width), 150, n // 2)
    valley_rise = apex_int * 0.05

    left = apex_idx
    for i in range(apex_idx - 1, max(0, apex_idx - search_range) - 1, -1):
        if intensities[i] <= floor_threshold:
            left = i
            break
        if (i > 0 and intensities[i] <= intensities[i - 1]
                and (intensities[i - 1] - intensities[i]) > valley_rise):
            left = i
            break
        left = i

    right = apex_idx
    for i in range(apex_idx + 1, min(n, apex_idx + search_range + 1)):
        if intensities[i] <= floor_threshold:
            right = i
            break
        if (i < n - 1 and intensities[i] <= intensities[i + 1]
                and (intensities[i + 1] - intensities[i]) > valley_rise):
            right = i
            break
        right = i

    min_left = max(0, apex_idx - half_width)
    min_right = min(n - 1, apex_idx + half_width)
    return min(left, min_left), max(right, min_right)


def _integrate_above_baseline(intensities, rts, left, right):
    segment = intensities[left:right + 1]
    rt_segment = rts[left:right + 1]
    local_baseline = np.linspace(float(segment[0]), float(segment[-1]), len(segment))
    above = np.maximum(segment - local_baseline, 0)
    return float(np.trapezoid(above, rt_segment))


def _merge_overlapping_peaks(raw_results, rts, intensities, min_gap_min=0.03):
    if len(raw_results) <= 1:
        return [{k: v for k, v in r.items() if k != "apex_idx"} for r in raw_results]

    sorted_peaks = sorted(raw_results, key=lambda x: x["left"])
    groups = [[sorted_peaks[0]]]
    for p in sorted_peaks[1:]:
        prev = groups[-1][-1]
        if p["left"] <= prev["right"] or (p["rt"] - prev["rt"]) < min_gap_min:
            groups[-1].append(p)
        else:
            groups.append([p])

    results = []
    for group in groups:
        best = max(group, key=lambda x: x["apex_intensity"])
        merged_left = min(p["left"] for p in group)
        merged_right = max(p["right"] for p in group)
        merged_area = _integrate_above_baseline(intensities, rts, merged_left, merged_right)
        results.append({
            "rt": best["rt"], "apex_intensity": best["apex_intensity"],
            "area": merged_area, "sn": best["sn"],
            "left": merged_left, "right": merged_right,
        })
    return results
