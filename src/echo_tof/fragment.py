"""
MS/MS 프래그먼트 최적화 — SCIEX Sciex.GetFormula.FragmentOptimizationHelper 이식.

프래그먼트 결과에서 최적 slope 포인트 탐색.
"""

from __future__ import annotations
import math
from typing import Optional

from .math_utils import standard_deviation, get_distance, get_slope


def set_best_slope_points(
    fragment_results: list[dict],
    fr_masses: list[dict],
    formula_col: str = "formula",
    fragment_col: str = "fragment_index",
    ppm_error_col: str = "error_ppm",
    even_el_col: str = "has_even_electrons",
) -> list[int]:
    """
    최적 slope 포인트 탐색.

    fragment_results: [{"fragment_index": int, "error_ppm": float, "formula": str, ...}, ...]
    fr_masses: [{"fragment_index": int, "fragment_mz": float}, ...]
      sorted by fragment_mz ASC.

    Returns: 선택된 인덱스 리스트
    """
    if not fragment_results or not fr_masses:
        return []

    # 첫 fragment의 결과 개수 = slope 후보 수
    fr_mass_index, num_slopes = _get_num_slopes(fragment_results, fragment_col)

    if num_slopes == 0:
        return []

    all_point_indices: list[list[int]] = []
    all_distances: list[float] = []
    all_slopes: list[list[float]] = []

    for i in range(num_slopes):
        points = [i]
        slopes: list[float] = []
        distance = 0.0

        x = _get_fragment_mz(fr_masses, fr_mass_index, fragment_col)
        y = fragment_results[i][ppm_error_col]

        for j in range(1, len(fr_masses)):
            fr_idx = fr_masses[j][fragment_col]
            closest = _get_closest_point_index(
                fragment_results, y, fr_idx, fragment_col, ppm_error_col, even_el_col
            )
            if closest != -1:
                y2 = fragment_results[closest][ppm_error_col]
                x2 = fr_masses[j]["fragment_mz"]
                distance += get_distance(x, x2, y, y2)
                slopes.append(get_slope(x, x2, y, y2))
                points.append(closest)
                x = x2
                y = y2

        all_point_indices.append(points)
        all_distances.append(distance)
        all_slopes.append(slopes)

    best = _get_best_slope_index(all_distances, all_slopes, use_distance=True)
    if best >= 0:
        return all_point_indices[best]
    return []


def _get_fragment_mz(
    fr_masses: list[dict],
    fr_index: int,
    fragment_col: str,
) -> float:
    for fm in fr_masses:
        if fm[fragment_col] == fr_index:
            return fm["fragment_mz"]
    return 0.0


def _get_num_slopes(
    fragment_results: list[dict],
    fragment_col: str,
) -> tuple[int, int]:
    """(첫 fragment_index, 해당 fragment에 속한 결과 수)"""
    min_idx = min(fr[fragment_col] for fr in fragment_results)
    count = sum(1 for fr in fragment_results if fr[fragment_col] == min_idx)
    return min_idx, count


def _get_closest_point_index(
    fragment_results: list[dict],
    last_point: float,
    fr_index: int,
    fragment_col: str,
    ppm_error_col: str,
    even_el_col: str,
) -> int:
    """주어진 fragment_index에서 이전 에러값에 가장 가까운 결과 인덱스"""
    best_idx = -1
    best_diff = 1000.0

    for i in range(1, len(fragment_results)):
        fr = fragment_results[i]
        if fr[fragment_col] < fr_index:
            continue
        if fr[fragment_col] > fr_index:
            break
        diff = abs(last_point - fr[ppm_error_col])
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    return best_idx


def _get_best_slope_index(
    point_distance: list[float],
    point_slopes: list[list[float]],
    use_distance: bool = True,
) -> int:
    """최적 slope 인덱스 (최소 거리 또는 최소 표준편차)"""
    if not point_distance:
        return -1
    if len(point_distance) < 2:
        return len(point_distance) - 1

    if use_distance:
        best = 0
        best_val = point_distance[0]
        for i in range(1, len(point_distance)):
            if point_distance[i] <= best_val:
                best = i
                best_val = point_distance[i]
        return best
    else:
        best = 0
        best_sd = standard_deviation(point_slopes[0]) if point_slopes[0] else float('inf')
        for i in range(len(point_slopes)):
            sd = standard_deviation(point_slopes[i]) if point_slopes[i] else float('inf')
            if sd <= best_sd:
                best = i
                best_sd = sd
        return best
