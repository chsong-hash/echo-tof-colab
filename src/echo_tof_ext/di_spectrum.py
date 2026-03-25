"""Direct Infusion 스펙트럼 처리.

ECHO-TOF는 크로마토그래피 없이 직접 주입(DI) 방식이므로,
LC-MS용 peak_integration.py 대신 이 모듈을 사용한다.

단일 mass spectrum에서 피크를 검출하고,
동위원소 클러스터를 그룹화한다.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional, Tuple
from .config import logger


def pick_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
    noise_factor: float = 3.0,
    min_intensity_pct: float = 0.1,
    local_window: int = 5,
) -> List[Dict]:
    """DI 스펙트럼에서 피크 검출.

    Parameters
    ----------
    mz : 1D array, m/z 값
    intensity : 1D array, 강도
    noise_factor : S/N 기준 배수 (median 기반)
    min_intensity_pct : 최대 피크 대비 최소 강도 (%)
    local_window : 로컬 최대값 판정 윈도우 (양쪽 포인트 수)

    Returns
    -------
    list of dict
        각 피크: {"mz", "intensity", "area", "sn", "index"}
    """
    if len(mz) == 0 or len(intensity) == 0:
        return []

    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    # 노이즈 추정: 하위 50% 강도의 median
    sorted_int = np.sort(intensity)
    noise_level = np.median(sorted_int[:max(len(sorted_int) // 2, 1)])
    if noise_level <= 0:
        noise_level = np.median(sorted_int[sorted_int > 0]) if np.any(sorted_int > 0) else 1.0

    threshold_sn = noise_level * noise_factor
    threshold_pct = np.max(intensity) * min_intensity_pct / 100.0
    threshold = max(threshold_sn, threshold_pct)

    # 로컬 최대값 검출
    peaks = []
    n = len(intensity)
    for i in range(local_window, n - local_window):
        if intensity[i] < threshold:
            continue
        window = intensity[max(0, i - local_window):i + local_window + 1]
        if intensity[i] == np.max(window):
            # 간이 면적: 삼각형 근사 (피크 폭 × 높이 / 2)
            left = max(0, i - 1)
            right = min(n - 1, i + 1)
            width = mz[right] - mz[left]
            area = intensity[i] * width * 0.5

            sn = intensity[i] / noise_level if noise_level > 0 else 0.0

            peaks.append({
                "mz": float(mz[i]),
                "intensity": float(intensity[i]),
                "area": float(area),
                "sn": float(sn),
                "index": int(i),
            })

    # 강도순 정렬
    peaks.sort(key=lambda p: p["intensity"], reverse=True)
    logger.info(f"DI 피크 검출: {len(peaks)}개 (threshold={threshold:.0f}, noise={noise_level:.0f})")
    return peaks


def group_isotope_clusters(
    peaks: List[Dict],
    charge: int = 1,
    mz_tolerance: float = 0.02,
    max_isotopes: int = 8,
) -> List[Dict]:
    """검출된 피크를 동위원소 클러스터로 그룹화.

    Parameters
    ----------
    peaks : pick_peaks() 결과
    charge : 전하 상태
    mz_tolerance : Da 단위 허용 오차
    max_isotopes : 클러스터당 최대 피크 수

    Returns
    -------
    list of dict
        각 클러스터: {
            "mono_mz": float (가장 낮은 m/z),
            "peaks": list[dict] (클러스터 내 피크들),
            "charge": int,
            "max_intensity": float,
            "total_area": float,
        }
    """
    if not peaks:
        return []

    spacing = 1.003355 / abs(charge) if charge != 0 else 1.003355
    used = set()
    clusters = []

    # 강도 높은 피크부터 클러스터 시작점으로
    sorted_peaks = sorted(peaks, key=lambda p: p["intensity"], reverse=True)

    for seed in sorted_peaks:
        if seed["index"] in used:
            continue

        cluster_peaks = [seed]
        used.add(seed["index"])

        # 상위 방향 (M+1, M+2, ...)
        for n in range(1, max_isotopes):
            expected_mz = seed["mz"] + n * spacing
            best = _find_closest_peak(peaks, expected_mz, mz_tolerance, used)
            if best is None:
                break
            cluster_peaks.append(best)
            used.add(best["index"])

        # 하위 방향 (seed가 M+1일 수도 있으므로)
        for n in range(1, 3):
            expected_mz = seed["mz"] - n * spacing
            best = _find_closest_peak(peaks, expected_mz, mz_tolerance, used)
            if best is None:
                break
            cluster_peaks.insert(0, best)
            used.add(best["index"])

        cluster_peaks.sort(key=lambda p: p["mz"])

        clusters.append({
            "mono_mz": cluster_peaks[0]["mz"],
            "peaks": cluster_peaks,
            "charge": charge,
            "max_intensity": max(p["intensity"] for p in cluster_peaks),
            "total_area": sum(p["area"] for p in cluster_peaks),
        })

    clusters.sort(key=lambda c: c["max_intensity"], reverse=True)
    logger.info(f"동위원소 클러스터: {len(clusters)}개 (charge={charge})")
    return clusters


def extract_cluster_at_mz(
    mz: np.ndarray,
    intensity: np.ndarray,
    target_mz: float,
    charge: int = 1,
    mz_tolerance_ppm: float = 10.0,
    n_isotopes: int = 5,
) -> Optional[Dict]:
    """특정 m/z 주변의 동위원소 클러스터를 추출.

    타겟 검증용: 예측된 m/z 위치에서 실측 피크를 추출하여
    isotope 패턴 매칭에 넘길 데이터를 준비한다.

    isotope spacing ~1.003/charge를 기준으로 하되,
    할로겐(Cl, Br) 등으로 인해 M+2가 강할 수 있으므로
    tolerance를 넉넉히 잡아 놓치지 않도록 한다.

    Returns
    -------
    dict or None
        {"mz_array": list[float], "int_array": list[float],
         "use_peak": list[bool], "found": bool, "sn": float}
    """
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    spacing = 1.003355 / abs(charge) if charge != 0 else 1.003355

    mz_list = []
    int_list = []
    use_list = []

    for i in range(n_isotopes):
        expected = target_mz + i * spacing
        # tolerance: ppm 기반이되 최소 0.01 Da (할로겐 패턴 대응)
        tol_da = max(expected * mz_tolerance_ppm / 1e6, 0.01)

        mask = np.abs(mz - expected) <= tol_da
        if np.any(mask):
            # 매칭 후보 중 가장 강한 피크
            actual_indices = np.where(mask)[0]
            best_local = np.argmax(intensity[actual_indices])
            best_idx = actual_indices[best_local]
            mz_list.append(float(mz[best_idx]))
            int_list.append(float(intensity[best_idx]))
            use_list.append(True)
        else:
            mz_list.append(expected)
            int_list.append(0.0)
            use_list.append(False)

    # 첫 피크(monoisotopic)가 없으면 검출 실패
    if not use_list[0] or int_list[0] <= 0:
        return None

    # S/N 추정 (로컬 노이즈)
    local_mask = (mz >= target_mz - 2) & (mz <= target_mz + n_isotopes * spacing + 2)
    local_int = intensity[local_mask]
    if len(local_int) > 0:
        # 하위 30% 강도의 중앙값을 노이즈로
        low_ints = local_int[local_int < np.percentile(local_int, 30)]
        noise = float(np.median(low_ints)) if len(low_ints) > 0 else float(np.median(local_int))
        noise = max(noise, 1.0)
    else:
        noise = 1.0

    return {
        "mz_array": mz_list,
        "int_array": int_list,
        "use_peak": use_list,
        "found": True,
        "sn": float(int_list[0] / noise),
        "mono_mz": mz_list[0],
    }


def _find_closest_peak(
    peaks: List[Dict],
    target_mz: float,
    tolerance: float,
    used: set,
) -> Optional[Dict]:
    """tolerance 내 가장 가까운 미사용 피크."""
    best = None
    best_diff = tolerance
    for p in peaks:
        if p["index"] in used:
            continue
        diff = abs(p["mz"] - target_mz)
        if diff < best_diff:
            best = p
            best_diff = diff
    return best
