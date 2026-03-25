"""SNIP/TopHat baseline 보정 + background ion 검출.

원본: LC 분석/20260306_0443/scripts/baseline_correction.py
"""
import numpy as np
from typing import Tuple, List
from .config import logger


def snip_baseline(intensities: np.ndarray, iterations: int = 40) -> np.ndarray:
    """SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) baseline 추정."""
    y = np.log(np.log(np.sqrt(intensities + 1) + 1) + 1)
    n = len(y)
    baseline = y.copy()
    for i in range(1, iterations + 1):
        for j in range(i, n - i):
            baseline[j] = min(baseline[j], (baseline[j - i] + baseline[j + i]) / 2)
    baseline = (np.exp(np.exp(baseline) - 1) - 1) ** 2 - 1
    return np.maximum(baseline, 0)


def tophat_baseline(intensities: np.ndarray, struct_size: int = 50) -> np.ndarray:
    """TopHat (morphological) baseline 추정."""
    from scipy.ndimage import grey_erosion, grey_dilation
    eroded = grey_erosion(intensities, size=struct_size)
    return grey_dilation(eroded, size=struct_size)


def correct_baseline(
    rts: np.ndarray,
    intensities: np.ndarray,
    method: str = "snip",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Baseline 보정 수행. (corrected, baseline, delta_pct) 반환."""
    if method == "snip":
        baseline = snip_baseline(intensities, **kwargs)
    elif method == "tophat":
        baseline = tophat_baseline(intensities, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    corrected = np.maximum(intensities - baseline, 0)
    total_raw = intensities.sum()
    delta_pct = ((total_raw - corrected.sum()) / total_raw * 100) if total_raw > 0 else 0.0
    logger.info(f"Baseline 보정 ({method}): {delta_pct:.1f}% 제거")
    return corrected, baseline, delta_pct


def detect_background_ions(
    mzs: np.ndarray,
    intensities: np.ndarray,
    blank_mzs: np.ndarray,
    blank_intensities: np.ndarray,
    fold_threshold: float = 3.0,
    mz_tolerance: float = 0.01,
) -> List[float]:
    """Blank 대비 fold change가 낮은 background ion 검출."""
    background_ions = []
    for i, mz in enumerate(blank_mzs):
        if blank_intensities[i] <= 0:
            continue
        mask = np.abs(mzs - mz) <= mz_tolerance
        if mask.any():
            fold = intensities[mask].max() / blank_intensities[i]
            if fold < fold_threshold:
                background_ions.append(float(mz))
    return background_ions
