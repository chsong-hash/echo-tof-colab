"""
수학 유틸리티 — SCIEX Sciex.Common.MiscUtils.MathHelper 이식.

선형회귀, 표준편차, 가중평균, 정규분포.
"""

from __future__ import annotations
import math
from typing import Optional
import numpy as np


def standard_deviation(values: list[float]) -> float:
    """표본 표준편차 (N-1)"""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    ss = sum((v - mean) ** 2 for v in values)
    return math.sqrt(ss / (len(values) - 1))


def linear_regression(
    x_values: list[float],
    y_values: list[float],
    weights: Optional[list[float]] = None,
) -> tuple[float, float, float]:
    """
    선형회귀.
    Returns: (slope, intercept, correlation_coefficient)
    """
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)

    if weights is not None:
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        mean_x = np.sum(w * x)
        mean_y = np.sum(w * y)
        ss_xx = np.sum(w * (x - mean_x) ** 2)
        ss_xy = np.sum(w * (x - mean_x) * (y - mean_y))
        ss_yy = np.sum(w * (y - mean_y) ** 2)
    else:
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        ss_xx = np.sum((x - mean_x) ** 2)
        ss_xy = np.sum((x - mean_x) * (y - mean_y))
        ss_yy = np.sum((y - mean_y) ** 2)

    if ss_xx == 0:
        return 0.0, mean_y, 1.0

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    if ss_yy == 0:
        r = 1.0
    else:
        r = ss_xy / math.sqrt(ss_xx * ss_yy)

    return slope, intercept, r


def get_slope(x1: float, x2: float, y1: float, y2: float) -> float:
    """두 점 사이 기울기"""
    if x2 == x1:
        return 0.0
    return (y2 - y1) / (x2 - x1)


def get_distance(x1: float, x2: float, y1: float, y2: float) -> float:
    """두 점 사이 거리"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_y_from_line(slope: float, intercept: float, x: float) -> float:
    """직선 위 y값"""
    return intercept + slope * x


def calculate_weighted_average(
    full_index: list[int],
    assigned_index: list[int],
    abundance: list[float],
    default_value: list[float],
    a_value: list[float],
) -> float:
    """가중평균 계산 (FragmentOptimization에서 사용)"""
    default_value = list(default_value)  # 복사

    for i, ai in enumerate(assigned_index):
        try:
            idx = full_index.index(ai)
            default_value[idx] = abs(a_value[i])
        except ValueError:
            pass

    total_weight = 0.0
    total_value = 0.0
    for j in range(len(abundance)):
        total_weight += abundance[j]
        total_value += abundance[j] * default_value[j]

    if total_weight <= 0:
        return 0.0
    return total_value / total_weight


def normal_distribution(x: float, mean: float, deviation: float) -> float:
    """정규분포 확률밀도"""
    if deviation == 0:
        return 0.0
    return (math.exp(-((x - mean) / deviation) ** 2 / 2.0) /
            math.sqrt(2.0 * math.pi) / deviation)


def array_to_log10(array: list[float]) -> list[float]:
    """배열 log10 변환"""
    return [math.log10(v) if v > 0 else -float('inf') for v in array]
