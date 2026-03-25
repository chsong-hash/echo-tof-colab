"""
계산 헬퍼 — SCIEX Sciex.GetFormula.CalculationsHelper 이식.

질량 오차 계산, charge state 추정, 배열 정규화 등.
"""

from __future__ import annotations
import math
from typing import Optional

from .elements import ELECTRON_MASS
from .molecule import Molecule


def get_mass_error(theor_mass: float, exp_mass: float, in_ppm: bool = True) -> float:
    """
    질량 오차 계산.
    ppm: 1e6 * (exp - theor) / theor
    mDa: (exp - theor) * 1000
    """
    if in_ppm:
        if theor_mass == 0:
            return 0.0
        return 1e6 * (exp_mass - theor_mass) / theor_mass
    return (exp_mass - theor_mass) * 1000.0


def get_mass_error_with_charge(
    theor_neutral_mass: float,
    exp_mz: float,
    charge: int,
    in_ppm: bool = True,
) -> float:
    """전하를 고려한 질량 오차 계산"""
    neutral = exp_mz * abs(charge) + ELECTRON_MASS * charge
    if in_ppm:
        if theor_neutral_mass == 0:
            return 0.0
        return 1e6 * (neutral - theor_neutral_mass) / theor_neutral_mass
    return (neutral - theor_neutral_mass) * 1000.0


def convert_to_mz(neutral_mass: float, charge: int) -> float:
    """중성 질량 → m/z 변환"""
    if charge == 0:
        return neutral_mass
    return (neutral_mass - charge * ELECTRON_MASS) / abs(charge)


def get_charge_state_from_mz_array(mz_array: list[float]) -> int:
    """
    m/z 배열에서 charge state 추정.
    인접 피크 간격 Δmz × charge ≈ 1.0 을 찾음.
    """
    if len(mz_array) <= 1:
        return 1
    delta = mz_array[1] - mz_array[0]
    best_charge = 0
    best_diff = 100.0
    for z in range(1, 9):
        diff = abs(delta * z - 1.0)
        if diff < best_diff:
            best_charge = z
            best_diff = diff
    return best_charge


def get_max_index(array: list[float]) -> int:
    """배열에서 최댓값 인덱스"""
    if not array:
        return -1
    max_val = 0.0
    max_idx = -1
    for i, v in enumerate(array):
        if v > max_val:
            max_val = v
            max_idx = i
    return max_idx


def largest_selected_in_array(
    raw_array: list[float],
    sel_array: Optional[list[bool]] = None,
) -> int:
    """선택된 항목 중 최댓값 인덱스"""
    max_val = 0.0
    max_idx = -1
    for i, v in enumerate(raw_array):
        selected = sel_array[i] if sel_array is not None else True
        if selected and v > max_val:
            max_val = v
            max_idx = i
    return max_idx


def normalize_array(
    raw_array: list[float],
    to_first: bool = False,
    in_percent: bool = False,
) -> list[float]:
    """배열 정규화"""
    if not raw_array:
        return []
    ref = raw_array[0] if to_first else raw_array[get_max_index(raw_array)]
    if ref == 0:
        return list(raw_array)
    factor = 100.0 / ref if in_percent else 1.0 / ref
    return [v * factor for v in raw_array]


def normalize_array_to_index(
    raw_array: list[float],
    ref_index: int,
    in_percent: bool = False,
) -> list[float]:
    """특정 인덱스 기준 정규화"""
    if ref_index < 0 or ref_index >= len(raw_array) or raw_array[ref_index] == 0:
        return list(raw_array)
    factor = 1.0 / raw_array[ref_index]
    if in_percent:
        factor *= 100.0
    return [v * factor for v in raw_array]


def confidence_factors(raw_array: list[float], cutoff: float) -> list[float]:
    """S/N 기반 신뢰도 계수"""
    result = []
    for v in raw_array:
        if v >= cutoff:
            result.append(1.0)
        else:
            result.append(v / cutoff if cutoff > 0 else 0.0)
    return result


def get_ratio(top: float, bottom: float) -> float:
    """안전한 나눗셈"""
    if bottom != 0.0:
        return top / bottom
    return top * 1e7


def get_number_of_atoms(symbol: str, mol: Molecule) -> int:
    """분자에서 특정 원소의 원자 수"""
    return mol.get_element_count(symbol)


def get_overall_element_composition(compositions: list[str]) -> str:
    """여러 조성의 원소별 최댓값 합성"""
    from .elements import parse_formula, PeriodicTable
    pt = PeriodicTable.instance()
    max_counts: dict[str, int] = {}
    elements_map: dict[str, any] = {}

    for comp in compositions:
        ecs = parse_formula(comp, pt)
        for ec in ecs:
            sym = ec.element.symbol
            elements_map[sym] = ec.element
            if sym not in max_counts or ec.count > max_counts[sym]:
                max_counts[sym] = ec.count

    parts = []
    for sym, cnt in max_counts.items():
        parts.append(f"{sym}{cnt}")
    return " ".join(parts)


def convert_to_ppm_error(mda_error: float, mass: float) -> float:
    """mDa → ppm 변환"""
    if mass == 0:
        return 0.0
    return mda_error / mass * 1e3


def convert_to_mda_error(ppm_error: float, mass: float) -> float:
    """ppm → mDa 변환"""
    return mass * ppm_error / 1e6 * 1000.0


def convert_to_da_error(ppm_error: float, mass: float) -> float:
    """ppm → Da 변환"""
    return mass * ppm_error / 1e6
