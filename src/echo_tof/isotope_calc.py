"""
동위원소 분포 계산기 — SCIEX Clearcore2.ElementsAndMolecules.IsotopicDistributionCalculator 이식.

핵심 알고리즘:
- 컨볼루션 기반 동위원소 패턴 계산
- 이진 지수화(binary exponentiation) 최적화
- 질량 공차 내 피크 병합
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
from sortedcontainers import SortedList

from .elements import (
    Element, ElementCount, Isotope, PeriodicTable,
    combine_element_counts,
)


@dataclass
class IsotopicPeak:
    """
    동위원소 분포 내 단일 피크 (IsotopicDistributionItem 대응).
    """
    mass: float
    abundance: float
    normalised_abundance: float = 0.0
    _start_mass: float = 0.0
    _end_mass: float = 0.0

    def __post_init__(self):
        if self._start_mass == 0.0:
            self._start_mass = self.mass
        if self._end_mass == 0.0:
            self._end_mass = self.mass

    def merge(self, other: IsotopicPeak):
        """두 피크 병합 (가중평균 질량, 존재비 합산)"""
        self._start_mass = min(self._start_mass, other._start_mass)
        self._end_mass = max(self._end_mass, other._end_mass)
        total = self.abundance + other.abundance
        if total > 0:
            self.mass = (self.mass * self.abundance + other.mass * other.abundance) / total
            self.abundance = total

    @staticmethod
    def convolute(item1: IsotopicPeak, item2: IsotopicPeak) -> IsotopicPeak:
        """두 피크의 컨볼루션 (질량 합, 존재비 곱)"""
        return IsotopicPeak(
            mass=item1.mass + item2.mass,
            abundance=item1.abundance * item2.abundance,
            _start_mass=item1._start_mass + item2._start_mass,
            _end_mass=item1._end_mass + item2._end_mass,
        )

    def within_tolerance(self, other: IsotopicPeak, ppm_tolerance: bool, tolerance: float) -> bool:
        """두 피크가 공차 범위 내인지 확인"""
        if ppm_tolerance:
            tol = (self.mass + other.mass) / 2.0 * tolerance / 1e6
        else:
            tol = tolerance
        return (other._end_mass + tol >= self._start_mass and
                other._start_mass - tol <= self._end_mass)


class IsotopicArray:
    """
    정렬된 동위원소 피크 배열. 공차 내 피크 자동 병합.
    SortedList<IsotopicDistributionItem> 대응.
    """

    def __init__(self, ppm_tolerance: bool = False, tolerance: float = 0.1):
        self._peaks: list[IsotopicPeak] = []
        self._ppm_tolerance = ppm_tolerance
        self._tolerance = tolerance

    def add(self, peak: IsotopicPeak, min_abundance: float = 1e-20):
        """피크 추가, 공차 내 기존 피크와 병합"""
        if min_abundance < 1e-20:
            min_abundance = 1e-20

        # 병합 대상 찾기
        merged = False
        for i, existing in enumerate(self._peaks):
            if existing.within_tolerance(peak, self._ppm_tolerance, self._tolerance):
                self._peaks.pop(i)
                existing.merge(peak)
                self.add(existing, min_abundance)
                merged = True
                break

        if not merged and peak.abundance > min_abundance:
            # 질량 순서대로 삽입
            idx = 0
            for idx, p in enumerate(self._peaks):
                if p.mass > peak.mass:
                    break
            else:
                idx = len(self._peaks)
            self._peaks.insert(idx, peak)

    def convolute(self, other: IsotopicArray, min_abundance: float = 0.0) -> IsotopicArray:
        """두 배열의 컨볼루션"""
        result = IsotopicArray(self._ppm_tolerance, self._tolerance)
        for p1 in self._peaks:
            for p2 in other._peaks:
                new_peak = IsotopicPeak.convolute(p1, p2)
                result.add(new_peak, min_abundance)
        return result

    def normalize(self) -> list[IsotopicPeak]:
        """최대 존재비 기준 정규화"""
        if not self._peaks:
            return []
        max_ab = max(p.abundance for p in self._peaks)
        for p in self._peaks:
            p.normalised_abundance = p.abundance / max_ab if max_ab > 0 else 0.0
        return self._peaks

    def to_array(self, min_normalised_abundance: float = 0.0) -> list[IsotopicPeak]:
        """정규화 후 최소 존재비 이상인 피크만 반환"""
        self.normalize()
        return [p for p in self._peaks if p.normalised_abundance >= min_normalised_abundance]


class IsotopicDistributionCalculator:
    """
    동위원소 분포 계산 엔진.

    이진 지수화(binary exponentiation) 최적화:
    원소 N개 = Σ(bit_i × 2^i) 로 분해하여 컨볼루션 횟수 최소화.

    사용법:
        idc = IsotopicDistributionCalculator(ppm_tolerance=True, tolerance=50.0)
        peaks = idc.calculate("C6H12O6")
        peaks = idc.calculate_from_counts(element_counts)
    """

    def __init__(self, ppm_tolerance: bool = False, tolerance: float = 0.1):
        self._ppm_tolerance = ppm_tolerance
        self._tolerance = tolerance
        self._min_abundance = 0.0  # SCIEX 원본 기본값: 필터 없음 (MoleculePattern에서 1% 컷오프)
        self._use_min_for_calc = False
        self._cache: dict[tuple[str, int], IsotopicArray] = {}

    @property
    def min_normalised_abundance(self) -> float:
        return self._min_abundance

    @min_normalised_abundance.setter
    def min_normalised_abundance(self, value: float):
        self._min_abundance = value

    @property
    def use_min_for_calc(self) -> bool:
        return self._use_min_for_calc

    @use_min_for_calc.setter
    def use_min_for_calc(self, value: bool):
        self._use_min_for_calc = value

    def set_tolerance(self, ppm_tolerance: bool, tolerance: float):
        if self._ppm_tolerance != ppm_tolerance or self._tolerance != tolerance:
            self._ppm_tolerance = ppm_tolerance
            self._tolerance = tolerance
            self._cache.clear()

    def calculate(self, composition: str, pt: Optional[PeriodicTable] = None) -> list[IsotopicPeak]:
        """분자식 문자열에서 동위원소 분포 계산"""
        from .elements import parse_formula
        if pt is None:
            pt = PeriodicTable.instance()
        ecs = parse_formula(composition, pt)
        return self.calculate_from_counts(ecs)

    def calculate_from_counts(self, element_counts: list[ElementCount]) -> list[IsotopicPeak]:
        """ElementCount 리스트에서 동위원소 분포 계산"""
        element_counts = combine_element_counts(element_counts)

        if not element_counts:
            return []

        # 첫 번째 원소의 분포로 시작
        result_array = self._get_element_array(element_counts[0])

        # 나머지 원소들과 컨볼루션
        min_ab = 0.0
        if self._use_min_for_calc and element_counts:
            min_ab = self._min_abundance / len(element_counts)

        for i in range(1, len(element_counts)):
            elem_array = self._get_element_array(element_counts[i])
            result_array = result_array.convolute(elem_array, min_ab)

        return result_array.to_array(self._min_abundance)

    def _get_element_array(self, ec: ElementCount) -> IsotopicArray:
        """원소×개수에 대한 동위원소 배열 (캐시 사용)"""
        if ec.count < 0:
            raise ValueError(f"음수 개수 불가: {ec.element.symbol}{ec.count}")

        cache_key = (ec.element.symbol, ec.count)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._calculate_element_array(ec)
        self._cache[cache_key] = result
        return result

    def _calculate_element_array(self, ec: ElementCount) -> IsotopicArray:
        """
        이진 지수화로 원소 N개의 분포 계산.
        N = sum(bit_i * 2^i) 으로 분해하여 효율적으로 컨볼루션.
        """
        result = None
        count = ec.count
        power = 1

        while count != 0:
            if count % 2 != 0:
                power_array = self._get_element_array_power_of_two(ec.element, power)
                if result is None:
                    result = power_array
                else:
                    result = result.convolute(power_array)
            count //= 2
            power *= 2

        return result if result is not None else IsotopicArray(self._ppm_tolerance, self._tolerance)

    def _get_element_array_power_of_two(self, element: Element, count: int) -> IsotopicArray:
        """2의 거듭제곱 개수에 대한 분포 (재귀적 제곱)"""
        cache_key = (element.symbol, count)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if count == 1:
            result = self._calculate_single_element_array(element)
        else:
            half = self._get_element_array_power_of_two(element, count // 2)
            result = half.convolute(half)

        self._cache[cache_key] = result
        return result

    def _calculate_single_element_array(self, element: Element) -> IsotopicArray:
        """단일 원자의 동위원소 분포"""
        array = IsotopicArray(self._ppm_tolerance, self._tolerance)
        for iso in element.isotopes:
            if iso.abundance > 0:
                peak = IsotopicPeak(mass=iso.mass, abundance=iso.abundance)
                array.add(peak)
        return array

    def clear_cache(self):
        self._cache.clear()
