"""
원소 조합 열거 — SCIEX Clearcore2.ElementsAndMolecules.FindElementalCompositions 이식.

주어진 질량에서 가능한 모든 원소 조합을 찾는 엔진.
질량 제약 기반 가지치기로 효율적 탐색.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .elements import (
    Element, ElementCount, PeriodicTable,
    parse_formula, formula_to_string,
    calculate_rdb,
)
from .molecule import Molecule


class ElectronState(Enum):
    BOTH = "both"
    ODD_ONLY = "odd_only"
    EVEN_ONLY = "even_only"


@dataclass
class ElementRange:
    """원소 범위: 최소~최대 원자 수"""
    element: Element
    min_number: int = 0
    max_number: int = 0
    _current_number: int = 0
    _current_mass: float = 0.0

    @property
    def current_number(self) -> int:
        return self._current_number

    @property
    def current_mass(self) -> float:
        return self._current_mass

    @property
    def min_possible_mass(self) -> float:
        return self.min_number * self.element.monoisotopic_mass

    @property
    def max_possible_mass(self) -> float:
        return self.max_number * self.element.monoisotopic_mass

    @property
    def is_at_minimum(self) -> bool:
        return self._current_number == self.min_number

    def set_current_number(self, n: int):
        self._current_number = n
        self._current_mass = n * self.element.monoisotopic_mass

    def set_current_constrained(self, n: int):
        n = max(self.min_number, min(self.max_number, n))
        self.set_current_number(n)

    def reset(self):
        self.set_current_number(self.min_number)

    def next(self) -> bool:
        """다음 값으로 이동. 최대 도달시 최소로 리셋하고 False 반환."""
        if self._current_number == self.max_number:
            self.set_current_number(self.min_number)
            return False
        self.set_current_number(self._current_number + 1)
        return True

    def validate(self):
        if self.min_number > self.max_number:
            raise ValueError(f"Min > Max for element {self.element.symbol}")
        if self.min_number < self.max_number and self.element.monoisotopic_mass <= 0:
            raise ValueError(f"Non-positive mass for {self.element.symbol}")


class FindElementalCompositions:
    """
    주어진 질량에 맞는 원소 조합 열거 엔진.

    사용법:
        finder = FindElementalCompositions(
            min_composition="",
            max_composition="C50 H200 N10 O10 S5",
        )
        results = finder.calculate(target_mass=180.063, mass_tolerance=0.005)
        # results = ["C6H12O6", "C5H8N2O5", ...]
    """

    def __init__(
        self,
        min_composition: str = "",
        max_composition: str = "C50 H200 N10 O10 S5",
        unlimited_max_composition: str = "",
        electron_state: ElectronState = ElectronState.BOTH,
        pt: Optional[PeriodicTable] = None,
    ):
        self._pt = pt or PeriodicTable.instance()
        self._electron_state = electron_state
        self._element_ranges: list[ElementRange] = []
        self._element_counts: list[ElementCount] = []
        self._min_mass = 0.0
        self._max_mass = 0.0

        self._setup_ranges(min_composition, max_composition, unlimited_max_composition)
        self._sort_ranges()
        self._allocate_counts()

    def _setup_ranges(self, min_comp: str, max_comp: str, unlimited_comp: str):
        """원소 범위 설정"""
        # 모든 원소 수집
        all_elements: dict[str, Element] = {}
        for comp in [min_comp, max_comp, unlimited_comp]:
            if not comp.strip():
                continue
            ecs = parse_formula(comp, self._pt)
            for ec in ecs:
                all_elements[ec.element.symbol] = ec.element

        # ElementRange 생성
        self._element_ranges = [ElementRange(element=elem) for elem in all_elements.values()]

        # 최소 조성 설정
        if min_comp.strip():
            for ec in parse_formula(min_comp, self._pt):
                er = self._get_range(ec.element)
                if er:
                    er.min_number = ec.count
                    er.max_number = 2**63 - 1  # long.MaxValue 대신

        # unlimited 설정
        if unlimited_comp.strip():
            for ec in parse_formula(unlimited_comp, self._pt):
                er = self._get_range(ec.element)
                if er:
                    er.max_number = 2**63 - 1

        # 최대 조성 설정
        if max_comp.strip():
            for ec in parse_formula(max_comp, self._pt):
                er = self._get_range(ec.element)
                if er:
                    er.max_number = ec.count
                    er.validate()

    def _sort_ranges(self):
        """첫 번째 원소를 range/mass 비율이 가장 큰 것으로 설정 (최적화)"""
        if not self._element_ranges:
            return
        best_idx = 0
        best_ratio = ((self._element_ranges[0].max_number - self._element_ranges[0].min_number)
                      / max(self._element_ranges[0].element.monoisotopic_mass, 1e-10))
        for i in range(1, len(self._element_ranges)):
            er = self._element_ranges[i]
            ratio = (er.max_number - er.min_number) / max(er.element.monoisotopic_mass, 1e-10)
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i
        if best_idx != 0:
            self._element_ranges[0], self._element_ranges[best_idx] = (
                self._element_ranges[best_idx], self._element_ranges[0])

    def _allocate_counts(self):
        self._element_counts = [
            ElementCount(er.element, 0) for er in self._element_ranges
        ]

    def _get_range(self, element: Element) -> Optional[ElementRange]:
        for er in self._element_ranges:
            if er.element.symbol == element.symbol:
                return er
        return None

    @property
    def _total_min_mass(self) -> float:
        return sum(er.min_possible_mass for er in self._element_ranges)

    @property
    def _total_max_mass(self) -> float:
        return sum(er.max_possible_mass for er in self._element_ranges)

    def calculate(
        self,
        target_mass: float,
        mass_tolerance: float,
        max_results: int = 10000,
    ) -> list[str]:
        """
        목표 질량 ± 공차 범위 내 모든 조합 열거.

        Returns: 분자식 문자열 리스트
        """
        if mass_tolerance < 0:
            raise ValueError("Mass tolerance cannot be negative")
        if max_results <= 0:
            raise ValueError("Max results must be positive")

        self._min_mass = target_mass - mass_tolerance
        self._max_mass = target_mass + mass_tolerance

        if self._max_mass < self._total_min_mass:
            raise ValueError(
                f"Maximum mass ({self._max_mass:.6f}) is less than minimum possible mass "
                f"({self._total_min_mass:.6f}). Composition constraints incompatible with target mass."
            )
        if self._min_mass > self._total_max_mass:
            raise ValueError(
                f"Minimum mass ({self._min_mass:.6f}) exceeds maximum possible mass "
                f"({self._total_max_mass:.6f}). Composition constraints incompatible with target mass."
            )

        # 모든 범위 리셋
        for er in self._element_ranges:
            er.reset()

        return self._enumerate(max_results)

    def _enumerate(self, max_results: int) -> list[str]:
        """조합 열거 (중첩 루프 시뮬레이션)"""
        results: list[str] = []

        while True:
            # 첫 번째 원소의 최적 카운트 초기화
            current_mass = self._init_first_element()

            if current_mass > self._max_mass:
                if not self._move_to_next_element():
                    break
                continue

            if current_mass >= self._min_mass:
                comp = self._get_current_composition()
                if comp:
                    results.append(comp)
                    if len(results) > max_results:
                        raise RuntimeError(f"Too many results (>{max_results})")

            if not self._move_to_next(0):
                break

        return results

    def _init_first_element(self) -> float:
        """첫 번째 원소의 카운트를 최적화 (질량 하한에 맞게 점프)"""
        other_mass = sum(self._element_ranges[i].current_mass
                        for i in range(1, len(self._element_ranges)))

        mono_mass = self._element_ranges[0].element.monoisotopic_mass
        if mono_mass > 0:
            needed = int((self._min_mass - other_mass) / mono_mass - 0.001)
            if self._element_ranges[0].current_number < needed:
                self._element_ranges[0].set_current_constrained(needed)

        return other_mass + self._element_ranges[0].current_mass

    def _get_current_composition(self) -> str:
        """현재 조합의 분자식 문자열 반환 (전자 상태 필터 포함)"""
        all_zero = True
        for i, er in enumerate(self._element_ranges):
            self._element_counts[i] = ElementCount(er.element, er.current_number)
            if er.current_number > 0:
                all_zero = False

        if all_zero:
            return ""

        # 전자 상태 필터
        if self._electron_state != ElectronState.BOTH:
            mol = Molecule(element_counts=self._element_counts)
            if mol.is_even_electron and self._electron_state == ElectronState.ODD_ONLY:
                return ""
            if not mol.is_even_electron and self._electron_state == ElectronState.EVEN_ONLY:
                return ""

        return formula_to_string(self._element_counts)

    def _move_to_next_element(self) -> bool:
        """현재 질량이 너무 클 때 다른 원소로 이동"""
        for i in range(len(self._element_ranges)):
            if not self._element_ranges[i].is_at_minimum:
                self._element_ranges[i].reset()
                return self._move_to_next(i + 1)
        return False

    def _move_to_next(self, start_range: int) -> bool:
        """다음 조합으로 이동"""
        for i in range(start_range, len(self._element_ranges)):
            if self._element_ranges[i].next():
                return True
        return False


def find_compositions(
    target_mass: float,
    mass_tolerance: float,
    min_composition: str = "",
    max_composition: str = "C50 H200 N10 O10 S5",
    electron_state: ElectronState = ElectronState.BOTH,
    max_results: int = 10000,
) -> list[str]:
    """편의 함수: 원소 조합 열거"""
    finder = FindElementalCompositions(
        min_composition=min_composition,
        max_composition=max_composition,
        electron_state=electron_state,
    )
    return finder.calculate(target_mass, mass_tolerance, max_results)
