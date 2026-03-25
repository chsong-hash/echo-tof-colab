"""
Molecule 클래스 — SCIEX Clearcore2.ElementsAndMolecules.Molecule 이식.

분자식 객체: 질량 계산, RDB, 전자 상태, 분자 연산 (Add/Subtract/Contains).
"""

from __future__ import annotations
import math
from typing import Optional

from .elements import (
    Element, ElementCount, PeriodicTable,
    parse_formula, formula_to_string,
    get_monoisotopic_mass, get_average_mass, get_nominal_mass,
    calculate_rdb, combine_element_counts,
    ELECTRON_MASS,
)


class Molecule:
    """
    분자식 객체.

    사용법:
        mol = Molecule("C6H12O6")
        mol = Molecule("C6H12O6", charge=1)
        mol = Molecule(element_counts=[...], charge=0)
    """

    def __init__(
        self,
        composition: str = "",
        charge: int = 0,
        name: str = "",
        element_counts: Optional[list[ElementCount]] = None,
        pt: Optional[PeriodicTable] = None,
    ):
        self._name = name
        self._charge = charge
        self._composition_str: Optional[str] = None

        if element_counts is not None:
            self._element_counts = list(element_counts)
        elif composition:
            self._pt = pt or PeriodicTable.instance()
            self._element_counts = parse_formula(composition, self._pt)
            self._composition_str = composition
        else:
            self._element_counts = []

        # 캐시
        self._monoisotopic_mass: Optional[float] = None
        self._average_mass: Optional[float] = None
        self._nominal_mass: Optional[int] = None
        self._rdb: Optional[float] = None

    # ─── 기본 속성 ──────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def charge_state(self) -> int:
        return self._charge

    @property
    def element_counts(self) -> list[ElementCount]:
        return self._element_counts

    @property
    def composition(self) -> str:
        """C# Cannonicalise() + GetCannonicalEmpiricalFormula 대응: 중복 합산 후 Hill 순서"""
        if self._composition_str is None:
            self._composition_str = formula_to_string(combine_element_counts(self._element_counts))
        return self._composition_str

    @property
    def is_empty(self) -> bool:
        """C# ElementCountList.IsEmpty 대응: 합산 후 양수 카운트가 없으면 빈 분자"""
        combined = combine_element_counts(self._element_counts)
        return len(combined) == 0

    # ─── 질량 관련 ──────────────────────────────────────

    @property
    def monoisotopic_mass(self) -> float:
        """전하 보정된 monoisotopic mass"""
        if self._monoisotopic_mass is None:
            mass = get_monoisotopic_mass(self._element_counts)
            mass -= self._charge * ELECTRON_MASS
            self._monoisotopic_mass = mass
        return self._monoisotopic_mass

    @property
    def average_mass(self) -> float:
        if self._average_mass is None:
            mass = get_average_mass(self._element_counts)
            mass -= self._charge * ELECTRON_MASS
            self._average_mass = mass
        return self._average_mass

    @property
    def nominal_mass(self) -> int:
        if self._nominal_mass is None:
            self._nominal_mass = get_nominal_mass(self._element_counts)
        return self._nominal_mass

    @property
    def neutral_mono_mass(self) -> float:
        """전하 보정 전 중성 질량"""
        return self.monoisotopic_mass + self._charge * ELECTRON_MASS

    @property
    def mono_mass_to_charge(self) -> float:
        """m/z 값"""
        if self._charge == 0:
            return self.monoisotopic_mass
        return self.monoisotopic_mass / abs(self._charge)

    @property
    def average_mass_to_charge(self) -> float:
        if self._charge == 0:
            return self.average_mass
        return self.average_mass / abs(self._charge)

    @property
    def mass_defect(self) -> float:
        return self.neutral_mono_mass - self.nominal_mass

    @property
    def observed_mass_defect(self) -> float:
        return self.neutral_mono_mass - round(self.neutral_mono_mass)

    # ─── RDB / 전자 상태 ──────────────────────────────────

    @property
    def rdb(self) -> float:
        """Ring and Double Bond equivalents"""
        if self._rdb is None:
            r = calculate_rdb(self._element_counts)
            r += 0.5 * self._charge
            self._rdb = r
        return self._rdb

    @property
    def is_even_electron(self) -> bool:
        n = int(2.0 * abs(self.rdb) + 0.5)
        return n % 2 == 0

    # ─── 분자 연산 ──────────────────────────────────────

    def contains(self, other: Molecule) -> bool:
        """other 분자가 self에 포함되는지 확인 (C# ElementCountList.Subtract + AnyPositiveCounts 대응)"""
        self_counts = {ec.element.symbol: ec.count for ec in combine_element_counts(self._element_counts)}
        for ec in combine_element_counts(other._element_counts):
            if self_counts.get(ec.element.symbol, 0) < ec.count:
                return False
        return True

    def add(self, other: Molecule) -> Molecule:
        """두 분자 합산"""
        counts: dict[str, ElementCount] = {}
        for ec in self._element_counts:
            counts[ec.element.symbol] = ElementCount(ec.element, ec.count)
        for ec in other._element_counts:
            sym = ec.element.symbol
            if sym in counts:
                counts[sym] = ElementCount(ec.element, counts[sym].count + ec.count)
            else:
                counts[sym] = ElementCount(ec.element, ec.count)
        result = [ec for ec in counts.values() if ec.count != 0]
        return Molecule(
            element_counts=result,
            charge=self._charge + other._charge,
            name=self._name,
        )

    def subtract(self, other: Molecule) -> Molecule:
        """분자 뺄셈"""
        counts: dict[str, ElementCount] = {}
        for ec in self._element_counts:
            counts[ec.element.symbol] = ElementCount(ec.element, ec.count)
        for ec in other._element_counts:
            sym = ec.element.symbol
            if sym in counts:
                counts[sym] = ElementCount(ec.element, counts[sym].count - ec.count)
            else:
                counts[sym] = ElementCount(ec.element, -ec.count)
        result = [ec for ec in counts.values() if ec.count != 0]
        return Molecule(
            element_counts=result,
            charge=self._charge - other._charge,
            name=self._name,
        )

    def multiply(self, value: int, multiply_charge: bool = False) -> Molecule:
        if value == 1:
            return self
        new_counts = [ElementCount(ec.element, ec.count * value) for ec in self._element_counts]
        new_charge = self._charge * value if multiply_charge else self._charge
        return Molecule(element_counts=new_counts, charge=new_charge, name=self._name)

    @property
    def any_negative_subscripts(self) -> bool:
        return any(ec.count < 0 for ec in self._element_counts)

    # ─── 원소 개수 조회 ─────────────────────────────────

    def get_element_count(self, symbol: str) -> int:
        """특정 원소의 개수 반환"""
        for ec in self._element_counts:
            if ec.element.symbol == symbol:
                return ec.count
        return 0

    # ─── 비교/해시 ──────────────────────────────────────

    def __eq__(self, other):
        if not isinstance(other, Molecule):
            return False
        if self._charge != other._charge:
            return False
        self_dict = {ec.element.symbol: ec.count for ec in self._element_counts}
        other_dict = {ec.element.symbol: ec.count for ec in other._element_counts}
        return self_dict == other_dict

    def __hash__(self):
        items = tuple(sorted((ec.element.symbol, ec.count) for ec in self._element_counts))
        return hash((items, self._charge))

    def __repr__(self):
        charge_str = ""
        if self._charge > 0:
            charge_str = f"+{self._charge}" if self._charge > 1 else "+"
        elif self._charge < 0:
            charge_str = str(self._charge) if self._charge < -1 else "-"
        return f"Molecule({self.composition}{charge_str})"

    def __str__(self):
        return self.composition
