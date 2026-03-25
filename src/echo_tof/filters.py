"""
분자식 후보 필터링 — SCIEX Sciex.GetFormula.MonoMassResults 이식.

5단계 필터링 파이프라인:
1. FitsRDB() — DBE 범위 체크
2. FitsElectronState() — 짝수/홀수 전자
3. FitsHeteroAtoms() — C/헤테로, O/S, O/P 비율
4. FitsMultipleElementRules() — 다원소 조합 규칙
5. FitsElementRatioRules() — 원소비율 규칙 (H/C, F/C, N/C 등)
"""

from __future__ import annotations
from typing import Optional

from .elements import (
    ELECTRON_MASS, PeriodicTable,
    parse_formula,
)
from .molecule import Molecule
from .calculations import get_mass_error
from .formula_enum import FindElementalCompositions


class FormulaFilter:
    """
    분자식 후보 생성 + 5단계 필터링.

    사용법:
        ff = FormulaFilter(
            mass_tolerance_ppm=5.0,
            dbe_from=-0.5,
            dbe_to=40.0,
            even_electron=True,
            odd_electron=True,
            common_rules=True,
        )
        candidates = ff.get_compositions(
            min_composition="",
            max_composition="C50 H200 N10 O10 S5",
            mono_mz=180.063,
            charge=0,
        )
    """

    def __init__(
        self,
        mass_tolerance_ppm: float = 5.0,
        int_tolerance: float = 100.0,
        min_c_to_hetero: float = 0.0,
        dbe_from: float = -0.5,
        dbe_to: float = 40.0,
        use_o_to_s: bool = False,
        use_o_to_p: bool = False,
        min_o_to_s: int = 0,
        min_o_to_p: int = 0,
        even_electron: bool = True,
        odd_electron: bool = True,
        common_rules: bool = True,
        mass_tol_in_ppm: bool = True,
    ):
        self._mass_tol = mass_tolerance_ppm
        self._int_tol = int_tolerance
        self._min_c_hetero = min_c_to_hetero
        self._dbe_from = dbe_from
        self._dbe_to = dbe_to
        self._use_o_to_s = use_o_to_s
        self._use_o_to_p = use_o_to_p
        self._min_o_to_s = min_o_to_s
        self._min_o_to_p = min_o_to_p
        self._even_ok = even_electron
        self._odd_ok = odd_electron
        self._common_rules = common_rules
        self._mass_tol_in_ppm = mass_tol_in_ppm

    def get_compositions(
        self,
        min_composition: str,
        max_composition: str,
        mono_mz: float,
        charge: int = 0,
        max_rdb: Optional[float] = None,
    ) -> list[Molecule]:
        """
        분자식 후보 생성 + 필터링.
        Returns: 필터 통과한 Molecule 리스트 (질량 오차순 정렬)
        """
        if max_rdb is None:
            max_rdb = self._dbe_to

        # 질량 공차 계산 (Da)
        if self._mass_tol_in_ppm:
            mass_tol_da = mono_mz * self._mass_tol / 1e6
        else:
            mass_tol_da = self._mass_tol / 1000.0

        # 중성 질량 계산
        if charge != 0:
            neutral_mass = mono_mz * abs(charge) + charge * ELECTRON_MASS
            mass_tol_da = abs(charge) * mass_tol_da
        else:
            neutral_mass = mono_mz

        # 원소 조합 열거
        finder = FindElementalCompositions(
            min_composition=min_composition,
            max_composition=max_composition,
        )
        compositions = finder.calculate(neutral_mass, mass_tol_da)

        # 필터링
        # C# 원본: SortedList<double, Molecule> 키 = massError + epsilon
        # epsilon은 매 루프 반복마다 1.0씩 증가 (추가 여부와 무관)
        results: list[tuple[float, Molecule]] = []
        epsilon = 1e-8
        for comp_str in compositions:
            mol = Molecule(composition=comp_str, charge=charge)

            if not self._fits_rdb(mol, max_rdb):
                epsilon += 1.0
                continue
            if not self._fits_electron_state(mol):
                epsilon += 1.0
                continue
            if not self._fits_hetero_atoms(mol):
                epsilon += 1.0
                continue
            if not self._fits_multiple_element_rules(mol):
                epsilon += 1.0
                continue
            if not self._fits_element_ratio_rules(mol):
                epsilon += 1.0
                continue

            error = get_mass_error(mol.monoisotopic_mass, mono_mz, in_ppm=False)
            sort_key = error + epsilon
            results.append((sort_key, mol))
            epsilon += 1.0

        # SortedList 동작 재현: 키 오름차순 정렬
        results.sort(key=lambda x: x[0])
        return [mol for _, mol in results]

    # ─── 5단계 필터 ─────────────────────────────────────

    def _fits_rdb(self, mol: Molecule, max_rdb: float) -> bool:
        """1단계: DBE 범위"""
        return self._dbe_from <= mol.rdb <= max_rdb

    def _fits_electron_state(self, mol: Molecule) -> bool:
        """2단계: 짝수/홀수 전자"""
        if mol.is_even_electron and self._even_ok:
            return True
        if not mol.is_even_electron and self._odd_ok:
            return True
        return False

    def _fits_hetero_atoms(self, mol: Molecule) -> bool:
        """3단계: C/헤테로, O/S, O/P 비율"""
        c_count = 0
        hetero_count = 0
        o_count = 0
        s_count = 0
        p_count = 0

        for ec in mol.element_counts:
            sym = ec.element.symbol
            if sym == "C":
                c_count = ec.count
            elif sym not in ("H", "D"):
                hetero_count += ec.count
                if sym == "S":
                    s_count = ec.count
                elif sym == "O":
                    o_count = ec.count
                elif sym == "P":
                    p_count = ec.count

        if self._use_o_to_p and p_count > 0:
            if o_count / p_count < self._min_o_to_p:
                return False

        if self._use_o_to_s and s_count > 0:
            if o_count / s_count < self._min_o_to_s:
                return False

        if hetero_count > 0:
            ratio = c_count / hetero_count
            if ratio < self._min_c_hetero:
                return False

        return True

    def _fits_multiple_element_rules(self, mol: Molecule) -> bool:
        """
        4단계: 다원소 조합 규칙.
        N, S, P, O 동시 존재 시 개수 제한.
        """
        if not self._common_rules:
            return True

        n = mol.get_element_count("N")
        s = mol.get_element_count("S")
        p = mol.get_element_count("P")
        o = mol.get_element_count("O")

        # Rule 1: N>1, S>1, O>1, P>1 → N<10, O<20, P<4, S<3
        if n > 1 and s > 1 and o > 1 and p > 1:
            if not (n < 10 and o < 20 and p < 4 and s < 3):
                return False

        # Rule 2: N>3, O>3, P>3 → N<11, O<22, P<6
        if n > 3 and o > 3 and p > 3:
            if not (n < 11 and o < 22 and p < 6):
                return False

        # Rule 3: O>1, P>1, S>1 → O<14, P<3, S<3
        if o > 1 and p > 1 and s > 1:
            if not (o < 14 and p < 3 and s < 3):
                return False

        # Rule 4: P>1, S>1, N>1 → P<3, S<3, N<4
        if p > 1 and s > 1 and n > 1:
            if not (p < 3 and s < 3 and n < 4):
                return False

        # Rule 5: N>6, O>6, S>6 → N<19, O<14, S<8
        if n > 6 and o > 6 and s > 6:
            if not (n < 19 and o < 14 and s < 8):
                return False

        return True

    def _fits_element_ratio_rules(self, mol: Molecule) -> bool:
        """
        5단계: 원소비율 규칙.
        H/C: 0.2~3.1, F/C<1.5, Cl/C<0.8, Br/C<0.8,
        N/C<1.3, O/C<1.2, P/C<0.3, S/C<0.8, Si/C<0.8
        """
        if not self._common_rules:
            return True

        c = mol.get_element_count("C")
        if c == 0:
            return True

        h_ratio = mol.get_element_count("H") / c
        if not (0.2 < h_ratio < 3.1):
            return False

        checks = [
            ("F", 1.5), ("Cl", 0.8), ("Br", 0.8),
            ("N", 1.3), ("O", 1.2), ("P", 0.3),
            ("S", 0.8), ("Si", 0.8),
        ]
        for sym, limit in checks:
            count = mol.get_element_count(sym)
            if count / c >= limit:
                return False

        return True
