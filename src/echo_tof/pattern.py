"""
MoleculePattern — SCIEX Sciex.GetFormula.MoleculePattern 이식.

이론적 동위원소 패턴 생성 및 실측 데이터와 비교.
ClusterError (강도 RMS), RMSError (질량 RMS) 계산.
"""

from __future__ import annotations
import math
from typing import Optional

from .molecule import Molecule
from .isotope_calc import IsotopicDistributionCalculator, IsotopicPeak
from .calculations import convert_to_mz, get_mass_error


# 1% 미만 abundance 피크는 무시
K_PERCENTINT_CUTOFF = 1.0


class MoleculePattern:
    """
    분자식의 이론적 동위원소 패턴 생성 및 실측 비교.

    사용법:
        mp = MoleculePattern(charge=1)
        mp.calculate_pattern(molecule)
        mz = mp.pattern_mz
        intensities = mp.pattern_rel_intensities
        cluster_error = mp.get_cluster_error(measured_mz, measured_int, use_peak, weighting)
    """

    def __init__(self, charge: int = 1):
        self._charge = charge
        self._result: list[IsotopicPeak] = []
        self._idc: Optional[IsotopicDistributionCalculator] = None

    def calculate_pattern(
        self,
        molecule: Molecule,
        idc: Optional[IsotopicDistributionCalculator] = None,
    ):
        """이론적 동위원소 패턴 계산"""
        self._idc = idc or IsotopicDistributionCalculator(ppm_tolerance=True, tolerance=50.0)

        # charge state 맞추기
        if molecule.charge_state != self._charge:
            mol = Molecule(composition=molecule.composition, charge=self._charge)
        else:
            mol = molecule

        self._result = self._idc.calculate_from_counts(mol.element_counts)

    def _get_above_cutoff(self) -> list[int]:
        """1% 이상 abundance 피크 인덱스"""
        indices = []
        for i, peak in enumerate(self._result):
            if peak.normalised_abundance * 100.0 >= K_PERCENTINT_CUTOFF:
                indices.append(i)
        return indices

    @property
    def pattern_masses(self) -> list[float]:
        """이론적 질량 (중성)"""
        return [self._result[i].mass for i in self._get_above_cutoff()]

    @property
    def pattern_mz(self) -> list[float]:
        """이론적 m/z"""
        if self._charge == 0:
            return [self._result[i].mass for i in self._get_above_cutoff()]
        return [convert_to_mz(self._result[i].mass, self._charge)
                for i in self._get_above_cutoff()]

    @property
    def pattern_rel_intensities(self) -> list[float]:
        """이론적 상대 강도 (%)"""
        return [self._result[i].normalised_abundance * 100.0
                for i in self._get_above_cutoff()]

    def _get_best_result_index(self, mass: float) -> int:
        """실측 질량에 가장 가까운 이론 피크 인덱스 (0.5 Da 이내, 1% 이상)"""
        best_diff = 0.5
        best_idx = -1
        charge_mult = abs(self._charge) if self._charge != 0 else 1
        for i, peak in enumerate(self._result):
            if peak.normalised_abundance * 100.0 < K_PERCENTINT_CUTOFF:
                continue
            diff = abs(peak.mass - mass * charge_mult)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return best_idx

    def get_cluster_error(
        self,
        measured_mz: list[float],
        norm_measured_int: list[float],
        use_peak: list[bool],
        sn_correction: list[float],
    ) -> float:
        """
        강도 기반 Cluster Error (RMS).
        최대 피크 제외, S/N 가중 적용.
        """
        max_idx = 0
        max_val = 0.0
        for i, v in enumerate(norm_measured_int):
            if v > max_val:
                max_val = v
                max_idx = i

        sum_sq = 0.0
        norm_sq = norm_measured_int[max_idx] ** 2

        for i in range(len(norm_measured_int)):
            if i == max_idx or not use_peak[i]:
                continue
            charge_mult = abs(self._charge) if self._charge != 0 else 1
            best_idx = self._get_best_result_index(measured_mz[i] * charge_mult)
            if best_idx == -1:
                continue
            theor_int = self._result[best_idx].normalised_abundance * 100.0
            diff = abs(theor_int - norm_measured_int[i]) * sn_correction[i]
            sum_sq += diff ** 2
            norm_sq += norm_measured_int[i] ** 2

        if norm_sq == 0:
            return float('nan')
        return math.sqrt(sum_sq / norm_sq)

    def get_rms_error(
        self,
        measured_mz: list[float],
        use_peak: list[bool],
        sn_correction: list[float],
        adjust_mono_mz: bool = False,
    ) -> float:
        """
        질량 기반 RMS Error (ppm).
        adjust_mono_mz: True면 mono 피크 오차로 전체 보정.
        """
        count = 0
        sum_sq = 0.0
        mono_offset = 0.0

        if adjust_mono_mz and self._result:
            if self._charge == 0:
                theor_mz = self._result[0].mass
            else:
                theor_mz = convert_to_mz(self._result[0].mass, self._charge)
            mono_offset = theor_mz - measured_mz[0]

        start = 1 if adjust_mono_mz else 0

        for i in range(start, len(measured_mz)):
            if not use_peak[i]:
                continue
            best_idx = self._get_best_result_index(measured_mz[i])
            if best_idx == -1:
                continue
            if self._charge == 0:
                theor_mz = self._result[best_idx].mass
            else:
                theor_mz = convert_to_mz(self._result[best_idx].mass, self._charge)
            error = get_mass_error(theor_mz, measured_mz[i] + mono_offset, in_ppm=True)
            error *= sn_correction[i]
            sum_sq += error ** 2
            count += 1

        if count == 0:
            return float('nan')
        return math.sqrt(sum_sq / count)
