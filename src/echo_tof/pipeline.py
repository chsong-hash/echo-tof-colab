"""
전체 파이프라인 오케스트레이터 — SCIEX Sciex.GetFormula.PatternResults 이식.

MS 동위원소 패턴 매칭 + MS/MS 프래그먼트 분석 통합.
4가지 오차 기반 순위 + Combined Score.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from .molecule import Molecule
from .elements import ELECTRON_MASS, PeriodicTable
from .isotope_calc import IsotopicDistributionCalculator
from .pattern import MoleculePattern
from .filters import FormulaFilter
from .calculations import (
    get_mass_error, get_mass_error_with_charge, convert_to_mz,
    get_max_index, largest_selected_in_array, normalize_array_to_index,
    get_ratio, get_number_of_atoms, get_overall_element_composition,
    convert_to_da_error,
)
from .elements import contains_halogen
from .math_utils import linear_regression, get_y_from_line, calculate_weighted_average


@dataclass
class CompositionResult:
    """단일 분자식 후보 결과"""
    composition: str
    monoisotopic_mass: float
    mono_mz: float
    error_mda: float
    error_ppm: float
    abs_error_ppm: float
    rdb: float
    is_even_electron: bool

    # 패턴 매칭 오차
    intensity_cluster_error: float = 0.0
    rms_error: float = 0.0
    rms_w_int_error: float = 0.0
    mass_cluster_error: float = 0.0  # = abs(rms) + 100 * cluster_error

    # 순위
    int_cluster_order: int = 0
    rms_error_order: int = 0
    mass_cluster_order: int = 0
    rms_w_int_order: int = 0
    overall_order: int = 0

    # MS/MS
    msms_error_ppm: float = 0.0
    abs_msms_error_ppm: float = 0.0
    num_msms_peaks: int = 0
    msms_order: int = 0

    # Combined
    combined_order: float = 0.0

    # Optimized MSMS
    opt_msms_slope: float = 0.0
    opt_msms_intercept: float = 0.0
    opt_msms_corr_coeff: float = 0.0
    opt_msms_error_ppm: float = 0.0

    # 피크 상세
    rel_diff_peaks_1and2: float = 0.0
    peak_details: list = field(default_factory=list)

    # 프래그먼트 상세
    fragment_results: list = field(default_factory=list)


@dataclass
class PeakDetail:
    """패턴 매칭 개별 피크 상세"""
    isotope_index: int
    theoretical_mz: float
    theoretical_intensity: float
    is_used: bool
    weighting: float
    error_ppm: float
    rel_int_error: float


@dataclass
class FragmentResult:
    """MS/MS 프래그먼트 할당 결과"""
    fragment_mz: float
    formula: str
    error_ppm: float
    abs_error_ppm: float
    error_mda: float
    abs_error_mda: float
    rdb: float
    is_even_electron: bool
    fragment_index: int = -1  # C# FragmentIndex 대응


class FormulaFinderPipeline:
    """
    전체 Formula Finder 파이프라인.

    사용법:
        pipeline = FormulaFinderPipeline()
        pipeline.init_ms(
            mass_tol_ppm=5.0,
            int_tol=100.0,
            min_comp="", max_comp="C50 H200 N10 O10 S5",
            dbe_from=-0.5, dbe_to=40.0,
        )
        pipeline.set_pattern(mz_array, int_array, use_peaks, weighting, charge)
        count = pipeline.propose_elemental_compositions()
        results = pipeline.results
    """

    DEFAULT_DBE_GAIN = 3

    def __init__(self, idc: Optional[IsotopicDistributionCalculator] = None):
        self._idc = idc or IsotopicDistributionCalculator(ppm_tolerance=True, tolerance=50.0)
        self._results: list[CompositionResult] = []

        # MS 설정
        self._mass_tol_ppm = 5.0
        self._int_tol = 100.0
        self._c_to_hetero = 0.0
        self._min_comp = ""
        self._max_comp = "C50 H200 N10 O10 S5"
        self._dbe_from = -0.5
        self._dbe_to = 40.0
        self._use_o_to_s = False
        self._use_o_to_p = False
        self._min_o_to_s = 0
        self._min_o_to_p = 0
        self._odd_electron_ok = True
        self._common_rules = True
        self._dbe_gain = self.DEFAULT_DBE_GAIN

        # 패턴 데이터
        self._charge = 1
        self._pattern_mz: list[float] = []
        self._pattern_int: list[float] = []
        self._pattern_weighting: list[float] = []
        self._use_peak: list[bool] = []
        self._norm_ref_peak = 0

        # 프래그먼트 데이터
        self._fragment_mz: Optional[list[float]] = None
        self._fragment_int: Optional[list[float]] = None
        self._fragment_charge: Optional[list[int]] = None

        # 후보 분자
        self._potential_molecules: list[Molecule] = []

    @property
    def results(self) -> list[CompositionResult]:
        return self._results

    @property
    def has_fragment_data(self) -> bool:
        return self._fragment_mz is not None

    def init_ms(
        self,
        mass_tol_ppm: float = 5.0,
        int_tol: float = 100.0,
        c_to_hetero: float = 0.0,
        min_comp: str = "",
        max_comp: str = "C50 H200 N10 O10 S5",
        dbe_from: float = -0.5,
        dbe_to: float = 40.0,
        use_o_to_s: bool = False,
        use_o_to_p: bool = False,
        min_o_to_s: int = 0,
        min_o_to_p: int = 0,
        even_electron: bool = True,
        odd_electron: bool = True,
        common_rules: bool = True,
    ):
        """MS 파라미터 초기화"""
        self._mass_tol_ppm = mass_tol_ppm
        self._int_tol = int_tol
        self._c_to_hetero = c_to_hetero
        self._min_comp = min_comp
        self._max_comp = max_comp
        self._dbe_from = dbe_from
        self._dbe_to = dbe_to
        self._use_o_to_s = use_o_to_s
        self._use_o_to_p = use_o_to_p
        self._min_o_to_s = min_o_to_s
        self._min_o_to_p = min_o_to_p
        self._odd_electron_ok = odd_electron
        self._common_rules = common_rules
        self._results.clear()
        self._potential_molecules.clear()

    def set_pattern(
        self,
        pattern_mz: list[float],
        pattern_int: list[float],
        use_peak: list[bool],
        peak_confidence: list[float],
        charge: int = 1,
    ):
        """실측 동위원소 패턴 설정"""
        self._charge = charge
        self._pattern_mz = list(pattern_mz)
        self._use_peak = list(use_peak)
        self._pattern_weighting = list(peak_confidence)

        # 선택된 피크 중 최대 강도 인덱스
        self._norm_ref_peak = largest_selected_in_array(pattern_int, use_peak)

        # 참조 피크 기준 정규화 (%)
        self._pattern_int = normalize_array_to_index(
            pattern_int, self._norm_ref_peak, in_percent=True
        )

    def set_fragments(
        self,
        mz: list[float],
        abundance: list[float],
        charge: list[int],
    ):
        """MS/MS 프래그먼트 데이터 설정"""
        if not mz:
            self._fragment_mz = None
            self._fragment_int = None
            self._fragment_charge = None
            return

        self._fragment_mz = list(mz)
        ref_idx = largest_selected_in_array(abundance, None)
        self._fragment_int = normalize_array_to_index(abundance, ref_idx, in_percent=True)

        # charge sign 보정
        is_positive = self._charge >= 0
        self._fragment_charge = []
        for c in charge:
            if c > 0 and not is_positive:
                self._fragment_charge.append(-c)
            else:
                self._fragment_charge.append(c)

    def propose_elemental_compositions(self) -> int:
        """
        분자식 후보 제안: 패턴 매칭 + 필터링.
        Returns: 찾은 후보 수
        """
        ff = FormulaFilter(
            mass_tolerance_ppm=self._mass_tol_ppm,
            int_tolerance=self._int_tol,
            min_c_to_hetero=self._c_to_hetero,
            dbe_from=self._dbe_from,
            dbe_to=self._dbe_to,
            use_o_to_s=self._use_o_to_s,
            use_o_to_p=self._use_o_to_p,
            min_o_to_s=self._min_o_to_s,
            min_o_to_p=self._min_o_to_p,
            even_electron=True,
            odd_electron=self._odd_electron_ok,
            common_rules=self._common_rules,
        )

        candidates = ff.get_compositions(
            self._min_comp, self._max_comp,
            self._pattern_mz[0], self._charge,
        )

        count = 0
        for mol in candidates:
            if self._add_one_valid_composition(mol):
                count += 1

        if count > 0:
            self._update_composition_ranking()

        return count

    def assign_overall_score(self, ms_contribution: float = 0.5):
        """
        MS + MS/MS 결합 스코어 계산.
        Combined = MSMSOrder × 4 × (1 - MSContribution) + OverallOrder × MSContribution
        """
        for r in self._results:
            r.combined_order = (
                r.msms_order * 4 * (1.0 - ms_contribution) +
                r.overall_order * ms_contribution
            )

    def calculate_fragment_data(
        self,
        mass_tolerance: float = 5.0,
        min_c_to_hetero: float = 0.0,
        odd_electron: bool = False,
        common_odd_electron: bool = True,
        msms_in_log: bool = False,
        mass_tol_in_ppm: bool = True,
    ):
        """
        MS/MS 프래그먼트 할당.
        C# PatternResults.CalculateFragmentData 이식.

        각 프래그먼트에 대해 가능한 분자식 찾아 부모 분자와 매칭.
        + FragmentsForComposition → WeightedFragmentError → MSMSErrorPPM 계산.
        """
        if self._fragment_mz is None:
            return

        # 물 손실 체인 기반 DBE gain 동적 설정
        self._set_dbe_gain_from_water_losses(mass_tolerance, mass_tol_in_ppm)

        # 기존 프래그먼트 결과 초기화
        for r in self._results:
            r.fragment_results.clear()

        # 전체 원소 범위 (모든 후보의 최대)
        all_comps = [r.composition for r in self._results]
        overall_comp = get_overall_element_composition(all_comps)

        # C# 원본: GetMonoMassResults에서 bOddElectron || bCommonOddElectron 사용
        ff = FormulaFilter(
            mass_tolerance_ppm=mass_tolerance,
            int_tolerance=100.0,
            min_c_to_hetero=min_c_to_hetero,
            dbe_from=self._dbe_from,
            dbe_to=self._dbe_to,
            use_o_to_s=self._use_o_to_s,
            use_o_to_p=self._use_o_to_p,
            min_o_to_s=self._min_o_to_s,
            min_o_to_p=self._min_o_to_p,
            even_electron=True,
            odd_electron=odd_electron or common_odd_electron,
            common_rules=False,
            mass_tol_in_ppm=mass_tol_in_ppm,
        )

        # C# 원본: _fragmentIndices 배열 (각 프래그먼트의 데이터셋 인덱스)
        fragment_indices = list(range(len(self._fragment_mz)))

        # 프래그먼트 강도 (log 변환 옵션 — C# AddOneFragmentToDataset 대응)
        frag_int_for_weighting = []
        for idx in range(len(self._fragment_mz)):
            val = self._fragment_int[idx]
            if msms_in_log and val > 0:
                frag_int_for_weighting.append(math.log10(val))
            else:
                frag_int_for_weighting.append(val)

        for i, frag_mz in enumerate(self._fragment_mz):
            frag_charge = self._fragment_charge[i]

            # SCIEX 원본: bConsiderOddEl = bOddElectron || (bCommonOddElectron && HasComplementFragment(i))
            consider_odd_el = odd_electron or (common_odd_electron and self._has_complement_fragment(i))

            frag_candidates = ff.get_compositions(
                "", overall_comp, frag_mz, frag_charge,
                max_rdb=self._dbe_to,
            )

            for frag_mol in frag_candidates:
                # 1단계: MoleculeValidForCompositions 프리필터 (RDB + dbeGain, +0.5 없음)
                if not self._molecule_valid_for_compositions(frag_mol, consider_odd_el):
                    continue

                # 2단계: 개별 부모와 매칭 (RDB + dbeGain + 0.5)
                matched_result_indices = []
                for k, result in enumerate(self._results):
                    parent_mol = self._potential_molecules[k]
                    if (parent_mol.contains(frag_mol) and
                            parent_mol.rdb + self._dbe_gain + 0.5 >= frag_mol.rdb):
                        matched_result_indices.append(k)

                if not matched_result_indices:
                    continue

                error_ppm = get_mass_error_with_charge(
                    frag_mol.neutral_mono_mass, frag_mz, frag_charge, in_ppm=True
                )
                error_mda = get_mass_error_with_charge(
                    frag_mol.neutral_mono_mass, frag_mz, frag_charge, in_ppm=False
                )
                fr = FragmentResult(
                    fragment_mz=frag_mz,
                    formula=frag_mol.composition,
                    error_ppm=error_ppm,
                    abs_error_ppm=abs(error_ppm),
                    error_mda=error_mda,
                    abs_error_mda=abs(error_mda),
                    rdb=frag_mol.rdb,
                    is_even_electron=frag_mol.is_even_electron,
                    fragment_index=i,
                )
                for k in matched_result_indices:
                    self._results[k].fragment_results.append(fr)

        # ── C# FragmentsForComposition + WeightedFragmentError 대응 ──
        # 각 부모 후보에 대해 MSMSErrorPPM, AbsMSMSErrorPPM, NumMSMSPeaks 계산
        for result in self._results:
            frag_errors = self._fragments_for_composition(
                result, mass_tolerance, mass_tol_in_ppm,
            )
            result.num_msms_peaks = len(frag_errors)
            result.msms_error_ppm = self._weighted_fragment_error(
                frag_errors, mass_tolerance, mass_tol_in_ppm,
                frag_int_for_weighting,
            )
            result.abs_msms_error_ppm = self._weighted_assigned_fragment_error(
                frag_errors, frag_int_for_weighting,
            )

        # MS/MS 순위 업데이트
        self._update_msms_ranking()

    def _fragments_for_composition(
        self,
        result: CompositionResult,
        mass_tolerance: float,
        mass_tol_in_ppm: bool,
    ) -> dict:
        """
        C# FragmentsForComposition 이식.
        각 프래그먼트에 대해 최소 abs error를 가진 결과만 유지.
        Returns: {fragment_index: min_abs_error_ppm}
        """
        frag_errors: dict[int, float] = {}

        for fr in result.fragment_results:
            fi = fr.fragment_index
            if fi < 0:
                continue

            # C# 원본: num2 = massTolerancemDa / _fragmentMZ[num] * 1000.0
            if mass_tol_in_ppm:
                tol_ppm = mass_tolerance
            else:
                tol_ppm = mass_tolerance / self._fragment_mz[fi] * 1000.0

            if fi in frag_errors:
                if fr.abs_error_ppm < frag_errors[fi]:
                    frag_errors[fi] = fr.abs_error_ppm
            elif fr.abs_error_ppm <= tol_ppm:
                frag_errors[fi] = fr.abs_error_ppm

        return frag_errors

    def _weighted_fragment_error(
        self,
        frag_errors: dict,
        mass_tolerance: float,
        mass_tol_in_ppm: bool,
        frag_intensities: list[float],
    ) -> float:
        """
        C# WeightedFragmentError 이식.
        할당된 프래그먼트: 실제 오차. 미할당: tolerance 패널티.
        강도 가중 평균.
        """
        abundances = []
        errors = []

        # 할당된 프래그먼트
        for fi, err in frag_errors.items():
            abundances.append(frag_intensities[fi])
            errors.append(err)

        # 미할당 프래그먼트 — 패널티 = tolerance
        for i in range(len(self._fragment_mz)):
            if i not in frag_errors:
                abundances.append(frag_intensities[i])
                if mass_tol_in_ppm:
                    errors.append(mass_tolerance)
                else:
                    errors.append(mass_tolerance / self._fragment_mz[i] * 1000.0)

        return self._calculate_weighted_error(abundances, errors)

    def _weighted_assigned_fragment_error(
        self,
        frag_errors: dict,
        frag_intensities: list[float],
    ) -> float:
        """
        C# WeightedAssignedFragmentError 이식.
        할당된 프래그먼트만 강도 가중 평균.
        """
        abundances = []
        errors = []
        for fi, err in frag_errors.items():
            abundances.append(frag_intensities[fi])
            errors.append(err)
        return self._calculate_weighted_error(abundances, errors)

    @staticmethod
    def _calculate_weighted_error(abundances: list[float], errors: list[float]) -> float:
        """C# CalculateWeightedError 이식."""
        total_weight = 0.0
        total_value = 0.0
        for i in range(len(abundances)):
            if abundances[i] >= 0.0 and errors[i] >= 0.0:
                total_weight += abundances[i]
                total_value += abundances[i] * errors[i]
        if total_weight <= 0.0:
            return 0.0
        return total_value / total_weight

    # ─── 내부 메서드 ────────────────────────────────────

    def _add_one_valid_composition(self, mol: Molecule) -> bool:
        """하나의 후보를 패턴 매칭하여 결과에 추가"""
        mp = MoleculePattern(self._charge)
        mp.calculate_pattern(mol, self._idc)

        # 패턴 피크 매핑
        theor_mz = mp.pattern_mz
        theor_int = mp.pattern_rel_intensities

        num_fit = min(len(self._pattern_mz), len(theor_mz))
        error_ppm = [0.0] * num_fit
        error_mda = [0.0] * num_fit
        rel_int_error = [0.0] * num_fit
        pattern_peaks = [-1] * num_fit

        # 패턴 배율 계산
        ref_idx = self._get_closest_index(
            self._pattern_mz[self._norm_ref_peak], theor_mz, theor_int
        )
        pattern_mult = 100.0 / theor_int[ref_idx] if ref_idx >= 0 else 1.0

        for i in range(num_fit):
            idx = self._get_closest_index(self._pattern_mz[i], theor_mz, theor_int)
            pattern_peaks[i] = idx
            if idx >= 0:
                error_ppm[i] = get_mass_error(theor_mz[idx], self._pattern_mz[i], in_ppm=True)
                error_mda[i] = get_mass_error(theor_mz[idx], self._pattern_mz[i], in_ppm=False)
                rel_int_error[i] = theor_int[idx] * pattern_mult - self._pattern_int[i]

        # 유효성 검사
        if not self._valid_pattern_peaks(pattern_peaks):
            return False
        if not self._all_pattern_peaks_present(pattern_peaks, mp):
            return False
        if not self._data_within_tolerance(error_ppm, self._mass_tol_ppm):
            return False
        if not self._data_within_tolerance(rel_int_error, self._int_tol):
            return False

        # 오차 계산
        cluster_error = mp.get_cluster_error(
            self._pattern_mz, self._pattern_int,
            self._use_peak, self._pattern_weighting,
        )
        rms_error = mp.get_rms_error(
            self._pattern_mz, self._use_peak,
            self._pattern_weighting, adjust_mono_mz=False,
        )
        rms_w_int = mp.get_rms_error(
            self._pattern_mz, self._use_peak,
            self._pattern_weighting, adjust_mono_mz=True,
        )

        rel_diff = (get_ratio(rel_int_error[1], rel_int_error[2])
                     if len(rel_int_error) > 2 else 0.0)

        # 결과 저장
        result = CompositionResult(
            composition=mol.composition,
            monoisotopic_mass=mol.monoisotopic_mass,
            mono_mz=mol.mono_mass_to_charge,
            error_mda=error_mda[0],
            error_ppm=error_ppm[0],
            abs_error_ppm=abs(error_ppm[0]),
            rdb=mol.rdb,
            is_even_electron=mol.is_even_electron,
            intensity_cluster_error=cluster_error,
            rms_error=rms_error,
            rms_w_int_error=rms_w_int,
            mass_cluster_error=abs(rms_error) + 100.0 * cluster_error,
            rel_diff_peaks_1and2=rel_diff,
        )

        # 피크 상세
        norm_factor = 100.0 / mp.pattern_rel_intensities[self._norm_ref_peak] if mp.pattern_rel_intensities else 1.0
        for i in range(num_fit):
            if pattern_peaks[i] >= 0:
                result.peak_details.append(PeakDetail(
                    isotope_index=i,
                    theoretical_mz=theor_mz[pattern_peaks[i]],
                    theoretical_intensity=theor_int[pattern_peaks[i]] * norm_factor,
                    is_used=self._use_peak[i] or i == 0,
                    weighting=self._pattern_weighting[i],
                    error_ppm=error_ppm[i],
                    rel_int_error=rel_int_error[i],
                ))

        self._results.append(result)
        self._potential_molecules.append(mol)
        return True

    def _get_closest_index(
        self, target_mz: float, mz_array: list[float], int_array: list[float]
    ) -> int:
        """실측 m/z에 가장 가까운 이론 피크 인덱스 (2 Da 이내, 1% 이상)"""
        best_idx = -1
        best_diff = 2.0
        for i in range(len(mz_array)):
            diff = abs(target_mz - mz_array[i])
            if diff < best_diff and int_array[i] > 1.0:
                best_idx = i
                best_diff = diff
        return best_idx

    def _valid_pattern_peaks(self, peaks: list[int]) -> bool:
        seen = set()
        for p in peaks:
            if p < 0 or p in seen:
                return False
            seen.add(p)
        return True

    def _all_pattern_peaks_present(self, peaks: list[int], mp: MoleculePattern) -> bool:
        theor_mz = mp.pattern_mz
        for i in range(len(self._pattern_mz)):
            if not self._use_peak[i]:
                continue
            if (self._pattern_weighting[i] > 0 and
                    self._pattern_int[i] / self._pattern_weighting[i] <= self._int_tol):
                continue
            exp_mz = self._pattern_mz[i]
            tol_da = convert_to_da_error(self._mass_tol_ppm, exp_mz)
            found = any(abs(theor_mz[p] - exp_mz) < tol_da for p in peaks if p >= 0)
            if not found:
                return False
        return True

    def _data_within_tolerance(self, data: list[float], tolerance: float) -> bool:
        for i in range(len(data)):
            if self._use_peak[i]:
                if abs(data[i]) > tolerance / self._pattern_weighting[i]:
                    return False
        return True

    # ─── 순위 계산 ──────────────────────────────────────

    def _update_composition_ranking(self):
        """4가지 오차 기반 순위 + Overall"""
        n_active = sum(1 for u in self._use_peak if u)

        if n_active > 1:
            self._rank_by("intensity_cluster_error", "int_cluster_order", 1)
            self._rank_by("rms_error", "rms_error_order", 1)
            self._rank_by("mass_cluster_error", "mass_cluster_order", 1)
            self._rank_by("rms_w_int_error", "rms_w_int_order", 1)
        else:
            self._rank_by("abs_error_ppm", "rms_error_order", 4)

        # Overall = 4개 순위 합
        for r in self._results:
            r.overall_order = (
                r.int_cluster_order + r.rms_error_order +
                r.mass_cluster_order + r.rms_w_int_order
            )

    def _update_msms_ranking(self):
        """MS/MS 순위"""
        self._rank_by("msms_error_ppm", "msms_order", 1)

    def _rank_by(self, sort_field: str, target_field: str, multiplier: int):
        """특정 필드 기준 순위 매기기"""
        sorted_results = sorted(self._results, key=lambda r: getattr(r, sort_field))
        prev_val = None
        rank = 0
        for i, r in enumerate(sorted_results):
            val = getattr(r, sort_field)
            if prev_val is None or val > prev_val:
                rank = i + 1
                prev_val = val
            setattr(r, target_field, rank * multiplier)

    def _set_dbe_gain_from_water_losses(
        self, mass_tolerance: float = 5.0, mass_tol_in_ppm: bool = True,
    ):
        """
        SCIEX SetDBEGainfromWaterLosses 이식.
        프래그먼트 중성 질량 기반 연속 물 손실 체인 → _dbe_gain 동적 설정.
        C# 원본: linked-array + GetContinuousCount 방식.

        교차검증 수정:
        - C# 역순 반복 (num = Length-1 → 0) 반영
        - neutral[num2] >= neutral[num] 가드 조건 추가
        - tolerance는 num2 (더 큰 질량) 기준 계산
        """
        if self._fragment_mz is None or len(self._fragment_mz) < 2:
            self.__dbe_gain = self.DEFAULT_DBE_GAIN
            return

        n = len(self._fragment_mz)
        water_mass = Molecule("H2O").monoisotopic_mass  # ~18.010565

        # m/z → 중성 질량 변환 (C# 원본과 동일)
        neutral = [0.0] * n
        for i in range(n):
            neutral[i] = self._fragment_mz[i] * abs(self._fragment_charge[i])

        # linked-array: chain[num2] = num 이면 num2는 num에서 H2O 손실
        # C# 원본: 역순 반복 + neutral[num2] >= neutral[num] 가드
        chain = [-1] * n
        for num in range(n - 1, -1, -1):
            for num2 in range(n - 1, -1, -1):
                if num == num2:
                    continue
                # C# 원본: if (num != num2 && !(array2[num2] < array2[num]))
                if neutral[num2] < neutral[num]:
                    continue
                # C# 원본: ConvertTomDaError(massTolerance, array2[num2])
                if mass_tol_in_ppm:
                    tol = neutral[num2] * mass_tolerance / 1e6
                else:
                    tol = mass_tolerance / 1000.0
                diff = abs(neutral[num2] - neutral[num] - water_mass)
                if diff <= tol:
                    chain[num2] = num

        # GetContinuousCount: 연속 체인 길이 계산
        max_continuous = 0
        for i in range(n):
            count = 0
            idx = i
            visited = set()
            while chain[idx] >= 0 and chain[idx] not in visited:
                visited.add(idx)
                count += 1
                idx = chain[idx]
            max_continuous = max(max_continuous, count)

        self.__dbe_gain = max(self.DEFAULT_DBE_GAIN, max_continuous)

    @property
    def _dbe_gain(self) -> int:
        if hasattr(self, '_FormulaFinderPipeline__dbe_gain'):
            return self.__dbe_gain
        return self.DEFAULT_DBE_GAIN

    def _has_complement_fragment(self, index: int) -> bool:
        """
        SCIEX HasComplementFragment 이식.
        프래그먼트가 ±H 질량 범위 내에 보완 프래그먼트를 갖는지 확인.
        C# 원본: Molecule("H") 의 MonoisotopicMass 사용, tolerance=0.001 Da.
        """
        if self._fragment_mz is None or index >= len(self._fragment_mz):
            return False
        h_mass = Molecule("H").monoisotopic_mass  # ~1.00782503223
        tol = 0.001
        target_mz = self._fragment_mz[index]
        for j, mz in enumerate(self._fragment_mz):
            if j == index:
                continue
            if abs(mz - target_mz - h_mass) < tol or abs(mz - target_mz + h_mass) < tol:
                return True
        return False

    def _molecule_valid_for_compositions(self, frag_mol: Molecule, consider_odd_el: bool) -> bool:
        """
        SCIEX MoleculeValidForCompositions 이식.
        프리필터: 홀수전자 + 할로겐 예외 확인, 부모 포함 + RDB 체크 (dbeGain, +0.5 없음).
        """
        # 홀수전자인데 허용 안 됨 → 할로겐 예외 확인
        if not consider_odd_el and not frag_mol.is_even_electron and not self._has_halogen(frag_mol):
            return False

        # 어떤 부모 분자라도 포함하고 RDB 조건 만족하면 통과
        for mol in self._potential_molecules:
            if (mol.contains(frag_mol) and
                    mol.rdb + self._dbe_gain >= frag_mol.rdb and
                    (consider_odd_el or frag_mol.is_even_electron or self._has_halogen(frag_mol))):
                return True
        return False

    @staticmethod
    def _has_halogen(mol: Molecule) -> bool:
        """분자가 할로겐(F, Cl, Br, I)을 포함하는지 확인"""
        return contains_halogen(mol.element_counts)
