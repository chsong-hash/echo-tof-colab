"""Isotope 패턴 타겟 검증 모듈.

Formula Finder의 MoleculePattern을 활용하여,
예측된 특정 분자식이 실측 스펙트럼에 존재하는지 확인한다.

De novo 식별(조합 폭발 문제)이 아닌 타겟 검증이므로
MW에 관계없이 동일하게 작동한다.
"""
from __future__ import annotations
import math
from typing import Optional, Dict, List
from .config import logger

# echo_tof core 모듈
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from echo_tof.molecule import Molecule
from echo_tof.isotope_calc import IsotopicDistributionCalculator
from echo_tof.pattern import MoleculePattern
from echo_tof.calculations import get_mass_error, normalize_array_to_index, largest_selected_in_array


# 매칭 등급 기준
MATCH_THRESHOLDS = {
    "excellent": {"cluster_error": 5.0, "rms_ppm": 3.0},
    "good":      {"cluster_error": 15.0, "rms_ppm": 5.0},
    "fair":      {"cluster_error": 30.0, "rms_ppm": 10.0},
}


class IsotopeVerifier:
    """분자식 기반 동위원소 패턴 타겟 검증기."""

    def __init__(self, ppm_tolerance: float = 10.0):
        self._idc = IsotopicDistributionCalculator(
            ppm_tolerance=True, tolerance=50.0
        )
        self._ppm_tolerance = ppm_tolerance

    def verify(
        self,
        formula: str,
        charge: int,
        mz_array: list[float],
        int_array: list[float],
        use_peak: Optional[list[bool]] = None,
        is_ion_formula: bool = False,
    ) -> Dict:
        """특정 분자식의 isotope 패턴이 실측 데이터와 일치하는지 검증.

        Parameters
        ----------
        formula : 중성 분자식 (e.g., "C20H25N3O4")
        charge : 전하 상태 (양이온: +1, +2, ...)
        mz_array : 실측 m/z 값 (monoisotopic부터 순서대로)
        int_array : 실측 강도 값
        use_peak : 각 피크 사용 여부 (None이면 강도>0인 것만)
        is_ion_formula : True면 formula가 이미 이온 분자식 ([M+H]+의 경우 H 추가됨)

        Returns
        -------
        dict
        """
        n = len(mz_array)
        if n == 0:
            return self._fail_result(formula, "실측 데이터 없음")

        if use_peak is None:
            use_peak = [v > 0 for v in int_array]

        # ── 핵심: 강도를 최대 피크 기준 0~100%로 정규화 ──
        # MoleculePattern.get_cluster_error()가 % 단위를 기대함
        ref_idx = largest_selected_in_array(int_array, use_peak)
        norm_int = normalize_array_to_index(int_array, ref_idx, in_percent=True)

        # S/N 보정 (모두 1.0 — DI에서는 별도 보정 불필요)
        sn_correction = [1.0] * n

        try:
            # [핵심] ESI 양이온: [M+nH]ⁿ⁺ → 중성 분자식에 H를 n개 추가
            # Molecule(charge=n)은 전자 제거(radical cation)를 하므로,
            # 프로톤 추가를 직접 처리해야 정확한 isotope 패턴이 나온다.
            if not is_ion_formula and charge > 0:
                ion_formula = self._add_protons(formula, charge)
            elif not is_ion_formula and charge < 0:
                ion_formula = self._remove_protons(formula, abs(charge))
            else:
                ion_formula = formula
            mol = Molecule(composition=ion_formula, charge=charge)
        except Exception as e:
            return self._fail_result(formula, f"분자식 파싱 오류: {e}")

        mp = MoleculePattern(charge)
        mp.calculate_pattern(mol, self._idc)

        theor_mz = mp.pattern_mz
        theor_int = mp.pattern_rel_intensities

        if not theor_mz:
            return self._fail_result(formula, "이론 패턴 생성 실패")

        # mono 피크 질량 오차
        mass_error_ppm = get_mass_error(theor_mz[0], mz_array[0], in_ppm=True)
        if abs(mass_error_ppm) > self._ppm_tolerance:
            return self._fail_result(
                formula,
                f"mono 질량 오차 {mass_error_ppm:.1f} ppm > 허용 {self._ppm_tolerance} ppm",
            )

        # 클러스터 에러 (강도 패턴 매칭) — 정규화된 강도 사용
        cluster_error = mp.get_cluster_error(
            mz_array, norm_int, use_peak, sn_correction
        )

        # RMS 에러 (질량 정확도)
        rms_error = mp.get_rms_error(
            mz_array, use_peak, sn_correction, adjust_mono_mz=False
        )

        # 매칭된 피크 수
        n_matched = sum(1 for u, v in zip(use_peak, int_array) if u and v > 0)

        # 등급 판정
        grade = "poor"
        for g in ("excellent", "good", "fair"):
            th = MATCH_THRESHOLDS[g]
            ce = cluster_error if not math.isnan(cluster_error) else 999
            re = abs(rms_error) if not math.isnan(rms_error) else 999
            if ce <= th["cluster_error"] and re <= th["rms_ppm"]:
                grade = g
                break

        matched = grade in ("excellent", "good", "fair")

        # 이론 패턴 정보
        theoretical = [
            {"mz": theor_mz[i], "rel_intensity": theor_int[i]}
            for i in range(len(theor_mz))
        ]

        details = (
            f"{formula} (charge={charge}): "
            f"CE={cluster_error:.2f}, RMS={rms_error:.2f} ppm, "
            f"mono_err={mass_error_ppm:.2f} ppm, "
            f"peaks={n_matched}/{n} → {grade}"
        )
        logger.info(details)

        return {
            "formula": formula,
            "matched": matched,
            "grade": grade,
            "cluster_error": round(cluster_error, 4) if not math.isnan(cluster_error) else None,
            "rms_error_ppm": round(rms_error, 4) if not math.isnan(rms_error) else None,
            "mass_error_ppm": round(mass_error_ppm, 4),
            "n_matched_peaks": n_matched,
            "n_total_peaks": n,
            "theoretical_pattern": theoretical,
            "details": details,
        }

    def verify_mw_only(
        self,
        mw: float,
        observed_mz: float,
        charge: int = 1,
    ) -> Dict:
        """분자식 없이 MW만으로 질량 매칭 (isotope 검증 불가).

        MW만 알 때는 exact mass 매칭만 수행하고,
        isotope 패턴 검증은 건너뛴다.
        confidence가 낮게 설정됨.
        """
        # [M+H]+: (MW + proton_mass) / |charge|
        proton_mass = 1.00728
        expected_mz = (mw + abs(charge) * proton_mass) / abs(charge) if charge != 0 else mw
        error_ppm = get_mass_error(expected_mz, observed_mz, in_ppm=True)

        matched = abs(error_ppm) <= self._ppm_tolerance

        return {
            "formula": None,
            "matched": matched,
            "grade": "mass_only" if matched else "poor",
            "cluster_error": None,
            "rms_error_ppm": None,
            "mass_error_ppm": round(error_ppm, 4),
            "n_matched_peaks": 1 if matched else 0,
            "n_total_peaks": 1,
            "theoretical_pattern": [],
            "details": (
                f"MW={mw:.4f}, expected m/z={expected_mz:.4f}, "
                f"observed={observed_mz:.4f}, err={error_ppm:.2f} ppm "
                f"→ {'매칭' if matched else '불일치'} (isotope 검증 불가: 분자식 없음)"
            ),
            "_warning": "분자식 없이 MW만으로 매칭. isotope 패턴 검증 미수행.",
        }

    @staticmethod
    def _add_protons(formula: str, n: int) -> str:
        """중성 분자식에 H를 n개 추가하여 [M+nH]ⁿ⁺ 이온 분자식 생성."""
        mol = Molecule(composition=formula, charge=0)
        h_mol = Molecule(composition=f"H{n}" if n > 1 else "H", charge=0)
        ion_mol = mol.add(h_mol)
        return ion_mol.composition

    @staticmethod
    def _remove_protons(formula: str, n: int) -> str:
        """중성 분자식에서 H를 n개 제거하여 [M-nH]ⁿ⁻ 이온 분자식 생성."""
        mol = Molecule(composition=formula, charge=0)
        h_mol = Molecule(composition=f"H{n}" if n > 1 else "H", charge=0)
        ion_mol = mol.subtract(h_mol)
        return ion_mol.composition

    @staticmethod
    def _fail_result(formula: str, reason: str) -> Dict:
        return {
            "formula": formula,
            "matched": False,
            "grade": "fail",
            "cluster_error": None,
            "rms_error_ppm": None,
            "mass_error_ppm": None,
            "n_matched_peaks": 0,
            "n_total_peaks": 0,
            "theoretical_pattern": [],
            "details": reason,
        }
