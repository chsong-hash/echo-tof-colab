"""
TOF 질량 재보정 — SCIEX Sciex.GetFormula.MassAdjuster 이식.

모델: √m = A × (t - t₀)
2-point calibration으로 A, t₀ 재계산.
"""

from __future__ import annotations
import math


class MassAdjuster:
    """
    TOF 질량 재보정.

    사용법:
        adj = MassAdjuster(is_tof=True, an_a=1000.0, a_t0=0.1)
        adj.update_cal_parameters(ref1, ref2, meas1, meas2)
        recalibrated = adj.get_recal_mass(exp_mass)
    """

    def __init__(self, is_tof: bool = True, an_a: float = 0.0, a_t0: float = 0.0):
        self._is_tof = is_tof
        self._an_a = an_a
        self._a_t0 = a_t0
        self._an_a_new = 0.0
        self._a_t0_new = 0.0

    @staticmethod
    def _get_t_exp(mass: float, a: float, t0: float) -> float:
        """질량 → 비행시간 역산: t = √m / A + t₀"""
        if a == 0:
            return 0.0
        return math.sqrt(mass) / a + t0

    @staticmethod
    def _get_a(mass1: float, mass2: float, t1: float, t2: float) -> float:
        """두 점에서 A 계산: A = (√m₁ - √m₂) / (t₁ - t₂)"""
        if t1 == t2:
            return 0.0
        return (math.sqrt(mass1) - math.sqrt(mass2)) / (t1 - t2)

    @staticmethod
    def _get_t0(t1: float, mass: float, a: float) -> float:
        """t₀ 계산: t₀ = t₁ - √m / A"""
        if a == 0:
            return 0.0
        return t1 - math.sqrt(mass) / a

    def update_cal_parameters(
        self,
        ref_mass1: float,
        ref_mass2: float,
        meas_mass1: float,
        meas_mass2: float,
    ) -> bool:
        """
        2-point calibration으로 새 파라미터 계산.
        ref: 참조(이론) 질량, meas: 실측 질량.
        """
        t1 = self._get_t_exp(meas_mass1, self._an_a, self._a_t0)
        t2 = self._get_t_exp(meas_mass2, self._an_a, self._a_t0)

        if t1 == 0 or t2 == 0:
            return False

        a_new = self._get_a(ref_mass1, ref_mass2, t1, t2)
        t0_new = self._get_t0(t1, ref_mass1, a_new)

        self._an_a_new = a_new
        self._a_t0_new = t0_new
        return True

    def get_recal_mass(self, exp_mass: float) -> float:
        """
        실측 질량을 재보정.
        재보정 모델: m' = [A_new × (t_exp - t₀_new)]²
        """
        if self._a_t0_new == 0.0 and self._an_a_new == 0.0:
            return exp_mass

        t_exp = self._get_t_exp(exp_mass, self._an_a, self._a_t0)
        val = self._an_a_new * (t_exp - self._a_t0_new)
        return val * val
