"""피크 분류기: Known / Inferred / Unknown.

수율 산출 파이프라인의 핵심 모듈.
TIC에서 검출된 모든 피크를 세 카테고리로 분류하고,
"설명 가능 비율"을 신뢰도 지표로 산출한다.

분류 기준:
  Known    — SM, Product, AI 예측 부산물, 용매, 매트릭스 (직접 매칭)
  Inferred — Δm/z 패턴으로 출발물질과의 관계를 추론할 수 있는 미지 피크
  Unknown  — 위 두 가지로 설명 불가능한 피크

기존 echo_tof 모듈의 Formula Finder를 import하여 활용.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from .config import logger, PipelineConfig
from .neutral_loss_db import DELTA_MZ_PATTERNS
from .mz_predict import predict_mz, H


class PeakClassification:
    """단일 피크의 분류 결과."""

    def __init__(self, mz: float, intensity: float, area: float = 0.0):
        self.mz = mz
        self.intensity = intensity
        self.area = area
        self.category = "Unknown"  # Known | Inferred | Unknown
        self.label = ""            # e.g., "SM", "Product", "Proto-dehalogenation"
        self.match_type = ""       # e.g., "exact", "adduct", "delta_mz"
        self.confidence = 0.0      # 0.0 ~ 1.0
        self.details = {}

    def __repr__(self):
        return (f"Peak(m/z={self.mz:.4f}, {self.category}: {self.label}, "
                f"conf={self.confidence:.2f})")


class PeakClassifier:
    """TIC 피크 분류 엔진."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.mz_tolerance = config.mz_tolerance

    def classify_peaks(
        self,
        peaks: List[Dict],
        sm_mw: Optional[float] = None,
        product_mw: Optional[float] = None,
        byproduct_mws: Optional[List[Dict]] = None,
        solvent_mzs: Optional[List[float]] = None,
        fragment_mzs: Optional[List[float]] = None,
    ) -> List[PeakClassification]:
        """모든 피크를 분류하고 결과 리스트 반환.

        Parameters
        ----------
        peaks : list of dict
            각 피크: {"mz": float, "intensity": float, "area": float, ...}
        sm_mw : float, optional
            출발물질 MW
        product_mw : float, optional
            목적 생성물 MW
        byproduct_mws : list of dict, optional
            예측 부산물: [{"name": str, "mw": float, "mz_mh": float}, ...]
        solvent_mzs : list of float, optional
            알려진 용매/매트릭스 m/z 목록
        fragment_mzs : list of float, optional
            fragmentation engine에서 예측된 fragment m/z 목록
        """
        results = []

        # 1) Known m/z 목록 구축
        known_targets = self._build_known_targets(
            sm_mw, product_mw, byproduct_mws, solvent_mzs, fragment_mzs
        )

        # 2) SM의 [M+H]+ m/z (Δm/z 추론용)
        sm_mz = sm_mw + H if sm_mw else None
        product_mz = product_mw + H if product_mw else None

        for peak in peaks:
            mz = peak.get("mz", 0.0)
            intensity = peak.get("intensity", peak.get("apex_intensity", 0.0))
            area = peak.get("area", 0.0)

            classification = PeakClassification(mz, intensity, area)

            # Step A: Known 매칭 (정확 m/z)
            matched = self._match_known(mz, known_targets)
            if matched:
                classification.category = "Known"
                classification.label = matched["label"]
                classification.match_type = matched["match_type"]
                classification.confidence = matched["confidence"]
                classification.details = matched
                results.append(classification)
                continue

            # Step B: Δm/z 패턴 추론 (SM 또는 Product 기준)
            inferred = self._infer_by_delta_mz(mz, sm_mz, product_mz)
            if inferred:
                classification.category = "Inferred"
                classification.label = inferred["label"]
                classification.match_type = "delta_mz"
                classification.confidence = inferred["confidence"]
                classification.details = inferred
                results.append(classification)
                continue

            # Step C: Unknown
            classification.category = "Unknown"
            classification.confidence = 0.0
            results.append(classification)

        return results

    def compute_reliability(
        self,
        classifications: List[PeakClassification],
    ) -> Dict:
        """분류 결과로부터 수율 계산 신뢰도 지표를 산출.

        Returns
        -------
        dict
            total_area, known_area, inferred_area, unknown_area,
            explained_ratio (%), unexplained_ratio (%),
            is_reliable (bool), summary (str)
        """
        total_area = sum(c.area for c in classifications)
        known_area = sum(c.area for c in classifications if c.category == "Known")
        inferred_area = sum(c.area for c in classifications if c.category == "Inferred")
        unknown_area = sum(c.area for c in classifications if c.category == "Unknown")

        if total_area == 0:
            return {
                "total_area": 0, "known_area": 0, "inferred_area": 0,
                "unknown_area": 0, "explained_ratio": 0.0,
                "unexplained_ratio": 100.0, "is_reliable": False,
                "summary": "피크 면적 합계가 0",
            }

        explained = (known_area + inferred_area) / total_area * 100
        unexplained = unknown_area / total_area * 100
        is_reliable = unexplained < self.config.unexplained_threshold

        n_known = sum(1 for c in classifications if c.category == "Known")
        n_inferred = sum(1 for c in classifications if c.category == "Inferred")
        n_unknown = sum(1 for c in classifications if c.category == "Unknown")

        summary = (
            f"피크 {len(classifications)}개: "
            f"Known {n_known}, Inferred {n_inferred}, Unknown {n_unknown} | "
            f"설명 가능 {explained:.1f}%, 설명 불가 {unexplained:.1f}%"
        )
        if not is_reliable:
            summary += f" → 경고: 설명 불가 비율 {unexplained:.1f}% > {self.config.unexplained_threshold}%"

        logger.info(summary)

        return {
            "total_area": total_area,
            "known_area": known_area,
            "inferred_area": inferred_area,
            "unknown_area": unknown_area,
            "explained_ratio": round(explained, 2),
            "unexplained_ratio": round(unexplained, 2),
            "is_reliable": is_reliable,
            "n_known": n_known,
            "n_inferred": n_inferred,
            "n_unknown": n_unknown,
            "summary": summary,
        }

    # ─── 내부 메서드 ──────────────────────────────────────

    def _build_known_targets(
        self, sm_mw, product_mw, byproduct_mws, solvent_mzs, fragment_mzs,
    ) -> List[Dict]:
        """Known 매칭 대상 m/z 목록을 구축."""
        targets = []

        # SM의 모든 adduct
        if sm_mw is not None:
            for adduct in predict_mz(sm_mw, "positive"):
                targets.append({
                    "mz": adduct["mz"], "label": f"SM ({adduct['adduct']})",
                    "match_type": "adduct", "confidence": 0.95,
                })

        # Product의 모든 adduct
        if product_mw is not None:
            for adduct in predict_mz(product_mw, "positive"):
                targets.append({
                    "mz": adduct["mz"], "label": f"Product ({adduct['adduct']})",
                    "match_type": "adduct", "confidence": 0.95,
                })

        # 예측 부산물
        if byproduct_mws:
            for bp in byproduct_mws:
                targets.append({
                    "mz": bp["mz_mh"], "label": bp["name"],
                    "match_type": "predicted_byproduct",
                    "confidence": 0.7 if bp.get("likelihood") == "high" else 0.5,
                })

        # 용매/매트릭스
        if solvent_mzs:
            for mz in solvent_mzs:
                targets.append({
                    "mz": mz, "label": "Solvent/Matrix",
                    "match_type": "solvent", "confidence": 0.9,
                })

        # fragmentation engine 예측
        if fragment_mzs:
            for mz in fragment_mzs:
                targets.append({
                    "mz": mz, "label": "Predicted Fragment",
                    "match_type": "fragment", "confidence": 0.6,
                })

        return targets

    def _match_known(self, mz: float, targets: List[Dict]) -> Optional[Dict]:
        """m/z가 known 목록의 어떤 항목과 매칭되는지 확인."""
        best = None
        best_error = float("inf")
        for t in targets:
            error = abs(mz - t["mz"])
            if error <= self.mz_tolerance and error < best_error:
                best = t
                best_error = error
        if best:
            return {**best, "error_da": round(best_error, 6)}
        return None

    def _infer_by_delta_mz(
        self, mz: float, sm_mz: Optional[float], product_mz: Optional[float],
    ) -> Optional[Dict]:
        """Δm/z 패턴으로 미지 피크의 관계를 추론."""
        references = []
        if sm_mz:
            references.append(("SM", sm_mz))
        if product_mz:
            references.append(("Product", product_mz))

        for ref_name, ref_mz in references:
            delta = mz - ref_mz
            for expected_delta, (name, description) in DELTA_MZ_PATTERNS.items():
                if abs(abs(delta) - expected_delta) <= self.mz_tolerance * 2:
                    sign = "+" if delta > 0 else "-"
                    return {
                        "label": f"{ref_name} {name}",
                        "description": description,
                        "ref_mz": ref_mz,
                        "delta_observed": round(delta, 4),
                        "delta_expected": expected_delta,
                        "confidence": 0.5,
                    }
        return None
