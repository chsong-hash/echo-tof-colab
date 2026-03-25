"""ECHO-TOF 통합 파이프라인.

Direct Infusion TOF-MS 수율 산출 파이프라인.
기존 echo_tof(Formula Finder)와 echo_tof_ext 모듈을 통합한다.

파이프라인 흐름:
  1. 스펙트럼 피크 검출 (DI 전용)
  2. 반응 예측 → 부산물 m/z 목록 생성
  3. 타겟 검증: 각 예측 m/z의 존재 여부 + isotope 패턴 확인
  4. 전체 피크 분류 (Known / Inferred / Unknown)
  5. 수율 계산 (SM 소모 기반)
  6. 신뢰도 산출 (설명 불가 피크 비율)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

from .config import PipelineConfig, logger
from .di_spectrum import pick_peaks, extract_cluster_at_mz
from .isotope_verifier import IsotopeVerifier
from .mz_predict import predict_mz, H
from .reaction_predictor import predict_product_mw, predict_byproduct_mws
from .peak_classifier import PeakClassifier, PeakClassification
from .neutral_loss_db import DELTA_MZ_PATTERNS


# ═══════════════════════════════════════════════════════════════
# 데이터 구조
# ═══════════════════════════════════════════════════════════════

@dataclass
class CompoundTarget:
    """검증 대상 화합물."""
    name: str
    mw: Optional[float] = None
    formula: Optional[str] = None
    role: str = ""           # "SM", "Product", "Byproduct", "Solvent"
    likelihood: str = "high"
    # 채워지는 필드
    adducts: List[Dict] = field(default_factory=list)
    verified: bool = False
    verification_results: List[Dict] = field(default_factory=list)
    best_mz: Optional[float] = None
    best_intensity: float = 0.0
    best_area: float = 0.0


@dataclass
class PipelineResult:
    """파이프라인 전체 결과."""
    # 입력 요약
    n_spectrum_points: int = 0
    n_detected_peaks: int = 0

    # 타겟 검증
    targets: List[CompoundTarget] = field(default_factory=list)
    n_targets_verified: int = 0
    n_targets_total: int = 0

    # 피크 분류
    classifications: List[PeakClassification] = field(default_factory=list)

    # 수율
    conversion: Optional[float] = None     # SM 소모 전환율 (%)
    sm_before_area: float = 0.0
    sm_after_area: float = 0.0

    # 신뢰도
    reliability: Dict = field(default_factory=dict)

    # 경고/이슈
    warnings: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# 메인 파이프라인
# ═══════════════════════════════════════════════════════════════

class EchoPipeline:
    """ECHO-TOF Direct Infusion 수율 산출 파이프라인.

    사용법:
        pipeline = EchoPipeline(config)

        # 반응 전 스펙트럼으로 SM baseline 측정
        pipeline.set_before_spectrum(mz_before, int_before)

        # 반응 후 스펙트럼으로 수율 산출
        result = pipeline.run(
            mz_after, int_after,
            sm_formula="C10H12N2O",
            product_formula="C18H20N2O3",
            reaction_type="amide_coupling",
            reagent_mw=150.068,
        )
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._verifier = IsotopeVerifier(ppm_tolerance=self._get_ppm_tolerance())
        self._classifier = PeakClassifier(self.config)

        # 반응 전 SM 기준
        self._sm_before_area: Optional[float] = None
        self._sm_before_mz: Optional[float] = None

        # 용매/매트릭스 m/z (백그라운드 제외용)
        self._solvent_mzs: List[float] = []

        # ── 검증된 문제점 기록 ──
        self._issues: List[str] = []

    def set_solvent_mzs(self, mzs: List[float]):
        """알려진 용매/매트릭스 m/z 설정."""
        self._solvent_mzs = list(mzs)

    def set_before_spectrum(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        sm_formula: Optional[str] = None,
        sm_mw: Optional[float] = None,
        charge: int = 1,
    ):
        """반응 전 스펙트럼에서 SM 기준 면적 측정.

        Parameters
        ----------
        mz, intensity : 반응 전 스펙트럼
        sm_formula : SM 분자식 (isotope 검증용, 권장)
        sm_mw : SM MW (분자식 없을 때)
        charge : 전하 상태
        """
        sm_target_mz = self._get_target_mz(sm_formula, sm_mw, charge)
        if sm_target_mz is None:
            self._issues.append("SM m/z를 계산할 수 없음: formula 또는 mw 필요")
            return

        cluster = extract_cluster_at_mz(
            mz, intensity, sm_target_mz, charge,
            mz_tolerance_ppm=self._get_ppm_tolerance(),
        )

        if cluster is None:
            self._issues.append(
                f"반응 전 스펙트럼에서 SM 피크 미검출 (expected m/z={sm_target_mz:.4f})"
            )
            return

        self._sm_before_area = sum(cluster["int_array"])
        self._sm_before_mz = cluster["mono_mz"]
        logger.info(
            f"SM before: m/z={self._sm_before_mz:.4f}, "
            f"area={self._sm_before_area:.0f}"
        )

    def run(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        sm_formula: Optional[str] = None,
        sm_mw: Optional[float] = None,
        product_formula: Optional[str] = None,
        product_mw: Optional[float] = None,
        reaction_type: Optional[str] = None,
        reagent_mw: float = 0.0,
        leaving_group: str = "Br",
        charge: int = 1,
        extra_targets: Optional[List[CompoundTarget]] = None,
    ) -> PipelineResult:
        """파이프라인 실행.

        Parameters
        ----------
        mz, intensity : 반응 후 스펙트럼
        sm_formula : SM 분자식 (e.g., "C10H12N2O")
        sm_mw : SM MW (분자식 없을 때 대안)
        product_formula : 목적 생성물 분자식
        product_mw : 목적 생성물 MW (분자식 없을 때)
        reaction_type : 반응 유형 (reaction_predictor 키)
        reagent_mw : 시약 MW
        leaving_group : 이탈기
        charge : 전하 상태
        extra_targets : 추가 검증 대상
        """
        mz = np.asarray(mz, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        result = PipelineResult(n_spectrum_points=len(mz))
        result.issues = list(self._issues)

        # ────────────────────────────────────────────
        # STEP 1: 피크 검출
        # ────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 1: DI 스펙트럼 피크 검출")
        detected_peaks = pick_peaks(mz, intensity)
        result.n_detected_peaks = len(detected_peaks)

        if len(detected_peaks) == 0:
            result.warnings.append("피크가 검출되지 않음")
            return result

        # ────────────────────────────────────────────
        # STEP 2: 타겟 목록 생성
        # ────────────────────────────────────────────
        logger.info("STEP 2: 검증 타겟 목록 생성")
        targets = self._build_targets(
            sm_formula, sm_mw, product_formula, product_mw,
            reaction_type, reagent_mw, leaving_group, charge,
            extra_targets,
        )
        result.targets = targets
        result.n_targets_total = len(targets)
        logger.info(f"  타겟 {len(targets)}개 생성")

        # ────────────────────────────────────────────
        # STEP 3: 타겟 검증 (핵심)
        # ────────────────────────────────────────────
        logger.info("STEP 3: 타겟 검증 (mass + isotope pattern)")
        for target in targets:
            self._verify_target(target, mz, intensity, charge)

        result.n_targets_verified = sum(1 for t in targets if t.verified)
        logger.info(
            f"  검증 결과: {result.n_targets_verified}/{result.n_targets_total} 확인됨"
        )

        # ────────────────────────────────────────────
        # STEP 4: 전체 피크 분류
        # ────────────────────────────────────────────
        logger.info("STEP 4: 전체 피크 분류 (Known/Inferred/Unknown)")

        # S/N + 상대강도 기준으로 유의미한 피크만 분류
        # [수정] 배경 노이즈 피크가 Unknown으로 과다 카운트되는 문제 해결
        sn_threshold = self.config.sn_threshold  # default 3.0
        if detected_peaks:
            base_peak_int = detected_peaks[0]["intensity"]  # 이미 강도순 정렬됨
            min_abs_int = base_peak_int * 0.5 / 100.0  # base peak의 0.5%
        else:
            min_abs_int = 0

        significant_peaks = [
            p for p in detected_peaks
            if p.get("sn", 0) >= sn_threshold and p["intensity"] >= min_abs_int
        ]
        logger.info(
            f"  유의미 피크: {len(significant_peaks)}/{len(detected_peaks)} "
            f"(S/N >= {sn_threshold}, >= {min_abs_int:.0f} counts)"
        )

        # 검증된 타겟의 fragment m/z 수집
        verified_fragment_mzs = []
        for t in targets:
            if t.verified and t.best_mz:
                verified_fragment_mzs.append(t.best_mz)

        byproduct_mws = []
        for t in targets:
            if t.role == "Byproduct" and t.mw:
                byproduct_mws.append({
                    "name": t.name,
                    "mw": t.mw,
                    "mz_mh": t.mw + H,
                    "likelihood": t.likelihood,
                })

        classifications = self._classifier.classify_peaks(
            peaks=significant_peaks,
            sm_mw=sm_mw or self._formula_to_mw(sm_formula),
            product_mw=product_mw or self._formula_to_mw(product_formula),
            byproduct_mws=byproduct_mws if byproduct_mws else None,
            solvent_mzs=self._solvent_mzs if self._solvent_mzs else None,
            fragment_mzs=verified_fragment_mzs if verified_fragment_mzs else None,
        )

        # 검증된 타겟 기반으로 분류 보강
        classifications = self._enhance_classifications(
            classifications, targets, charge
        )

        result.classifications = classifications

        # ────────────────────────────────────────────
        # STEP 5: 수율 계산 (SM 소모 기반)
        # ────────────────────────────────────────────
        logger.info("STEP 5: 수율 계산")

        sm_target = next((t for t in targets if t.role == "SM"), None)
        if sm_target and sm_target.best_area > 0:
            result.sm_after_area = sm_target.best_area

        if self._sm_before_area and self._sm_before_area > 0:
            result.sm_before_area = self._sm_before_area
            if result.sm_after_area >= 0:
                conversion = (1.0 - result.sm_after_area / self._sm_before_area) * 100
                conversion = max(0.0, min(100.0, conversion))
                result.conversion = round(conversion, 2)
                logger.info(f"  전환율: {result.conversion}%")
            else:
                result.conversion = 100.0
                logger.info("  SM 미검출 → 전환율 100%")
        else:
            result.warnings.append(
                "반응 전 SM 기준 없음: set_before_spectrum()을 먼저 호출하세요"
            )

        # ────────────────────────────────────────────
        # STEP 6: 신뢰도 산출
        # ────────────────────────────────────────────
        logger.info("STEP 6: 신뢰도 산출")
        result.reliability = self._classifier.compute_reliability(classifications)

        if not result.reliability.get("is_reliable", False):
            result.warnings.append(
                f"설명 불가 비율 {result.reliability.get('unexplained_ratio', 0):.1f}% "
                f"> 임계값 {self.config.unexplained_threshold}%"
            )

        # ── 이슈 검증 결과 추가 ──
        self._validate_results(result, targets, charge)

        logger.info("=" * 60)
        logger.info(f"파이프라인 완료: 전환율={result.conversion}%, "
                     f"신뢰도={result.reliability.get('explained_ratio', 0):.1f}%")
        if result.warnings:
            for w in result.warnings:
                logger.warning(f"  ⚠ {w}")

        return result

    # ═══════════════════════════════════════════════════════════
    # 내부 메서드
    # ═══════════════════════════════════════════════════════════

    def _get_ppm_tolerance(self) -> float:
        """config에서 ppm tolerance 계산.

        [문제점 발견] config.mz_tolerance가 Da 단위(0.005)인데,
        m/z에 따라 ppm이 달라짐. 기본값 10 ppm 사용.
        """
        # config.mz_tolerance는 Da 단위 → ppm 변환은 m/z 의존적
        # 보수적으로 10 ppm 기본값 사용
        return 10.0

    def _get_target_mz(
        self,
        formula: Optional[str],
        mw: Optional[float],
        charge: int,
    ) -> Optional[float]:
        """분자식 또는 MW에서 [M+nH]ⁿ⁺ target m/z 계산.

        [수정] Molecule(charge=1)은 전자 제거(radical cation)를 하지만,
        ESI-MS에서 실제 이온은 [M+H]⁺(프로톤 추가)이므로
        중성 MW + n×proton_mass 방식으로 계산한다.
        """
        neutral_mw = None

        if formula:
            try:
                from echo_tof.molecule import Molecule
                mol = Molecule(composition=formula, charge=0)  # 반드시 중성으로
                neutral_mw = mol.monoisotopic_mass
            except Exception:
                pass

        if neutral_mw is None and mw is not None:
            neutral_mw = mw

        if neutral_mw is None:
            return None

        if charge == 0:
            return neutral_mw
        # [M+nH]ⁿ⁺: (MW + n × proton_mass) / n
        return (neutral_mw + abs(charge) * H) / abs(charge)

    def _formula_to_mw(self, formula: Optional[str]) -> Optional[float]:
        """분자식 → 중성 MW."""
        if not formula:
            return None
        try:
            from echo_tof.molecule import Molecule
            mol = Molecule(composition=formula, charge=0)
            return mol.monoisotopic_mass
        except Exception:
            return None

    def _build_targets(
        self,
        sm_formula, sm_mw, product_formula, product_mw,
        reaction_type, reagent_mw, leaving_group, charge,
        extra_targets,
    ) -> List[CompoundTarget]:
        """모든 검증 대상 화합물 목록 생성."""
        targets = []

        # SM
        if sm_formula or sm_mw:
            sm = CompoundTarget(
                name="Starting Material",
                formula=sm_formula,
                mw=sm_mw or self._formula_to_mw(sm_formula),
                role="SM",
            )
            sm.adducts = predict_mz(sm.mw, "positive") if sm.mw else []
            targets.append(sm)

        # Product
        if product_formula or product_mw:
            prod = CompoundTarget(
                name="Product",
                formula=product_formula,
                mw=product_mw or self._formula_to_mw(product_formula),
                role="Product",
            )
            prod.adducts = predict_mz(prod.mw, "positive") if prod.mw else []
            targets.append(prod)
        elif reaction_type and sm_mw:
            # 반응 예측으로 product MW 추정
            pred_mw = predict_product_mw(sm_mw, reaction_type, reagent_mw, leaving_group)
            if pred_mw:
                prod = CompoundTarget(
                    name="Product (predicted)",
                    mw=pred_mw,
                    role="Product",
                )
                prod.adducts = predict_mz(pred_mw, "positive")
                targets.append(prod)

        # Byproducts (반응 예측)
        if reaction_type and (sm_mw or (sm_formula and self._formula_to_mw(sm_formula))):
            _sm_mw = sm_mw or self._formula_to_mw(sm_formula)
            byproducts = predict_byproduct_mws(
                _sm_mw, reaction_type, reagent_mw, leaving_group
            )
            for bp in byproducts:
                t = CompoundTarget(
                    name=bp["name"],
                    mw=bp["mw"],
                    role="Byproduct",
                    likelihood=bp.get("likelihood", "medium"),
                )
                t.adducts = predict_mz(bp["mw"], "positive")
                targets.append(t)

        # 추가 타겟
        if extra_targets:
            for et in extra_targets:
                if et.mw and not et.adducts:
                    et.adducts = predict_mz(et.mw, "positive")
                targets.append(et)

        return targets

    def _verify_target(
        self,
        target: CompoundTarget,
        mz: np.ndarray,
        intensity: np.ndarray,
        charge: int,
    ):
        """단일 타겟의 존재 여부를 검증.

        1) exact mass 매칭 → 2) isotope 패턴 검증 (분자식 있을 때)
        """
        target_mz = self._get_target_mz(target.formula, target.mw, charge)
        if target_mz is None:
            return

        # 스펙트럼에서 해당 m/z 주변 클러스터 추출
        cluster = extract_cluster_at_mz(
            mz, intensity, target_mz, charge,
            mz_tolerance_ppm=self._get_ppm_tolerance(),
        )

        if cluster is None:
            # [M+H]+ 에서 못 찾으면 다른 adduct도 확인
            for adduct_info in target.adducts:
                if adduct_info["adduct"] == "[M+H]+":
                    continue
                alt_cluster = extract_cluster_at_mz(
                    mz, intensity, adduct_info["mz"], adduct_info.get("charge", 1),
                    mz_tolerance_ppm=self._get_ppm_tolerance(),
                )
                if alt_cluster is not None:
                    cluster = alt_cluster
                    logger.info(
                        f"  {target.name}: [M+H]+ 미검출, "
                        f"{adduct_info['adduct']}에서 발견"
                    )
                    break

        if cluster is None:
            logger.info(f"  {target.name}: 미검출")
            return

        # Isotope 패턴 검증
        if target.formula:
            result = self._verifier.verify(
                formula=target.formula,
                charge=charge,
                mz_array=cluster["mz_array"],
                int_array=cluster["int_array"],
                use_peak=cluster["use_peak"],
            )
        else:
            # [문제점] 분자식 없음 → mass-only 매칭
            result = self._verifier.verify_mw_only(
                mw=target.mw,
                observed_mz=cluster["mono_mz"],
                charge=charge,
            )

        target.verification_results.append(result)
        target.verified = result["matched"]
        target.best_mz = cluster["mono_mz"]
        target.best_intensity = max(cluster["int_array"])
        target.best_area = sum(cluster["int_array"])

    def _enhance_classifications(
        self,
        classifications: List[PeakClassification],
        targets: List[CompoundTarget],
        charge: int,
    ) -> List[PeakClassification]:
        """검증된 타겟 결과로 기존 분류를 보강.

        1) isotope 검증 통과 → confidence 상향
        2) 검증된 화합물의 동위원소 피크(M+1, M+2 등)도 Known으로 마킹
        3) 검증된 화합물의 adduct 피크도 Known으로 마킹
        """
        ppm_tol = self._get_ppm_tolerance()

        for cls in classifications:
            if cls.category == "Known" and cls.confidence >= 0.9:
                continue

            for target in targets:
                if not target.verified or not target.best_mz:
                    continue

                tol_da = cls.mz * ppm_tol / 1e6

                # monoisotopic 매칭
                if abs(cls.mz - target.best_mz) <= tol_da:
                    self._set_verified_classification(cls, target, "mono")
                    break

                # isotope 피크: 이론 패턴 m/z 기반 매칭
                # [수정] 고정 spacing 대신 실제 이론 패턴 사용
                # → Br, Cl 등 비정상 spacing 정확히 처리
                matched_isotope = False
                if target.verification_results:
                    theor_pattern = target.verification_results[0].get(
                        "theoretical_pattern", []
                    )
                    # 이론 패턴의 M+1, M+2, ... 피크와 비교
                    for pi, tp in enumerate(theor_pattern):
                        if pi == 0:
                            continue  # mono는 위에서 처리
                        # 이론 m/z에 mono 오프셋 적용
                        # (이론 mono와 실측 mono의 차이를 보정)
                        theor_mono = theor_pattern[0]["mz"]
                        offset = target.best_mz - theor_mono
                        expected_iso = tp["mz"] + offset
                        if abs(cls.mz - expected_iso) <= tol_da:
                            self._set_verified_classification(
                                cls, target, f"isotope_M+{pi}"
                            )
                            matched_isotope = True
                            break
                if matched_isotope:
                    break

                # adduct 매칭 ([M+Na]+, [M+K]+ 등)
                for adduct_info in target.adducts:
                    if abs(cls.mz - adduct_info["mz"]) <= tol_da:
                        cls.category = "Known"
                        cls.label = f"{target.name} ({adduct_info['adduct']})"
                        cls.match_type = "adduct_verified"
                        cls.confidence = 0.85
                        break

        return classifications

    @staticmethod
    def _set_verified_classification(
        cls: PeakClassification,
        target: CompoundTarget,
        match_detail: str,
    ):
        """검증된 타겟으로 분류를 설정."""
        cls.category = "Known"
        cls.label = f"{target.name} ({match_detail})"
        cls.match_type = "isotope_verified"
        grade = (target.verification_results[0].get("grade", "")
                 if target.verification_results else "")
        if grade == "excellent":
            cls.confidence = 0.98
        elif grade == "good":
            cls.confidence = 0.90
        elif grade == "fair":
            cls.confidence = 0.75
        elif grade == "mass_only":
            cls.confidence = 0.60
            cls.match_type = "mass_only"
        else:
            cls.confidence = 0.70

    def _validate_results(
        self,
        result: PipelineResult,
        targets: List[CompoundTarget],
        charge: int,
    ):
        """결과 검증 및 이슈 보고.

        파이프라인 실행 후 발견 가능한 문제점들을 체크한다.
        """
        # 이슈 1: Product가 검출되지 않음
        product_target = next((t for t in targets if t.role == "Product"), None)
        if product_target and not product_target.verified:
            result.issues.append(
                f"목적 생성물 미검출: {product_target.name} "
                f"(expected m/z ~{self._get_target_mz(product_target.formula, product_target.mw, charge):.4f})"
            )

        # 이슈 2: SM이 완전히 소모됐는데 Product도 없음
        sm_target = next((t for t in targets if t.role == "SM"), None)
        if sm_target and not sm_target.verified and product_target and not product_target.verified:
            result.issues.append(
                "SM과 Product 모두 미검출 — 스펙트럼 품질 또는 입력 파라미터 확인 필요"
            )

        # 이슈 3: 분자식 없이 MW만 사용한 타겟
        mw_only_targets = [
            t for t in targets
            if t.formula is None and t.verified
        ]
        if mw_only_targets:
            names = ", ".join(t.name for t in mw_only_targets)
            result.warnings.append(
                f"MW만으로 매칭 (isotope 미검증): {names}. "
                f"분자식(SMILES)을 제공하면 신뢰도 향상."
            )

        # 이슈 4: 전환율이 음수에 가까움
        if result.conversion is not None and result.conversion < 5.0:
            result.warnings.append(
                f"전환율 {result.conversion}%: 반응이 거의 진행되지 않았거나, "
                f"before/after 스펙트럼 간 이온화 조건이 다를 수 있음"
            )

        # 이슈 5: 높은 미설명 비율이지만 byproduct 예측이 없음
        byproduct_targets = [t for t in targets if t.role == "Byproduct"]
        unexplained = result.reliability.get("unexplained_ratio", 0)
        if unexplained > 20.0 and len(byproduct_targets) == 0:
            result.issues.append(
                f"설명 불가 피크 {unexplained:.1f}%인데 부산물 예측 없음. "
                f"reaction_type을 지정하면 부산물 예측이 활성화됨."
            )


# ═══════════════════════════════════════════════════════════════
# 편의 함수
# ═══════════════════════════════════════════════════════════════

def run_echo_pipeline(
    mz_before: np.ndarray,
    int_before: np.ndarray,
    mz_after: np.ndarray,
    int_after: np.ndarray,
    sm_formula: Optional[str] = None,
    sm_mw: Optional[float] = None,
    product_formula: Optional[str] = None,
    product_mw: Optional[float] = None,
    reaction_type: Optional[str] = None,
    reagent_mw: float = 0.0,
    leaving_group: str = "Br",
    charge: int = 1,
    solvent_mzs: Optional[List[float]] = None,
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """한 줄로 파이프라인 실행.

    Example
    -------
    >>> result = run_echo_pipeline(
    ...     mz_before, int_before,    # 반응 전
    ...     mz_after, int_after,      # 반응 후
    ...     sm_formula="C10H12BrNO",
    ...     product_formula="C16H17NO2",
    ...     reaction_type="suzuki",
    ...     reagent_mw=121.0891,      # phenylboronic acid
    ... )
    >>> print(f"전환율: {result.conversion}%")
    >>> print(f"신뢰도: {result.reliability['explained_ratio']}%")
    """
    pipeline = EchoPipeline(config)

    if solvent_mzs:
        pipeline.set_solvent_mzs(solvent_mzs)

    pipeline.set_before_spectrum(
        np.asarray(mz_before), np.asarray(int_before),
        sm_formula=sm_formula, sm_mw=sm_mw, charge=charge,
    )

    return pipeline.run(
        np.asarray(mz_after), np.asarray(int_after),
        sm_formula=sm_formula, sm_mw=sm_mw,
        product_formula=product_formula, product_mw=product_mw,
        reaction_type=reaction_type, reagent_mw=reagent_mw,
        leaving_group=leaving_group, charge=charge,
    )
