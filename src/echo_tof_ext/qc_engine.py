"""QC 플래그 판정 + 신뢰도 점수 산출.

플래그 종류:
  - IS_RECOVERY_WARNING : IS 회수율 범위 이탈
  - IONIZATION_FLAG     : UV/MS 괴리 (이온화 편향)
  - RT_DRIFT_WARNING    : RT 드리프트
  - NOT_DETECTED        : S/N 미달
  - QC_FAIL             : IS CV 초과 (전체 플레이트)
  - BASELINE_WARNING    : baseline 이상
  - YIELD_OUT_OF_RANGE  : 수율 0~100% 범위 이탈

원본: LC 분석/20260306_0443/scripts/qc_engine.py
"""
import numpy as np
import pandas as pd
from .config import logger, PipelineConfig

FLAG_WEIGHTS = {
    "IS_RECOVERY_WARNING": 0.10,
    "IONIZATION_FLAG": 0.15,
    "RT_DRIFT_WARNING": 0.10,
    "NOT_DETECTED": 1.00,
    "QC_FAIL": 0.30,
    "BASELINE_WARNING": 0.10,
    "YIELD_OUT_OF_RANGE": 0.15,
}


def run_qc(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """모든 QC 체크를 실행하고 플래그 + 신뢰도 점수를 df에 추가."""
    result = df.copy()
    n = len(result)
    flags_list = [[] for _ in range(n)]

    # 1. IS 회수율 체크
    if "area_standard" in result.columns:
        median_is = result["area_standard"].median()
        if median_is > 0:
            result["is_recovery"] = (result["area_standard"] / median_is) * 100
            lo, hi = config.is_recovery_range
            for i in range(n):
                rec = result["is_recovery"].iloc[i]
                if not np.isnan(rec) and (rec < lo or rec > hi):
                    flags_list[i].append("IS_RECOVERY_WARNING")

            # IS 면적 CV
            is_cv = (result["area_standard"].std() / result["area_standard"].mean() * 100
                     if result["area_standard"].mean() > 0 else 0)
            if is_cv > config.is_cv_max:
                for i in range(n):
                    flags_list[i].append("QC_FAIL")
                logger.warning(f"IS 면적 CV = {is_cv:.1f}% > {config.is_cv_max}%: 전체 QC_FAIL")

    # 2. UV vs MS 괴리
    if "ionization_flag" in result.columns:
        for i in range(n):
            if result["ionization_flag"].iloc[i]:
                flags_list[i].append("IONIZATION_FLAG")

    # 3. RT 드리프트
    if "rt" in result.columns:
        check_rt = True
        if "compound_id" in result.columns and result["compound_id"].nunique() > 5:
            check_rt = False
        if check_rt:
            median_rt = result["rt"].median()
            for i in range(n):
                if abs(result["rt"].iloc[i] - median_rt) > config.rt_tolerance:
                    flags_list[i].append("RT_DRIFT_WARNING")

    # 4. S/N 체크
    if "sn" in result.columns:
        for i in range(n):
            sn = result["sn"].iloc[i]
            if not np.isnan(sn) and sn < config.sn_threshold:
                flags_list[i].append("NOT_DETECTED")

    # 5. 수율 범위 체크
    yield_cols = [c for c in result.columns if c.endswith("_yield")]
    for i in range(n):
        for yc in yield_cols:
            val = result[yc].iloc[i]
            if not np.isnan(val) and (val > 100 or val < 0):
                if "YIELD_OUT_OF_RANGE" not in flags_list[i]:
                    flags_list[i].append("YIELD_OUT_OF_RANGE")

    # QC 플래그 문자열 + 신뢰도 점수
    result["qc_flags"] = [";".join(f) if f else "" for f in flags_list]
    scores = []
    for f_list in flags_list:
        if "NOT_DETECTED" in f_list:
            scores.append(0.0)
        else:
            penalty = sum(FLAG_WEIGHTS.get(f, 0.1) for f in f_list)
            scores.append(max(0.0, 1.0 - penalty))
    result["confidence_score"] = scores
    logger.info(f"QC 완료: {sum(1 for s in scores if s < 1.0)}/{n} wells flagged")
    return result
