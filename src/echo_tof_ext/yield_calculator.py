"""6가지 수율 계산 모드 구현.

모드:
  1. IS_corrected     — IS 보정 수율
  2. SM_consumption   — SM 소비율 (출발물질 기준 전환율)
  3. UV_MS_cross      — UV/MS 교차 검증
  4. LCAP             — LC Area Percent (UV 기준)
  5. area_ratio_all   — 전체 면적비
  6. area_ratio_sm_only — SM 대비 면적비

원본: LC 분석/20260306_0443/scripts/yield_calculator.py
"""
import numpy as np
import pandas as pd
from typing import Optional
from .config import logger, PipelineConfig


def calc_is_corrected(df: pd.DataFrame, config: PipelineConfig) -> pd.Series:
    """IS 보정 수율: (area_product / area_standard) * calibration_factor * 100."""
    with np.errstate(divide="ignore", invalid="ignore"):
        yield_vals = (df["area_product"] / df["area_standard"]) * config.calibration_factor * 100
    return yield_vals.replace([np.inf, -np.inf], np.nan)


def calc_sm_consumption(df: pd.DataFrame, config: PipelineConfig) -> pd.Series:
    """SM 소비율: (1 - area_sm / area_sm_blank) * 100."""
    if df["area_sm"].isna().all():
        return pd.Series(np.nan, index=df.index)

    if config.sm_blank_well:
        blank_mask = df["well_id"] == config.sm_blank_well
        if blank_mask.any():
            area_sm_blank = df.loc[blank_mask, "area_sm"].iloc[0]
        else:
            area_sm_blank = df["area_sm"].dropna().quantile(0.95)
            logger.warning(f"sm_blank_well '{config.sm_blank_well}' 미발견 → 95th percentile 사용")
    else:
        area_sm_blank = df["area_sm"].dropna().quantile(0.95)

    if area_sm_blank == 0 or np.isnan(area_sm_blank):
        return pd.Series(np.nan, index=df.index)

    with np.errstate(divide="ignore", invalid="ignore"):
        yield_vals = (1 - df["area_sm"] / area_sm_blank) * 100
    return yield_vals.replace([np.inf, -np.inf], np.nan)


def calc_uv_ms_cross(df: pd.DataFrame, config: PipelineConfig):
    """UV/MS 교차 검증 수율. (avg_yield, uv_yield, divergence_flag) 반환."""
    ms_yield = calc_is_corrected(df, config)

    if df["uv_area"].isna().all():
        return ms_yield, pd.Series(np.nan, index=df.index), pd.Series(False, index=df.index)

    if "uv_area_is" in df.columns and not df["uv_area_is"].isna().all():
        with np.errstate(divide="ignore", invalid="ignore"):
            uv_yield = (df["uv_area"] / df["uv_area_is"]) * config.calibration_factor_uv * 100
    else:
        uv_max = df["uv_area"].max()
        uv_yield = (df["uv_area"] / uv_max) * 100 if uv_max > 0 else pd.Series(np.nan, index=df.index)
    uv_yield = uv_yield.replace([np.inf, -np.inf], np.nan)

    valid_mask = ms_yield.notna() & uv_yield.notna()
    flag = pd.Series(False, index=df.index)
    if valid_mask.sum() >= 5:
        ms_rank = ms_yield[valid_mask].rank(pct=True)
        uv_rank = uv_yield[valid_mask].rank(pct=True)
        rank_diff = (ms_rank - uv_rank).abs()
        threshold = config.uv_ms_divergence_max / 100.0
        flag_valid = rank_diff > max(threshold, 0.3)
        flag.loc[flag_valid.index] = flag_valid

    avg_yield = (ms_yield + uv_yield) / 2
    return avg_yield, uv_yield, flag


def calc_lcap(df: pd.DataFrame) -> pd.Series:
    """LCAP (LC Area Percent) 수율: uv_area / uv_total_area * 100."""
    if "uv_total_area" not in df.columns or df["uv_total_area"].isna().all():
        logger.warning("LCAP: uv_total_area 없음 → NaN 반환")
        return pd.Series(np.nan, index=df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        yield_vals = (df["uv_area"] / df["uv_total_area"]) * 100
    return yield_vals.replace([np.inf, -np.inf], np.nan)


def calc_area_ratio_all(df: pd.DataFrame) -> pd.Series:
    """전체 면적비 수율: product / (product + SM + byproducts)."""
    area_sm = df["area_sm"].fillna(0)
    area_bp_col = "area_byproducts_sum"
    has_byproducts = (area_bp_col in df.columns
                      and df[area_bp_col].notna().any()
                      and df[area_bp_col].sum() > 0)

    if has_byproducts:
        total = df["area_product"] + area_sm + df[area_bp_col].fillna(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            yield_vals = (df["area_product"] / total) * 100
        return yield_vals.replace([np.inf, -np.inf], np.nan)

    if "uv_total_area" in df.columns and df["uv_total_area"].notna().any():
        logger.info("area_ratio_all: byproduct 없음 → LCAP(UV area%)로 대체")
        with np.errstate(divide="ignore", invalid="ignore"):
            yield_vals = (df["uv_area"] / df["uv_total_area"]) * 100
        return yield_vals.replace([np.inf, -np.inf], np.nan)

    logger.warning("area_ratio_all: byproduct/UV 모두 없음 → SM-only ratio 폴백")
    total = df["area_product"] + area_sm
    with np.errstate(divide="ignore", invalid="ignore"):
        yield_vals = (df["area_product"] / total) * 100
    return yield_vals.replace([np.inf, -np.inf], np.nan)


def calc_area_ratio_sm_only(df: pd.DataFrame) -> pd.Series:
    """SM 대비 면적비 수율: product / (product + SM)."""
    total = df["area_product"] + df["area_sm"].fillna(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        yield_vals = (df["area_product"] / total) * 100
    return yield_vals.replace([np.inf, -np.inf], np.nan)


def calculate_all_yields(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """선택된 모드에 따라 모든 수율을 계산하여 df에 추가."""
    result = df.copy()

    mode_funcs = {
        "IS_corrected": lambda: ("IS_corrected_yield", calc_is_corrected(df, config)),
        "SM_consumption": lambda: ("SM_consumption_yield", calc_sm_consumption(df, config)),
        "area_ratio_all": lambda: ("area_ratio_all_yield", calc_area_ratio_all(df)),
        "area_ratio_sm_only": lambda: ("area_ratio_sm_only_yield", calc_area_ratio_sm_only(df)),
        "LCAP": lambda: ("LCAP_yield", calc_lcap(df)),
    }

    for mode in config.yield_modes:
        if mode in mode_funcs:
            col_name, values = mode_funcs[mode]()
            result[col_name] = values
            logger.info(f"{mode} 수율 계산 완료")
        elif mode == "UV_MS_cross":
            avg, uv, flag = calc_uv_ms_cross(df, config)
            result["uv_ms_cross_yield"] = avg
            result["uv_yield_raw"] = uv
            result["ionization_flag"] = flag
            logger.info("UV_MS_cross 수율 계산 완료")

    return result
