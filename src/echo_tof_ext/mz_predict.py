"""분자량(MW) → 예상 m/z 값 계산.

ESI(Electrospray Ionization)에서 관측되는 adduct 형태:
  - Positive: [M+H]+, [M+Na]+, [M+K]+, [M+NH4]+, [M+2H]2+, [M+H-H2O]+
  - Negative: [M-H]-, [M+FA-H]-, [M+Cl]-, [M-2H]2-

원본: LC 분석/20260306_0443/scripts/mz_predict.py
"""
import numpy as np
from typing import List, Dict, Optional
from .config import logger


# 정밀 질량 상수 (IUPAC 2016)
H = 1.00728       # proton mass
NA = 22.98922      # sodium
K = 38.96316       # potassium
NH4 = 18.03437     # ammonium
CL = 34.96885      # chlorine
FORMATE = 44.99820  # HCOO-
H2O = 18.01056     # water loss
ELECTRON = 0.00055  # electron mass

ADDUCTS = {
    "positive": [
        ("[M+H]+",       1,  H),
        ("[M+Na]+",      1,  NA - ELECTRON),
        ("[M+K]+",       1,  K - ELECTRON),
        ("[M+NH4]+",     1,  NH4),
        ("[M+2H]2+",     2,  2 * H),
        ("[M+H-H2O]+",   1,  H - H2O),
    ],
    "negative": [
        ("[M-H]-",       1,  -H),
        ("[M+FA-H]-",    1,  FORMATE - H),
        ("[M+Cl]-",      1,  CL - ELECTRON),
        ("[M-2H]2-",     2,  -2 * H),
    ],
}


def predict_mz(mw: float, mode: str = "both") -> List[Dict]:
    """MW로부터 예상 m/z 값 목록을 반환.

    Parameters
    ----------
    mw : float
        Monoisotopic molecular weight (Da).
    mode : str
        "positive", "negative", or "both"

    Returns
    -------
    list of dict
        {"adduct": str, "mz": float, "charge": int, "mode": str}
    """
    results = []
    modes = []
    if mode in ("positive", "both"):
        modes.append("positive")
    if mode in ("negative", "both"):
        modes.append("negative")

    for m in modes:
        for name, charge, shift in ADDUCTS[m]:
            mz = (mw + shift) / abs(charge)
            results.append({
                "adduct": name,
                "mz": round(mz, 4),
                "charge": charge,
                "mode": m,
            })
    return results


def mz_to_mw(mz: float, adduct: str = "[M+H]+") -> Optional[float]:
    """관측 m/z → MW 역산."""
    for mode_adducts in ADDUCTS.values():
        for name, charge, shift in mode_adducts:
            if name == adduct:
                return round(mz * abs(charge) - shift, 4)
    return None


def format_mz_table(mw: float, mode: str = "both") -> str:
    """m/z 예측 결과를 테이블 문자열로 반환."""
    predictions = predict_mz(mw, mode)
    lines = [f"MW = {mw:.4f} Da", ""]
    lines.append(f"{'Adduct':<18} {'m/z':>12} {'Charge':>8} {'Mode':>10}")
    lines.append("-" * 52)
    for p in predictions:
        lines.append(f"{p['adduct']:<18} {p['mz']:>12.4f} {p['charge']:>8} {p['mode']:>10}")
    return "\n".join(lines)
