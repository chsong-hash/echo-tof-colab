"""반응 유형 기반 Product/Byproduct MW 예측 + SM/Product 동정.

3가지 독립 관점의 앙상블:
  1. 화학적 예측 — 반응 유형 + SM MW → Product MW 계산
  2. 통계적 검증 — SM↓ ↔ Product↑ anti-correlation
  3. MS 패턴 검증 — adduct pair 확인

원본: LC 분석/20260306_0443/scripts/reaction_predictor.py
"""
import numpy as np
from typing import Dict, List, Optional
from .config import logger
from .mz_predict import predict_mz, H


# ═══════════════════════════════════════════════════════════
# 반응 유형별 MW 변화량 (delta MW)
# ═══════════════════════════════════════════════════════════

REACTION_TYPES = {
    "amide_coupling": {
        "name": "Amide Coupling (HATU/EDC)",
        "description": "R-COOH + H2N-R' → R-CONH-R' + H2O",
        "delta_mw": -18.0106,
        "needs_reagent_mw": True,
    },
    "suzuki": {
        "name": "Suzuki Coupling",
        "description": "Ar-X + Ar'-B(OH)2 → Ar-Ar' + X-B(OH)2",
        "delta_mw": None,
        "needs_reagent_mw": True,
        "leaving_groups": {
            "Br": 78.9183, "Cl": 34.9689,
            "I": 126.9045, "OTf": 148.9520,
        },
        "boronic_acid_loss": 45.0148,
    },
    "buchwald_hartwig": {
        "name": "Buchwald-Hartwig Amination",
        "description": "Ar-X + H-NR2 → Ar-NR2 + HX",
        "delta_mw": None,
        "needs_reagent_mw": True,
        "leaving_groups": {"Br": 78.9183, "Cl": 34.9689, "I": 126.9045},
    },
    "snar": {
        "name": "Nucleophilic Aromatic Substitution (SNAr)",
        "description": "Ar-X + Nu-H → Ar-Nu + HX",
        "delta_mw": None,
        "needs_reagent_mw": True,
        "leaving_groups": {"F": 18.9984, "Cl": 34.9689, "NO2": 45.9929},
    },
    "reductive_amination": {
        "name": "Reductive Amination",
        "description": "R-CHO + H2N-R' → R-CH2-NH-R'",
        "delta_mw": -15.9949,
        "needs_reagent_mw": True,
    },
    "boc_deprotection": {
        "name": "Boc Deprotection",
        "description": "R-NHBoc → R-NH2 + CO2 + isobutylene",
        "delta_mw": -100.0524,
        "needs_reagent_mw": False,
    },
    "cbz_deprotection": {
        "name": "Cbz Deprotection",
        "description": "R-NHCbz → R-NH2 + CO2 + toluene",
        "delta_mw": -134.0368,
        "needs_reagent_mw": False,
    },
    "alkylation": {
        "name": "N/O-Alkylation",
        "description": "R-XH + R'-LG → R-X-R' + H-LG",
        "delta_mw": None,
        "needs_reagent_mw": True,
        "leaving_groups": {
            "Br": 78.9183, "Cl": 34.9689, "I": 126.9045,
            "OMs": 95.0116, "OTs": 171.0116,
        },
    },
    "click_chem": {
        "name": "CuAAC Click Chemistry",
        "description": "R-N3 + R'-CCH → triazole product",
        "delta_mw": 0.0,
        "needs_reagent_mw": True,
    },
}


def predict_product_mw(
    sm_mw: float,
    reaction_type: str,
    reagent_mw: float = 0.0,
    leaving_group: str = "Br",
) -> Optional[float]:
    """반응 유형 + SM MW + 시약 MW → Product MW 예측."""
    if reaction_type not in REACTION_TYPES:
        logger.warning(f"알 수 없는 반응 유형: {reaction_type}")
        return None

    rxn = REACTION_TYPES[reaction_type]
    h_mass = 1.00783

    if reaction_type == "suzuki":
        lg_mass = rxn["leaving_groups"].get(leaving_group, 79.9262)
        return round(sm_mw - lg_mass + reagent_mw - rxn["boronic_acid_loss"], 4)

    elif reaction_type in ("buchwald_hartwig", "snar", "alkylation"):
        lg_mass = rxn["leaving_groups"].get(leaving_group, 78.9183)
        return round(sm_mw - lg_mass + reagent_mw - h_mass, 4)

    elif reaction_type == "reductive_amination":
        return round(sm_mw + reagent_mw - 18.0106 + 2.0157, 4)

    elif reaction_type in ("amide_coupling", "click_chem"):
        return round(sm_mw + reagent_mw + rxn["delta_mw"], 4)

    elif rxn["delta_mw"] is not None and not rxn.get("needs_reagent_mw", True):
        return round(sm_mw + rxn["delta_mw"], 4)

    return None


def predict_byproduct_mws(
    sm_mw: float,
    reaction_type: str,
    reagent_mw: float = 0.0,
    leaving_group: str = "Br",
) -> List[Dict]:
    """반응 유형별 예상 byproduct MW 목록을 반환."""
    if reaction_type not in REACTION_TYPES:
        return []

    rxn = REACTION_TYPES[reaction_type]
    byproducts = []
    h_mass = 1.00783

    if reaction_type == "suzuki":
        lg_mass = rxn["leaving_groups"].get(leaving_group, 78.9183)
        dehal_mw = sm_mw - lg_mass + h_mass
        byproducts.append({
            "name": "Proto-dehalogenation (SM-X → SM-H)",
            "mw": round(dehal_mw, 4),
            "mz_mh": round(dehal_mw + H, 4),
            "likelihood": "high",
        })
        homo_mw = 2 * (sm_mw - lg_mass)
        byproducts.append({
            "name": "SM Homo-coupling",
            "mw": round(homo_mw, 4),
            "mz_mh": round(homo_mw + H, 4),
            "likelihood": "medium",
        })
        if reagent_mw > 0:
            reagent_homo_mw = 2 * (reagent_mw - rxn["boronic_acid_loss"])
            byproducts.append({
                "name": "Reagent Homo-coupling",
                "mw": round(reagent_homo_mw, 4),
                "mz_mh": round(reagent_homo_mw + H, 4),
                "likelihood": "medium",
            })

    elif reaction_type in ("buchwald_hartwig", "snar"):
        lg_mass = rxn["leaving_groups"].get(leaving_group, 78.9183)
        dehal_mw = sm_mw - lg_mass + h_mass
        byproducts.append({
            "name": "Dehalogenation (Ar-X → Ar-H)",
            "mw": round(dehal_mw, 4),
            "mz_mh": round(dehal_mw + H, 4),
            "likelihood": "high",
        })
        if reagent_mw > 0 and reaction_type == "buchwald_hartwig":
            double_mw = 2 * (sm_mw - lg_mass) + reagent_mw - 2 * h_mass
            byproducts.append({
                "name": "Bis-arylation",
                "mw": round(double_mw, 4),
                "mz_mh": round(double_mw + H, 4),
                "likelihood": "medium",
            })

    elif reaction_type == "amide_coupling":
        byproducts.append({
            "name": "Unreacted SM (activated ester hydrolysis)",
            "mw": round(sm_mw, 4),
            "mz_mh": round(sm_mw + H, 4),
            "likelihood": "high",
        })
        anhydride_mw = 2 * sm_mw - 18.0106
        byproducts.append({
            "name": "Symmetric anhydride",
            "mw": round(anhydride_mw, 4),
            "mz_mh": round(anhydride_mw + H, 4),
            "likelihood": "low",
        })

    elif reaction_type == "reductive_amination":
        if reagent_mw > 0:
            over_mw = sm_mw + 2 * reagent_mw - 2 * 18.0106 + 2 * 2.0157
            byproducts.append({
                "name": "Over-alkylation (tertiary amine)",
                "mw": round(over_mw, 4),
                "mz_mh": round(over_mw + H, 4),
                "likelihood": "medium",
            })
        alcohol_mw = sm_mw + 2.0157
        byproducts.append({
            "name": "Aldehyde reduction (R-CHO → R-CH2OH)",
            "mw": round(alcohol_mw, 4),
            "mz_mh": round(alcohol_mw + H, 4),
            "likelihood": "medium",
        })

    return byproducts


def find_adduct_pairs(
    mz_list: np.ndarray,
    intensity_list: np.ndarray,
    mz_tolerance: float = 0.02,
) -> List[Dict]:
    """스펙트럼에서 adduct pair를 찾아 분자 동정을 확인."""
    PAIR_DELTAS = [
        ("H_Na",   21.9819, "[M+H]+ / [M+Na]+"),
        ("H_K",    37.9559, "[M+H]+ / [M+K]+"),
        ("H_NH4",  17.0271, "[M+H]+ / [M+NH4]+"),
        ("H_H2O", -18.0106, "[M+H]+ / [M+H-H2O]+"),
        ("Na_K",   15.9739, "[M+Na]+ / [M+K]+"),
    ]
    pairs = []
    n = len(mz_list)
    for i in range(n):
        for j in range(i + 1, n):
            delta = mz_list[j] - mz_list[i]
            for pair_name, expected_delta, description in PAIR_DELTAS:
                if abs(delta - expected_delta) <= mz_tolerance:
                    pairs.append({
                        "pair_type": pair_name,
                        "description": description,
                        "mz_1": round(float(mz_list[i]), 4),
                        "mz_2": round(float(mz_list[j]), 4),
                        "delta_observed": round(float(delta), 4),
                        "predicted_mw": round(float(mz_list[i]) - H, 4),
                    })
    return pairs


def validate_compound_by_adducts(
    mz_list: np.ndarray,
    intensity_list: np.ndarray,
    expected_mw: float,
    mz_tolerance: float = 0.02,
    mode: str = "positive",
) -> Dict:
    """expected MW에 해당하는 adduct들이 스펙트럼에 존재하는지 확인."""
    predicted = predict_mz(expected_mw, mode)
    found = []
    for pred in predicted:
        target_mz = pred["mz"]
        matches = np.where(np.abs(mz_list - target_mz) <= mz_tolerance)[0]
        if len(matches) > 0:
            best_idx = matches[np.argmax(intensity_list[matches])]
            found.append({
                "adduct": pred["adduct"],
                "expected_mz": target_mz,
                "observed_mz": round(float(mz_list[best_idx]), 4),
                "error_da": round(abs(float(mz_list[best_idx]) - target_mz), 4),
            })

    n_found = len(found)
    n_possible = len(predicted)
    return {
        "expected_mw": expected_mw,
        "adducts_found": found,
        "n_found": n_found,
        "n_possible": n_possible,
        "confidence": round(n_found / max(n_possible, 1), 3),
        "confirmed": n_found >= 2,
    }
