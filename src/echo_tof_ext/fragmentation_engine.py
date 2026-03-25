"""구조 기반 MS/MS fragmentation 예측 엔진.

SMILES 입력 → 이론적 fragment 리스트 반환.
9가지 fragmentation 전략을 순차 적용:
  1. BRICS 분해
  2. 의약 결합 절단 (SMARTS)
  3. 체계적 단일결합 절단
  4. 중성 소실 (Neutral Loss)
  5. 헤테로고리 고리 개열
  6. 고리 내 결합 절단
  7. 연속 소실 (2차 fragmentation)
  8. McLafferty 재배열
  9. Ortho 효과

원본: mass값 예측기 test.py (PyQt5 GUI)에서 순수 로직만 분리.
RDKit 의존.
"""
import re
from typing import List, Dict, Optional

from .neutral_loss_db import (
    PHARMA_BOND_SMARTS,
    NEUTRAL_LOSSES,
    HETEROCYCLE_LOSSES,
)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, BRICS, Descriptors
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


MAX_FRAGMENTS = 35


def predict_fragments(smiles: str) -> List[Dict]:
    """SMILES → fragment 리스트 반환.

    Returns
    -------
    list of dict
        각 fragment: smiles, mol, type, formula, exact_mass,
                     mz_pos ([M+H]+), mz_neg ([M-H]-), rel_intensity
    """
    if not HAS_RDKIT:
        raise ImportError("RDKit가 설치되어 있지 않습니다: conda install -c conda-forge rdkit")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    mol_noh = Chem.RemoveHs(mol)
    parent_mass = Descriptors.ExactMolWt(mol_noh)

    all_frags = []
    seen = set()

    strategies = [
        _brics,
        _pharma_bonds,
        _systematic,
        lambda m: _neutral_losses(m, parent_mass),
        lambda m: _heterocycle_losses(m, parent_mass),
        _ring_bond_fragmentation,
        lambda m: _consecutive_losses(m, parent_mass),
        lambda m: _mclafferty(m, parent_mass),
        lambda m: _ortho_effect(m, parent_mass),
    ]

    for strategy in strategies:
        for frag in strategy(mol_noh):
            if frag["smiles"] not in seen:
                seen.add(frag["smiles"])
                all_frags.append(frag)

    # 속성 계산
    for frag in all_frags:
        if frag["mol"] is not None:
            frag["formula"] = CalcMolFormula(frag["mol"])
            frag["exact_mass"] = Descriptors.ExactMolWt(frag["mol"])
        frag["mz_pos"] = frag["exact_mass"] + 1.00728  # [M+H]+
        frag["mz_neg"] = abs(frag["exact_mass"] - 1.00728)  # [M-H]-

    # 강도 추정
    _estimate_intensities(all_frags, parent_mass)

    # 필터: 모분자의 5% 미만 제거
    all_frags = [f for f in all_frags if f["exact_mass"] > parent_mass * 0.05]

    # 상위 MAX_FRAGMENTS개만
    all_frags.sort(key=lambda x: x["rel_intensity"], reverse=True)
    all_frags = all_frags[:MAX_FRAGMENTS]

    # 질량 내림차순 재정렬
    all_frags.sort(key=lambda x: x["exact_mass"], reverse=True)
    return all_frags


def get_parent_mass(smiles: str) -> Optional[float]:
    """SMILES의 exact monoisotopic mass 반환."""
    if not HAS_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.ExactMolWt(mol)


def get_molecular_formula(smiles: str) -> Optional[str]:
    """SMILES의 분자식 반환."""
    if not HAS_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return CalcMolFormula(mol)


# ─── 내부 구현 ────────────────────────────────────────────

def _clean_fragment(smi: str):
    """BRICS/bond-cleavage dummy atom 제거 → RDKit mol 반환."""
    clean = re.sub(r"\[\d*\*\]", "[H]", smi)
    clean = clean.replace("*", "")
    fm = Chem.MolFromSmiles(clean)
    if fm and fm.GetNumHeavyAtoms() >= 2:
        fm = Chem.RemoveHs(fm)
        AllChem.Compute2DCoords(fm)
        return fm
    return None


def _brics(mol) -> List[Dict]:
    fragments = []
    try:
        frags = BRICS.BRICSDecompose(mol, returnMols=False)
    except Exception:
        return fragments
    for smi in frags:
        fm = _clean_fragment(smi)
        if fm:
            fragments.append({
                "smiles": Chem.MolToSmiles(fm), "mol": fm,
                "type": "BRICS", "bond_idx": None, "atom_pair": None,
            })
    return fragments


def _pharma_bonds(mol) -> List[Dict]:
    fragments = []
    for bond_name, smarts in PHARMA_BOND_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        for match in mol.GetSubstructMatches(pattern):
            bond = mol.GetBondBetweenAtoms(match[0], match[1])
            if bond is None:
                continue
            try:
                frag_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True)
                for fs in Chem.MolToSmiles(frag_mol).split("."):
                    fm = _clean_fragment(fs)
                    if fm:
                        fragments.append({
                            "smiles": Chem.MolToSmiles(fm), "mol": fm,
                            "type": bond_name, "bond_idx": bond.GetIdx(),
                            "atom_pair": (match[0], match[1]),
                        })
            except Exception:
                continue
    return fragments


def _systematic(mol) -> List[Dict]:
    fragments = []
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            continue
        try:
            frag_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True)
            for fs in Chem.MolToSmiles(frag_mol).split("."):
                fm = _clean_fragment(fs)
                if fm:
                    fragments.append({
                        "smiles": Chem.MolToSmiles(fm), "mol": fm,
                        "type": "Bond", "bond_idx": bond.GetIdx(),
                        "atom_pair": (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                    })
        except Exception:
            continue
    return fragments


def _neutral_losses(mol, parent_mass) -> List[Dict]:
    fragments = []
    for name, loss_mass, smarts in NEUTRAL_LOSSES:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            mass_after = parent_mass - loss_mass
            if mass_after > 10:
                fragments.append({
                    "smiles": f"[M-{name}]", "mol": None,
                    "type": f"NL(-{name})", "formula": f"-{name}",
                    "exact_mass": mass_after, "bond_idx": None, "atom_pair": None,
                })
    return fragments


def _heterocycle_losses(mol, parent_mass) -> List[Dict]:
    fragments = []
    for name, loss_mass, smarts in HETEROCYCLE_LOSSES:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            mass_after = parent_mass - loss_mass
            if mass_after > 10:
                fragments.append({
                    "smiles": f"[M-{name}]", "mol": None,
                    "type": f"Ring(-{name})", "formula": f"-{name}",
                    "exact_mass": mass_after, "bond_idx": None, "atom_pair": None,
                })
    return fragments


def _ring_bond_fragmentation(mol) -> List[Dict]:
    fragments = []
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() == 0:
        return fragments

    hetero_bonds = set()
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            continue
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        is_hetero = (a1.GetAtomicNum() not in (6, 1)) or (a2.GetAtomicNum() not in (6, 1))
        if not is_hetero:
            for nb in list(a1.GetNeighbors()) + list(a2.GetNeighbors()):
                if nb.GetAtomicNum() not in (6, 1) and nb.IsInRing():
                    is_hetero = True
                    break
        if is_hetero:
            hetero_bonds.add(bond.GetIdx())

    hb_list = sorted(hetero_bonds)
    for i in range(len(hb_list)):
        for j in range(i + 1, min(i + 4, len(hb_list))):
            try:
                frag_mol = Chem.FragmentOnBonds(mol, [hb_list[i], hb_list[j]], addDummies=True)
                for fs in Chem.MolToSmiles(frag_mol).split("."):
                    fm = _clean_fragment(fs)
                    if fm:
                        fragments.append({
                            "smiles": Chem.MolToSmiles(fm), "mol": fm,
                            "type": "RingOpen", "bond_idx": hb_list[i], "atom_pair": None,
                        })
            except Exception:
                continue
    return fragments


def _consecutive_losses(mol, parent_mass) -> List[Dict]:
    fragments = []
    all_losses = list(NEUTRAL_LOSSES) + list(HETEROCYCLE_LOSSES)

    applicable = []
    for name, loss_mass, smarts in all_losses:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            applicable.append((name, loss_mass))

    priority = ["H2O", "NH3", "CO", "HCN", "CO2", "HF", "HCl",
                "CH2CO", "NO", "SO2", "N2", "C2H2"]
    primary = []
    seen_names = set()
    for pname in priority:
        for name, mass in applicable:
            if name == pname and name not in seen_names:
                primary.append((name, mass))
                seen_names.add(name)
                break
    for name, mass in applicable:
        if name not in seen_names and len(primary) < 6:
            primary.append((name, mass))
            seen_names.add(name)

    seen = set()
    for n1, m1 in primary:
        for n2, m2 in primary:
            if n1 == n2:
                continue
            combo = tuple(sorted([n1, n2]))
            if combo in seen:
                continue
            seen.add(combo)
            mass_after = parent_mass - m1 - m2
            if mass_after > parent_mass * 0.1:
                fragments.append({
                    "smiles": f"[M-{n1}-{n2}]", "mol": None,
                    "type": f"Consec(-{n1}-{n2})", "formula": f"-{n1}-{n2}",
                    "exact_mass": mass_after, "bond_idx": None, "atom_pair": None,
                })
            if len(fragments) >= 15:
                return fragments
    return fragments


def _mclafferty(mol, parent_mass) -> List[Dict]:
    fragments = []
    patterns = [
        ("[CX3](=O)[CX4][CX4][CX4;H1,H2,H3]", "C2H4", 28.0313),
        ("[CX3](=O)[OX2][CX4][CX4][CX4;H1,H2,H3]", "C2H4", 28.0313),
        ("[CX3](=O)([OX2H])[CX4][CX4][CX4;H1,H2,H3]", "C2H4", 28.0313),
        ("[CX3](=O)([NX3])[CX4][CX4][CX4;H1,H2,H3]", "C2H4", 28.0313),
        ("[CX3](=O)[CX4][CX4][CX4][CX4;H1,H2,H3]", "C3H6", 42.0470),
        ("[CX3](=O)[CX4][CX4][CX4][CX4][CX4;H1,H2,H3]", "C4H8", 56.0626),
    ]
    seen = set()
    for smarts, loss_name, loss_mass in patterns:
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            key = f"McLaff-{loss_name}"
            if key not in seen:
                seen.add(key)
                mass_after = parent_mass - loss_mass
                if mass_after > 10:
                    fragments.append({
                        "smiles": f"[M-{loss_name}(McLaff)]", "mol": None,
                        "type": f"McLafferty(-{loss_name})", "formula": f"-{loss_name}",
                        "exact_mass": mass_after, "bond_idx": None, "atom_pair": None,
                    })
    return fragments


def _ortho_effect(mol, parent_mass) -> List[Dict]:
    fragments = []
    ortho_rules = [
        ("[cX3]([OX2H])[cX3][CX3](=O)[OX2H]", "H2O+CO", 45.9949 + 0.0106),
        ("[cX3]([OX2H])[cX3][CX3](=O)", "H2O", 18.0106),
        ("[cX3]([NX3;H2])[cX3][CX3](=O)", "NH3", 17.0266),
        ("[cX3]([OX2H])[cX3][NX3](=O)=O", "HNO2", 44.9982),
        ("[cX3]([OX2H])[cX3][Cl]", "HCl(ortho)", 35.9767),
        ("[cX3]([OX2H])[cX3][F]", "HF(ortho)", 20.0062),
        ("[cX3]([OX2H])[cX3][Br]", "HBr(ortho)", 79.9262),
        ("[cX3]([NX3;H2])[cX3][Cl]", "HCl(ortho-N)", 35.9767),
        ("[cX3]([SX2H])[cX3][CX3](=O)", "H2S", 33.9877),
    ]
    seen = set()
    for smarts, loss_name, loss_mass in ortho_rules:
        pat = Chem.MolFromSmarts(smarts)
        if pat and mol.HasSubstructMatch(pat):
            if loss_name not in seen:
                seen.add(loss_name)
                mass_after = parent_mass - loss_mass
                if mass_after > 10:
                    fragments.append({
                        "smiles": f"[M-{loss_name}]", "mol": None,
                        "type": f"Ortho(-{loss_name})", "formula": f"-{loss_name}",
                        "exact_mass": mass_after, "bond_idx": None, "atom_pair": None,
                    })
    return fragments


def _estimate_intensities(fragments, parent_mass):
    if not fragments:
        return
    for frag in fragments:
        ratio = frag["exact_mass"] / parent_mass if parent_mass > 0 else 0
        base_map = {
            "BRICS": 60, "Amide": 85, "Ester": 80, "Ether": 50,
            "Amine": 55, "Sulfonamide": 75, "Urea": 70,
            "Carbamate": 70, "Thioether": 45, "ArylC-N": 65,
            "ArylC-O": 60, "BenzylC": 50, "Bond": 35, "RingOpen": 55,
        }
        base = base_map.get(frag["type"], 60)
        if frag["type"].startswith("NL"):
            base = 70
        elif frag["type"].startswith("Ring"):
            base = 75
        elif frag["type"].startswith("Consec"):
            base = 45
        if 0.4 < ratio < 0.85:
            base *= 1.2
        elif ratio > 0.95:
            base *= 0.3
        frag["rel_intensity"] = min(100, max(5, base))

    max_int = max(f["rel_intensity"] for f in fragments)
    if max_int > 0:
        for f in fragments:
            f["rel_intensity"] = int(f["rel_intensity"] / max_int * 999)
