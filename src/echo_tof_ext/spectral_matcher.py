"""matchms 라이브러리 매칭 (MGF/MSP/GNPS).

원본: LC 분석/20260306_0443/scripts/spectral_matcher.py
"""
import pandas as pd
from pathlib import Path
from typing import Optional, List
from .config import logger


def load_library(path: str):
    """MGF 또는 MSP 라이브러리 로드."""
    try:
        from matchms.importing import load_from_mgf, load_from_msp
    except ImportError:
        raise ImportError("matchms 미설치: pip install matchms")

    p = Path(path)
    if p.suffix.lower() == ".mgf":
        spectra = list(load_from_mgf(str(p)))
    elif p.suffix.lower() == ".msp":
        spectra = list(load_from_msp(str(p)))
    else:
        raise ValueError(f"지원하지 않는 형식: {p.suffix}")

    logger.info(f"라이브러리 로드: {len(spectra)} spectra from {p.name}")
    return spectra


def match_spectra(query_spectra, library_spectra, top_k: int = 5,
                  min_score: float = 0.3) -> pd.DataFrame:
    """matchms CosineGreedy로 스펙트럼 매칭."""
    try:
        from matchms import calculate_scores
        from matchms.similarity import CosineGreedy
    except ImportError:
        raise ImportError("matchms 미설치: pip install matchms")

    similarity = CosineGreedy(tolerance=0.1)
    scores = calculate_scores(library_spectra, query_spectra, similarity)

    results = []
    for i, query in enumerate(query_spectra):
        query_mz = query.get("precursor_mz", 0.0)
        sorted_matches = scores.scores_by_query(query, "CosineGreedy_score", sort=True)
        for rank, (ref, score_tuple) in enumerate(sorted_matches[:top_k]):
            score_val = score_tuple[0] if isinstance(score_tuple, tuple) else float(score_tuple)
            matched_peaks = score_tuple[1] if isinstance(score_tuple, tuple) and len(score_tuple) > 1 else 0
            if score_val < min_score:
                continue
            results.append({
                "query_idx": i, "query_mz": query_mz,
                "match_rank": rank + 1,
                "matched_name": ref.get("compound_name", "Unknown"),
                "cosine_score": round(score_val, 4),
                "matched_peaks": int(matched_peaks),
            })
    return pd.DataFrame(results)


def run_spectral_matching(query_data, library_path: Optional[str],
                          top_k: int = 5) -> pd.DataFrame:
    """전체 스펙트럼 매칭 파이프라인."""
    if not library_path or not Path(library_path).exists():
        logger.info("라이브러리 없음 - 매칭 생략")
        return pd.DataFrame()
    library = load_library(library_path)
    if not query_data:
        return pd.DataFrame()
    return match_spectra(query_data, library, top_k=top_k)
