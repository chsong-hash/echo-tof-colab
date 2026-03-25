"""파이프라인 설정 및 공통 유틸리티."""
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("echo-tof-ext")


@dataclass
class PipelineConfig:
    """수율 산출 파이프라인 설정."""
    # --- 입력 ---
    input_path: str = ""
    input_format: str = "csv"  # csv | mzml

    # --- 수율 계산 ---
    yield_modes: List[str] = field(
        default_factory=lambda: ["SM_consumption"]
    )
    calibration_factor: float = 1.0
    calibration_factor_uv: float = 1.0

    # --- QC ---
    is_recovery_range: Tuple[float, float] = (70.0, 130.0)
    hit_threshold: float = 50.0
    confidence_min: float = 0.5
    rt_tolerance: float = 0.1
    sn_threshold: float = 3.0
    is_cv_max: float = 20.0
    uv_ms_divergence_max: float = 20.0

    # --- MS ---
    target_mz_list: Optional[List[float]] = None
    mz_tolerance: float = 0.005  # 5 ppm at 1000 Da
    rt_window: float = 0.5

    # --- 라이브러리 ---
    library_path: Optional[str] = None

    # --- 반응 예측 ---
    reaction_type: Optional[str] = None
    reagent_mw: float = 0.0
    sm_mw: Optional[float] = None
    leaving_group: str = "Br"
    sm_blank_well: Optional[str] = None
    blank_wells: Optional[List[str]] = None

    # --- 피크 분류 ---
    unexplained_threshold: float = 5.0  # 설명 불가 피크 비율(%) 경고 임계값

    # --- 출력 ---
    output_dir: str = "./output"
