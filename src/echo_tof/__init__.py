"""
ECHO-TOF: SCIEX Formula Finder Python 이식.

분자식 계산, 동위원소 패턴 예측, MS/MS 프래그먼트 분석.
원본: SCIEX OS FormulaFinder 플러그인 (Clearcore2 v3.0.0.0, .NET 4.7.2)
"""

__version__ = "0.1.0"

# 핵심 타입
from .elements import (
    Element,
    ElementCount,
    Isotope,
    PeriodicTable,
    ELECTRON_MASS,
    parse_formula,
    formula_to_string,
    get_monoisotopic_mass,
    get_average_mass,
    get_nominal_mass,
    calculate_rdb,
)

# 분자 객체
from .molecule import Molecule

# 동위원소 계산
from .isotope_calc import (
    IsotopicDistributionCalculator,
    IsotopicPeak,
)

# 분자식 열거
from .formula_enum import (
    FindElementalCompositions,
    ElectronState,
    find_compositions,
)

# 패턴 비교
from .pattern import MoleculePattern

# 필터링
from .filters import FormulaFilter

# 계산 유틸리티
from .calculations import (
    get_mass_error,
    get_mass_error_with_charge,
    convert_to_mz,
    get_charge_state_from_mz_array,
)

# 수학 유틸리티
from .math_utils import (
    linear_regression,
    standard_deviation,
    normal_distribution,
)

# TOF 재보정
from .mass_adjuster import MassAdjuster

# 프래그먼트 최적화
from .fragment import set_best_slope_points

# 전체 파이프라인
from .pipeline import (
    FormulaFinderPipeline,
    CompositionResult,
    FragmentResult,
)
