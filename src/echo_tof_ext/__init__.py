"""ECHO-TOF 확장 모듈.

기존 echo_tof 코드를 건드리지 않고, 수율 산출 파이프라인에 필요한
확장 기능을 모듈별로 분할하여 제공한다.

모듈 구성:
    config              - 파이프라인 설정 (PipelineConfig)
    neutral_loss_db     - 중성 소실 / 결합 절단 패턴 DB
    fragmentation_engine - 구조 기반 fragmentation 예측 (RDKit)
    mz_predict          - MW → m/z adduct 계산
    reaction_predictor  - 반응 유형별 Product/Byproduct MW 예측
    yield_calculator    - 6가지 수율 계산 모드
    qc_engine           - QC 플래그 + 신뢰도 점수
    peak_integration    - EIC 추출 + 피크 검출/적분
    baseline_correction - SNIP/TopHat baseline 보정
    spectral_matcher    - matchms 라이브러리 매칭
    peak_classifier     - 피크 분류 (Known/Inferred/Unknown)
    cfm_id_client       - CFM-ID 웹 API 클라이언트
"""
__version__ = "0.1.0"
