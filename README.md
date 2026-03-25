# Echo-TOF WIFF2 Data Processing & Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chsong-hash/echo-tof-colab/blob/main/notebooks/01_EchoTOF_WIFF2_Analysis.ipynb)

Echo-TOF (Acoustic Ejection Mass Spectrometry) 데이터를 처리하고 분석하는 Colab 노트북 컬렉션입니다.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | **WIFF2 Analysis** | WIFF2 파일 로드 → 피크 검출 → 적분 → 96웰 히트맵 → 화합물 매칭 |

## Quick Start

1. 위의 **Open in Colab** 배지를 클릭
2. 런타임 → 모두 실행
3. 샘플 데이터로 데모 실행 (실제 데이터 없이도 동작)

## 실제 데이터 사용

```python
# WIFF2 → mzML 변환 (ProteoWizard msconvert 사용)
# msconvert sample.wiff2 --mzML

# Colab에서 로드
spectra = load_mzml('/content/drive/MyDrive/data/sample.mzML')
```

## Features

- **WIFF2/mzML 데이터 로드** (pyOpenMS)
- **자동 피크 검출** (S/N 기반, baseline 보정)
- **96웰 플레이트 히트맵** (수율 시각화)
- **화합물 구조 매칭** (SMILES → m/z → 피크 할당)
- **배치 적분** (다중 타겟 m/z 동시 분석)

## Requirements

```
pyopenms, rdkit-pypi, pandas, matplotlib, seaborn, numpy, scipy
```

Colab에서 자동 설치됩니다.
