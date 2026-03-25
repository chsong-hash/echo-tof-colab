"""CFM-ID 4.0 웹 서비스 클라이언트 (ML 기반 MS/MS 예측).

별도 설치 불필요 — requests만으로 동작.
원본: mass값 예측기 test.py 의 CFMIDAPI 클래스.
"""
import re
import time
from typing import Dict, Optional, Callable
from .config import logger

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class CFMIDClient:
    """CFM-ID 4.0 웹 서비스를 HTTP로 호출하는 클라이언트."""

    BASE_URL = "https://cfmid.wishartlab.com"

    def predict(
        self,
        smiles: str,
        ion_mode: str = "positive",
        adduct: str = "[M+H]+",
        threshold: float = 0.001,
        max_wait: int = 300,
        on_progress: Optional[Callable] = None,
    ) -> Dict:
        """SMILES → CFM-ID 예측 스펙트럼 반환.

        Returns
        -------
        dict
            {
                'spectra': {'energy0': [(mz, intensity), ...], ...},
                'fragments': [{'id': int, 'mass': float, 'smiles': str}, ...],
                'energy_labels': ['10V', '20V', '40V'],
            }
        """
        if not HAS_REQUESTS:
            raise ImportError("requests 미설치: pip install requests")

        session = requests.Session()

        # 1) CSRF 토큰
        r = session.get(f"{self.BASE_URL}/predict", timeout=30)
        r.raise_for_status()
        csrf_match = re.findall(r'authenticity_token.*?value="(.*?)"', r.text)
        if not csrf_match:
            raise RuntimeError("CFM-ID CSRF 토큰 획득 실패")

        # 2) 예측 제출
        data = {
            "utf8": "\u2713",
            "authenticity_token": csrf_match[0],
            "predict_query[compound]": smiles,
            "predict_query[spectra_type]": "ESI",
            "predict_query[ion_mode]": ion_mode,
            "predict_query[adduct_type]": adduct,
            "predict_query[threshold]": str(threshold),
            "commit": "Submit",
        }
        r2 = session.post(f"{self.BASE_URL}/predict/new", data=data,
                          timeout=60, allow_redirects=True)
        result_url = r2.url

        status = self._check_response(r2.text)
        if status == "done":
            return self._parse_result(r2.text)
        if status == "error":
            raise ValueError(self._extract_error(r2.text))

        # 3) 결과 폴링
        intervals = [1, 2, 3] + [5] * 60
        elapsed = 0
        for wait in intervals:
            if elapsed >= max_wait:
                break
            time.sleep(wait)
            elapsed += wait
            if on_progress:
                on_progress(f"CFM-ID 계산 중... {elapsed}초 경과")
            r3 = session.get(result_url, timeout=30)
            status = self._check_response(r3.text)
            if status == "done":
                return self._parse_result(r3.text)
            if status == "error":
                raise ValueError(self._extract_error(r3.text))

        raise TimeoutError(f"CFM-ID 예측이 {max_wait}초 내에 완료되지 않았습니다.")

    @staticmethod
    def _check_response(text):
        if "Input Errors" in text or "is invalid" in text:
            return "error"
        tables = re.findall(r"<table[^>]*>(.*?)</table>", text, re.DOTALL)
        return "done" if len(tables) >= 3 else "processing"

    @staticmethod
    def _extract_error(text):
        err = re.findall(r"Input Errors?:\s*([^<]+)", text)
        return f"CFM-ID 오류: {err[0].strip()}" if err else "CFM-ID 입력값 오류"

    @staticmethod
    def _parse_result(html) -> Dict:
        """CFM-ID 결과 HTML 파싱."""
        tables = re.findall(r"<table[^>]*>(.*?)</table>", html, re.DOTALL)
        if len(tables) < 3:
            raise RuntimeError("CFM-ID 결과 파싱 실패")

        energy_labels = re.findall(r"Energy\s*MsMs\s*Spectrum\s*\((\d+V)\)", html)
        if not energy_labels:
            energy_labels = ["10V", "20V", "40V"]

        spectra = {}
        for i, table in enumerate(tables[:3]):
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.DOTALL)
            peaks = []
            for row in rows:
                cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
                if len(cells) >= 2:
                    try:
                        mz = float(cells[0].strip())
                        intensity = float(cells[1].strip())
                        peaks.append((mz, intensity))
                    except ValueError:
                        continue
            spectra[f"energy{i}"] = peaks

        # Fragment table (4번째 테이블 이후)
        fragments = []
        if len(tables) > 3:
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tables[3], re.DOTALL)
            for row in rows:
                cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
                if len(cells) >= 3:
                    try:
                        fragments.append({
                            "id": int(cells[0].strip()),
                            "mass": float(cells[1].strip()),
                            "smiles": cells[2].strip(),
                        })
                    except (ValueError, IndexError):
                        continue

        return {
            "spectra": spectra,
            "fragments": fragments,
            "energy_labels": energy_labels,
        }
