"""
c08_5m_data.py — 연구 10용: 32종목 5분봉 받기

- 바이낸스 공개 데이터(data.binance.vision)의 USDⓈ-M 선물 5분봉 '월별 파일'
- 기간: 2023-10 ~ 2026-10 (모의고사 + 수능 기간 전부 — 수능 기간도 받아두기만 하고 열어보지 않음)
- 상장 전 달은 파일이 없어서 건너뜀 (.missing 표시)
- 중간에 끊겨도 다시 실행하면 이어서 받음

필요: universe/train32.txt
실행: python c08_5m_data.py
결과: data/5m/코인/코인-5m-YYYY-MM.csv  (열: time, open, high, low, close)
"""
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

DATA = Path("data")
OUT = DATA / "5m"
MONTHS = pd.period_range("2023-10", "2026-10", freq="M")
WORKERS = 16


def parse(raw: bytes) -> pd.DataFrame:
    """머리줄 있는 파일/없는 파일 둘 다 처리. 시각 단위(ms/µs)도 자동 판별"""
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if not str(df.iloc[0, 0]).isdigit():          # 첫 줄이 머리줄이면 버림
        df = df.iloc[1:]
    t = pd.to_numeric(df[0])
    unit = "us" if t.iloc[0] > 1e14 else "ms"
    return pd.DataFrame({
        "time": pd.to_datetime(t, unit=unit, utc=True),
        "open": pd.to_numeric(df[1]), "high": pd.to_numeric(df[2]),
        "low": pd.to_numeric(df[3]), "close": pd.to_numeric(df[4]),
    })


def fetch(symbol: str, month: pd.Period) -> str:
    folder = OUT / symbol
    name = f"{symbol}-5m-{month}"
    csv, miss = folder / f"{name}.csv", folder / f"{name}.missing"
    if csv.exists() or miss.exists():
        return "skip"
    url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/5m/{name}.zip"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                miss.touch()
                return "missing"
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                df = parse(z.read(z.namelist()[0]))
            tmp = csv.with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(csv)
            return "ok"
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


if __name__ == "__main__":
    symbols = [s.strip() for s in Path("universe/train32.txt").read_text().split() if s.strip()]
    for sym in symbols:
        (OUT / sym).mkdir(parents=True, exist_ok=True)
    jobs = [(s, m) for s in symbols for m in MONTHS]
    print(f"{len(symbols)}종목 × {len(MONTHS)}개월 = 파일 {len(jobs):,}개")

    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for _ in ex.map(lambda a: fetch(*a), jobs):
            done += 1
            if done % 100 == 0:
                print(f"  {done:,}/{len(jobs):,}")

    print("\n코인별 받은 달 수")
    for sym in symbols:
        n = len(list((OUT / sym).glob("*.csv")))
        print(f"  {sym:<14}{n}개월")
