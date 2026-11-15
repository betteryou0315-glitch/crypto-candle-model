"""
c05_multi_data.py — 32종목으로 넓히기: 정답지 + 힌트 만들기

- 코인 목록: universe/train32.txt (crypto-storm-model 폴더의 것을 그대로 복사 — 연구 07·08과 같은 32종목)
- 정답지·힌트 규칙은 BTC 때와 완전히 같음 (c01_label.make_label, c02_features.make_features 를 그대로 씀)
- '평소'는 코인마다 따로 (그 코인의 직전 14일)

필요: 같은 폴더에 c01_label.py, c02_features.py, universe/train32.txt
실행: python c05_multi_data.py
결과: data/multi_features.csv (모든 코인을 한 파일로, symbol 열 추가)

시간: OI 하루 파일을 코인마다 받아서 처음엔 1~2시간 걸릴 수 있음.
      중간에 끊겨도 다시 실행하면 받은 데서부터 이어감 (data/multi/, data/metrics/코인/ 에 저장)
"""
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from c01_label import make_label
from c02_features import END, START, make_features

DATA = Path("data")
MULTI = DATA / "multi"
MULTI.mkdir(parents=True, exist_ok=True)


def download_klines(symbol: str) -> pd.DataFrame:
    path = MULTI / f"{symbol}_1h.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["time"])
    url = "https://fapi.binance.com/fapi/v1/klines"
    start = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp(END, tz="UTC").timestamp() * 1000)
    rows = []
    while start < end:
        r = requests.get(url, params={"symbol": symbol, "interval": "1h", "startTime": start,
                                      "endTime": end - 1, "limit": 1500}, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        start = batch[-1][0] + 3_600_000
        time.sleep(0.2)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = pd.DataFrame({
        "time": pd.to_datetime(df[0], unit="ms", utc=True),
        "open": df[1].astype(float), "high": df[2].astype(float),
        "low": df[3].astype(float), "close": df[4].astype(float),
        "volume": df[5].astype(float), "taker_buy_vol": df[9].astype(float),
    }).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


WORKERS = 16   # 하루 파일을 동시에 몇 개씩 받을지


def fetch_day(url: str, csv: Path, miss: Path) -> None:
    """하루 파일 하나 받기 (끊기면 3번까지 다시 시도)"""
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                miss.touch()          # 없는 날도 기록해서 다음엔 다시 안 물어봄
                return
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                tmp = csv.with_suffix(".tmp")
                tmp.write_bytes(z.read(z.namelist()[0]))
                tmp.replace(csv)      # 다 쓴 다음에 이름 바꾸기 → 반쯤 쓴 파일이 남지 않음
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def download_oi(symbol: str, first_day: pd.Timestamp) -> pd.DataFrame:
    """상장 후(첫 1시간봉 날짜)부터만 받음 — 상장 전 날짜는 요청 안 함"""
    folder = DATA / "metrics" / symbol
    folder.mkdir(parents=True, exist_ok=True)
    base = f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}"
    days = pd.date_range(first_day.tz_localize(None).normalize(),
                         pd.Timestamp(END) - pd.Timedelta(days=1), freq="D")
    todo = []
    for d in days:
        name = f"{symbol}-metrics-{d:%Y-%m-%d}"
        csv, miss = folder / f"{name}.csv", folder / f"{name}.missing"
        if not csv.exists() and not miss.exists():
            todo.append((f"{base}/{name}.zip", csv, miss))
    if todo:
        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(lambda a: fetch_day(*a), todo))

    parts, missing = [], 0
    for d in days:
        name = f"{symbol}-metrics-{d:%Y-%m-%d}"
        csv = folder / f"{name}.csv"
        if not csv.exists():
            missing += 1
            continue
        part = pd.read_csv(csv)
        if "create_time" not in part.columns:
            part = pd.read_csv(csv, header=None).iloc[:, :3]
            part.columns = ["create_time", "symbol", "sum_open_interest"]
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["oi_time", "oi"]), missing
    oi = pd.concat(parts, ignore_index=True)
    oi = pd.DataFrame({"oi_time": pd.to_datetime(oi["create_time"], utc=True),
                       "oi": pd.to_numeric(oi["sum_open_interest"], errors="coerce")})
    oi = oi[oi["oi"] > 0].dropna().drop_duplicates("oi_time").sort_values("oi_time")
    return oi.reset_index(drop=True), missing


if __name__ == "__main__":
    symbols = [s.strip() for s in Path("universe/train32.txt").read_text().split() if s.strip()]
    print(f"{len(symbols)}종목")
    out = []
    for i, sym in enumerate(symbols, 1):
        k = download_klines(sym)
        if k.empty:
            print(f"[{i:2d}/{len(symbols)}] {sym}: 1시간봉 없음 — 빠짐")
            continue
        oi, missing = download_oi(sym, k["time"].iloc[0])
        lab = make_label(k)                   # 정답지 (BTC와 같은 규칙)
        feat = make_features(k, oi)           # 힌트 10개 (BTC와 같은 규칙)
        df = lab.merge(feat, on="time", how="left")
        df.insert(0, "symbol", sym)
        out.append(df)
        print(f"[{i:2d}/{len(symbols)}] {sym}: {len(df):,}시간, 장대봉 {df['big'].mean():.1%}, "
              f"OI 빈칸 {df['oi_1h'].isna().mean():.1%}" + (f" (OI 없는 날 {missing})" if missing else ""))

    all_df = pd.concat(out, ignore_index=True)
    all_df.to_csv(DATA / "multi_features.csv", index=False)
    print(f"\n합계 {all_df['symbol'].nunique()}종목, {len(all_df):,}시간, 장대봉 {all_df['big'].mean():.1%}, "
          f"장대 중 양봉 {all_df['big_up'].mean():.1%}")
