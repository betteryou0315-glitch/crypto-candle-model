"""
c02_features.py — 다음 봉 장대봉 모델: 단서(힌트) 만들기 (BTC만)

단서 (2026-11-15 결정) — 모두 지금 봉(t)이 끝난 시각까지 나온 정보만 사용
  가격   vol_4h, vol_24h   최근 4·24봉 몸통 평균 ÷ 평소 몸통        (요즘 얼마나 출렁이나)
         ret_1h, ret_4h, ret_24h  최근 1·4·24봉 가격 변화 ÷ 평소 몸통 (어느 쪽으로 얼마나 갔나)
  거래량 volu_1h, volu_4h  지금 봉 / 최근 4봉 평균 거래량 ÷ 직전 14일 평균 거래량
         taker_buy         지금 봉 시장가 매수 거래량 ÷ 전체 거래량 (0.5 초과 = 사는 쪽이 급함)
  OI     oi_1h, oi_24h     OI(코인 개수) 1·24시간 변화율 — 연구 07과 같은 정의
                           (봉 마감 시각 기준 30분 넘게 끊긴 OI는 빈칸)

필요: 같은 폴더의 data/c01_label.csv (c01_label.py 먼저 실행)
실행: python c02_features.py
결과: data/c02_features.csv, 화면에 단서별 요약
첫 실행은 OI 기록(하루 파일 약 1,100개)을 받느라 10분 안팎 걸림 — 받은 파일은 data/metrics/ 에 저장돼 다음부턴 바로 끝남
"""
import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

SYMBOL = "BTCUSDT"
START = "2023-10-14"      # 14일 평소 + 24시간 변화 계산용 여유분
END = "2026-11-01"
NORMAL_BARS = 336         # 14일
OI_TOLERANCE = pd.Timedelta(minutes=30)

DATA = Path("data")
MET = DATA / "metrics"
MET.mkdir(parents=True, exist_ok=True)


# ───────────────────────── 데이터 받기 ─────────────────────────
def download_klines() -> pd.DataFrame:
    """1시간봉 + 거래량 + 시장가 매수 거래량 (c01 파일엔 거래량이 없어서 다시 받음)"""
    path = DATA / "btc_1h_full.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["time"])

    url = "https://fapi.binance.com/fapi/v1/klines"
    start = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp(END, tz="UTC").timestamp() * 1000)
    rows = []
    while start < end:
        r = requests.get(url, params={"symbol": SYMBOL, "interval": "1h",
                                      "startTime": start, "endTime": end - 1,
                                      "limit": 1500}, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        start = batch[-1][0] + 3_600_000
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    df = pd.DataFrame({
        "time": pd.to_datetime(df[0], unit="ms", utc=True),
        "open": df[1].astype(float), "close": df[4].astype(float),
        "volume": df[5].astype(float),          # 거래량 (코인 개수)
        "taker_buy_vol": df[9].astype(float),   # 그중 시장가 매수
    })
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def download_oi() -> pd.DataFrame:
    """바이낸스 공개 데이터(data.binance.vision)의 5분 OI 기록, 하루 파일씩"""
    base = f"https://data.binance.vision/data/futures/um/daily/metrics/{SYMBOL}"
    days = pd.date_range(START, pd.Timestamp(END) - pd.Timedelta(days=1), freq="D")
    parts, missing = [], 0
    for i, d in enumerate(days):
        name = f"{SYMBOL}-metrics-{d:%Y-%m-%d}"
        csv = MET / f"{name}.csv"
        if not csv.exists():
            r = requests.get(f"{base}/{name}.zip", timeout=30)
            if r.status_code == 404:
                missing += 1
                continue
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                csv.write_bytes(z.read(z.namelist()[0]))
            time.sleep(0.05)
        part = pd.read_csv(csv)
        if "create_time" not in part.columns:   # 머리줄 없는 옛 파일 대비
            part = pd.read_csv(csv, header=None).iloc[:, :3]
            part.columns = ["create_time", "symbol", "sum_open_interest"]
        parts.append(part)
        if (i + 1) % 100 == 0:
            print(f"  OI 기록 {i + 1}/{len(days)}일")
    if missing:
        print(f"  (OI 기록 없는 날 {missing}일 — 그 시간은 빈칸)")

    oi = pd.concat(parts, ignore_index=True)
    oi = pd.DataFrame({
        "oi_time": pd.to_datetime(oi["create_time"], utc=True),
        "oi": oi["sum_open_interest"].astype(float),   # 코인 개수
    })
    oi = oi[oi["oi"] > 0]   # 0으로 찍힌 잘못된 기록은 버림 (없는 것으로 취급 → 빈칸)
    return oi.dropna().drop_duplicates("oi_time").sort_values("oi_time").reset_index(drop=True)


# ───────────────────────── 단서 만들기 ─────────────────────────
def oi_at(times: pd.Series, oi: pd.DataFrame) -> pd.Series:
    """각 시각 '이전'의 마지막 OI 기록 (30분 넘게 끊겼으면 빈칸) — 미래 OI는 절대 안 씀"""
    left = pd.DataFrame({"t": pd.to_datetime(times, utc=True).astype("datetime64[ns, UTC]")}).reset_index(drop=True)
    left = left.reset_index()
    oi = oi.assign(oi_time=oi["oi_time"].astype("datetime64[ns, UTC]"))
    m = pd.merge_asof(left.sort_values("t"), oi, left_on="t", right_on="oi_time",
                      direction="backward", tolerance=OI_TOLERANCE)
    return m.sort_values("index")["oi"].to_numpy()


def make_features(k: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
    k = k.copy()
    body = (k["close"] - k["open"]).abs()
    normal = body.rolling(NORMAL_BARS, min_periods=NORMAL_BARS).mean()   # c01과 같은 '평소'

    # 가격 — 변동성
    k["vol_4h"] = body.rolling(4).mean() / normal
    k["vol_24h"] = body.rolling(24).mean() / normal
    # 가격 — 방향 (평소 몸통 몇 개만큼 움직였나)
    for n in (1, 4, 24):
        k[f"ret_{n}h"] = (k["close"] - k["close"].shift(n)) / normal

    # 거래량 — '평소'는 지금 봉을 뺀 직전 336봉 평균
    vol_normal = k["volume"].shift(1).rolling(NORMAL_BARS, min_periods=NORMAL_BARS).mean()
    k["volu_1h"] = k["volume"] / vol_normal
    k["volu_4h"] = k["volume"].rolling(4).mean() / vol_normal
    k["taker_buy"] = (k["taker_buy_vol"] / k["volume"]).where(k["volume"] > 0)

    # OI — 봉 마감 시각(t + 1시간) 기준
    close_time = k["time"] + pd.Timedelta(hours=1)
    oi_now = oi_at(close_time, oi)
    oi_1h_ago = oi_at(close_time - pd.Timedelta(hours=1), oi)
    oi_24h_ago = oi_at(close_time - pd.Timedelta(hours=24), oi)
    k["oi_1h"] = oi_now / oi_1h_ago - 1
    k["oi_24h"] = oi_now / oi_24h_ago - 1

    cols = ["vol_4h", "vol_24h", "ret_1h", "ret_4h", "ret_24h",
            "volu_1h", "volu_4h", "taker_buy", "oi_1h", "oi_24h"]
    return k[["time"] + cols]


def summary(df: pd.DataFrame, cols: list[str]) -> None:
    print(f"\n[{SYMBOL}] {len(df):,}시간")
    print("\n단서별 요약 (전체 기간 — 분포·빈칸 확인용, 정답과의 관계는 아직 안 봄)")
    out = pd.DataFrame({
        "빈칸": df[cols].isna().mean().map("{:.1%}".format),
        "중앙값": df[cols].median().round(3),
        "하위 5%": df[cols].quantile(0.05).round(3),
        "상위 5%": df[cols].quantile(0.95).round(3),
    })
    print(out.to_string())


if __name__ == "__main__":
    label = pd.read_csv(DATA / "c01_label.csv", parse_dates=["time"])
    print("1시간봉 받는 중...")
    k = download_klines()
    print("OI 기록 받는 중...")
    oi = download_oi()

    feat = make_features(k, oi)
    df = label.merge(feat, on="time", how="left")   # 정답지에 있는 시간만
    cols = [c for c in feat.columns if c != "time"]
    df.to_csv(DATA / "c02_features.csv", index=False)
    summary(df, cols)
