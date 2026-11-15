"""
c01_label.py — 다음 봉 장대봉 모델: 정답지 만들기 (BTC만, 1단계 최소 버전)

정답지 규칙 (2026-11-15 결정)
- 언제 묻나 : 매 1시간봉이 끝날 때 (지금 봉 = t)
- 무엇을    : 다음 봉(t+1)이 장대봉일까?
- 봉 크기   : 몸통 = |종가 - 시가|
- 평소      : t까지 끝난 직전 336봉(14일)의 몸통 평균  ← t+1은 절대 안 씀
- 장대봉    : 몸통(t+1) >= 평소(t) x 2
- 2단계용   : 장대봉이면 색 (종가 > 시가 = 양봉 1, 아니면 음봉 0)

실행: python c01_label.py
결과: data/btc_1h.csv (원본), data/c01_label.csv (정답지), 화면에 빈도 요약
"""
import time
from pathlib import Path

import pandas as pd
import requests

SYMBOL = "BTCUSDT"
START = "2023-10-15"      # 14일 '평소' 계산용 여유분 포함
END = "2026-11-01"        # 이 시각 전까지 (연구 07과 같은 기간 끝)
STUDY_START = "2023-11-01"
NORMAL_BARS = 336         # 14일 x 24봉
MULT = 2.0                # 평소의 2배 = 장대봉

DATA = Path("data")
DATA.mkdir(exist_ok=True)


def download(symbol: str) -> pd.DataFrame:
    """바이낸스 선물 1시간봉 다운로드 (이미 받았으면 파일 재사용)"""
    path = DATA / "btc_1h.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["time"])

    url = "https://fapi.binance.com/fapi/v1/klines"
    start = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp(END, tz="UTC").timestamp() * 1000)
    rows = []
    while start < end:
        r = requests.get(url, params={"symbol": symbol, "interval": "1h",
                                      "startTime": start, "endTime": end - 1,
                                      "limit": 1500}, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        start = batch[-1][0] + 3_600_000
        time.sleep(0.2)

    df = pd.DataFrame(rows).iloc[:, :5]
    df.columns = ["time", "open", "high", "low", "close"]
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def make_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["body"] = (df["close"] - df["open"]).abs()

    # ── 핵심: 평소 크기는 t까지 끝난 봉만으로 계산 (미래 정보 없음)
    df["normal"] = df["body"].rolling(NORMAL_BARS, min_periods=NORMAL_BARS).mean()

    # ── 다음 봉(t+1)의 정보 = 정답 (힌트로는 절대 쓰지 않음)
    nxt_body = df["body"].shift(-1)
    nxt_up = (df["close"].shift(-1) > df["open"].shift(-1)).astype(int)

    df["big"] = (nxt_body >= MULT * df["normal"]).astype(int)       # 1단계 정답
    df["big_up"] = nxt_up.where(df["big"] == 1)                      # 2단계 정답 (장대봉일 때만)
    df["next_ratio"] = nxt_body / df["normal"]                       # 참고: 평소 대비 몇 배

    # 평소 계산이 안 되는 앞부분, 다음 봉이 없는 마지막 봉 제외
    ok = df["normal"].notna() & nxt_body.notna()
    # 1시간이 비어 있는 곳(거래소 점검 등): 다음 봉이 정확히 1시간 뒤가 아니면 제외
    ok &= (df["time"].shift(-1) - df["time"]) == pd.Timedelta(hours=1)
    df = df[ok & (df["time"] >= pd.Timestamp(STUDY_START, tz="UTC"))]
    return df[["time", "open", "close", "body", "normal", "next_ratio", "big", "big_up"]]


def summary(lab: pd.DataFrame) -> None:
    print(f"\n[{SYMBOL}] {lab['time'].min():%Y-%m-%d} ~ {lab['time'].max():%Y-%m-%d}, {len(lab):,}시간")
    n_big = int(lab["big"].sum())
    print(f"장대봉(평소 x{MULT:g} 이상): {n_big:,}번 = {lab['big'].mean():.1%}")
    up = lab["big_up"].dropna()
    print(f"  그중 양봉 {up.mean():.1%} / 음봉 {1 - up.mean():.1%}")

    print("\n연도별 (12개월 단위, 11월 시작)")
    year = lab["time"].apply(lambda t: f"{t.year}-{t.year + 1}" if t.month >= 11 else f"{t.year - 1}-{t.year}")
    g = lab.groupby(year)
    out = pd.DataFrame({
        "시간": g.size(),
        "장대봉 비율": g["big"].mean().map("{:.1%}".format),
        "장대 중 양봉": g["big_up"].mean().map("{:.1%}".format),
    })
    print(out.to_string())

    print("\n참고 — 기준을 바꾸면 비율이 얼마나 되나 (바꾸려는 게 아니라 감 잡기용)")
    for m in [1.5, 2.0, 2.5, 3.0]:
        print(f"  x{m:g}: {(lab['next_ratio'] >= m).mean():.1%}")


if __name__ == "__main__":
    raw = download(SYMBOL)
    lab = make_label(raw)
    lab.to_csv(DATA / "c01_label.csv", index=False)
    summary(lab)
