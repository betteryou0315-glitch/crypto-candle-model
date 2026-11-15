"""
c09_storm_mock.py — 연구 10 모의고사: 폭풍 속 돌파 따라가기 (32종목)

규칙 (연구 노트 10, 2026-11-15 확정)
1 폭풍 예보 : 연구 09 1단계 모델(힌트 10개 + 시간대)을 공부 기간으로 학습 →
              공부 기간 예측 확률의 상위 10% 기준선 이상인 시간 = 폭풍
2 출발 신호 : 폭풍 다음 1시간봉의 시가에서 5분봉 종가가 평소 몸통 x 0.5 이상 벌어지면
              그 방향으로, 그 5분봉 종가에 진입 (한 시간에 한 번, 먼저 닿은 쪽)
3 나가기    : 그 1시간봉 종가(마지막 5분봉 종가)에 청산
4 합격      : 비용(왕복 0.14%) 후 거래당 평균 > 0, 날 단위 부트스트랩 500회 95% 아래 끝 > 0

필요: data/multi_features.csv (c05), data/5m/ (c08), c03_mock.py, c04_hour_check.py
실행: python c09_storm_mock.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from c03_mock import ALL, DATA, GAP, N_BOOT, SEED, TEST_END, TEST_START, TRAIN_START, fit_predict
from c04_hour_check import HOURS, add_hours

TOP = 0.10        # 폭풍 = 상위 10%
K = 0.5           # 출발 = 평소 몸통 x 0.5
COST = 0.0014     # 왕복 비용
BIG_COINS = {"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"}   # 참고용 '큰 코인' (미리 고정)


def load_5m(symbol: str) -> pd.DataFrame:
    """5분봉 → 1시간 단위 표: 행 = 1시간봉 시작 시각, 열 = 그 안의 5분봉 12개 (0~11)"""
    files = sorted((DATA / "5m" / symbol).glob("*.csv"))
    if not files:
        return None, None
    m = pd.concat([pd.read_csv(f, parse_dates=["time"]) for f in files], ignore_index=True)
    m = m.drop_duplicates("time").sort_values("time")
    m["h"] = m["time"].dt.floor("h")
    m["k"] = m["time"].dt.minute // 5
    close = m.pivot(index="h", columns="k", values="close").reindex(columns=range(12))
    open0 = m[m["k"] == 0].set_index("h")["open"]
    return close, open0


def simulate(rows: pd.DataFrame, close: pd.DataFrame, open0: pd.Series) -> pd.DataFrame:
    """rows: 신호를 판단한 1시간봉 t (time, normal). 거래는 그 다음 봉 t+1 안에서 일어남"""
    trade_h = rows["time"] + pd.Timedelta(hours=1)
    C = close.reindex(trade_h).to_numpy()                 # (n, 12)
    O = open0.reindex(trade_h).to_numpy()                 # (n,)
    thr = K * rows["normal"].to_numpy()
    ok = ~np.isnan(C).any(axis=1) & ~np.isnan(O)          # 5분봉 12개가 다 있어야 함

    dev = C - O[:, None]
    hit = (np.abs(dev) >= thr[:, None]) & ok[:, None]
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1), -1)

    n = len(rows)
    side = np.zeros(n)
    entry = np.full(n, np.nan)
    idx = np.flatnonzero(any_hit)
    side[idx] = np.sign(dev[idx, first[idx]])             # +1 롱 / -1 숏
    entry[idx] = C[idx, first[idx]]
    exit_ = C[:, 11]
    gross = side * (exit_ - entry) / entry

    return pd.DataFrame({
        "symbol": rows["symbol"].to_numpy(), "time": trade_h.to_numpy(),
        "ok": ok, "traded": any_hit, "side": side, "entry_k": first,
        "gross": gross, "net": gross - COST, "big": rows["big"].to_numpy(),
    })


def boot_mean(tr: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    day = pd.to_datetime(tr["time"]).dt.floor("D").to_numpy()
    days = np.unique(day)
    by_day = [np.flatnonzero(day == d) for d in days]
    x = tr["net"].to_numpy()
    means = [x[np.concatenate([by_day[i] for i in rng.integers(0, len(days), len(days))])].mean()
             for _ in range(N_BOOT)]
    return np.percentile(means, [2.5, 97.5])


def describe(tr: pd.DataFrame, label: str) -> None:
    if len(tr) == 0:
        print(f"  {label:<14}거래 없음")
        return
    w, l = tr[tr["net"] > 0]["net"], tr[tr["net"] <= 0]["net"]
    print(f"  {label:<14}{len(tr):>7,}번  비용 전 {tr['gross'].mean():+.3%}  비용 후 {tr['net'].mean():+.3%}  "
          f"승률 {len(w) / len(tr):.1%}  평균 이익 {w.mean() if len(w) else 0:+.2%} / 손실 {l.mean() if len(l) else 0:+.2%}")


if __name__ == "__main__":
    df = add_hours(pd.read_csv(DATA / "multi_features.csv", parse_dates=["time"]))
    train = df[(df["time"] >= TRAIN_START) & (df["time"] < TEST_START - GAP)].reset_index(drop=True)
    test = df[(df["time"] >= TEST_START) & (df["time"] < TEST_END)].reset_index(drop=True)

    # 1. 폭풍 예보 — 공부 기간으로만 학습·기준선
    cols = ALL + HOURS
    p_test, _ = fit_predict(train, test, cols)
    p_train, _ = fit_predict(train, train, cols)
    line = np.quantile(p_train, 1 - TOP)
    test["storm"] = p_test >= line
    print(f"\n폭풍 기준선 (공부 기간 상위 {TOP:.0%}): 장대봉 확률 ≥ {line:.1%}")
    print(f"시험 기간 폭풍 {test['storm'].sum():,}시간 ({test['storm'].mean():.1%}) — "
          f"실제 장대봉 {test.loc[test['storm'], 'big'].mean():.1%} (폭풍 아님 {test.loc[~test['storm'], 'big'].mean():.1%})")

    # 2~3. 5분봉으로 거래 흉내
    # 거래는 다음 봉에서 일어나므로, 거래 시각이 수능 기간(TEST_END 이후)으로 넘어가는 마지막 시간은 뺌
    sim_rows = test[test["time"] + pd.Timedelta(hours=1) < TEST_END]
    parts = []
    for sym, rows in sim_rows.groupby("symbol"):
        close, open0 = load_5m(sym)
        if close is None:
            continue
        sim = simulate(rows.reset_index(drop=True), close, open0)
        sim["storm"] = rows["storm"].to_numpy()
        parts.append(sim)
    sim = pd.concat(parts, ignore_index=True)

    storm = sim[sim["storm"] & sim["ok"]]
    trades = storm[storm["traded"]].reset_index(drop=True)
    print(f"\n폭풍 시간 중 5분봉 다 있는 시간 {len(storm):,} → 출발 신호 {len(trades):,}번 "
          f"(출발 없음 {1 - len(trades) / max(len(storm), 1):.1%})")

    # 4. 판정
    ci = boot_mean(trades)
    mean = trades["net"].mean()
    passed = bool(ci[0] > 0)
    print("\n판정: 비용 후 거래당 평균 (날 단위 부트스트랩 500회 95% 범위)")
    print(f"  {mean:+.3%}  ({ci[0]:+.3%} ~ {ci[1]:+.3%})  기준: 아래 끝 > 0  {'✓' if passed else '✗'}")
    print(f"\n판정: {'합격' if passed else '불합격'}")

    # 참고 (판정 아님)
    print("\n참고 (판정 아님)")
    calm = sim[~sim["storm"] & sim["ok"] & sim["traded"]]
    describe(trades, "폭풍 시간")
    describe(calm, "폭풍 아님")
    describe(trades[trades["side"] > 0], "  └ 롱")
    describe(trades[trades["side"] < 0], "  └ 숏")
    describe(trades[trades["symbol"].isin(BIG_COINS)], "  └ 큰 코인")
    describe(trades[~trades["symbol"].isin(BIG_COINS)], "  └ 알트")
    describe(trades[trades["big"] == 1], "  └ 진짜 장대봉")
    describe(trades[trades["big"] == 0], "  └ 장대봉 아님")
    print("\n  들어간 시점(5분봉 몇 번째)별 비용 후 평균")
    for k, g in trades.groupby("entry_k"):
        print(f"    {int(k) * 5:>2}~{int(k) * 5 + 5:>2}분: {len(g):>6,}번  {g['net'].mean():+.3%}")

    out = {"line": float(line), "storm_hours": int(len(storm)), "trades": int(len(trades)),
           "mean_net": float(mean), "ci": ci.tolist(), "passed": passed}
    (DATA / "c09_storm_mock.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
