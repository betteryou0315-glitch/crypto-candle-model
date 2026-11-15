"""
c04_hour_check.py — 의심 확인: 학생이 배운 게 "사람이 몰렸다"인가, "지금 몇 시다"인가? (BTC 모의고사)

방법 (점수 보기 전 확정, 2026-11-15)
- 양쪽 모두에 시간대(UTC 0~23시, 24칸 원-핫)를 똑같이 줌
    상대  = 변동성 2개 + 시간대
    학생  = 힌트 10개 + 시간대
- 기준은 c03과 동일: ① AUC 차이, ② 향상도 차이 둘 다 날 단위 부트스트랩 95% 범위 아래 끝 > 0
- 해석
    학생 여전히 합격 → 거래량·OI는 "몇 시"를 빼고도 남는 진짜 정보
    학생 불합격     → c03의 승리는 대부분 시간대 효과였음
- 모의고사를 보고 생긴 의심이라 "변경 1회"로 기록

필요: data/c02_features.csv, 같은 폴더의 c03_mock.py
실행: python c04_hour_check.py
"""
import json

import numpy as np
import pandas as pd

from c03_mock import (ALL, BASE, DATA, GAP, TEST_END, TEST_START, TRAIN_START,
                      bootstrap, fit_predict, honesty, scores)

HOURS = [f"h{h:02d}" for h in range(1, 24)]   # 0시는 기준칸 (나머지 23칸이 0시 대비 차이)


def add_hours(df: pd.DataFrame) -> pd.DataFrame:
    """방금 끝난 봉(t)의 시각 — 다음 봉 시각은 +1시간이라 정보는 같음. 미래 정보 아님"""
    h = df["time"].dt.hour
    for i in range(1, 24):
        df[f"h{i:02d}"] = (h == i).astype(int)
    return df


if __name__ == "__main__":
    df = add_hours(pd.read_csv(DATA / "c02_features.csv", parse_dates=["time"]))
    train = df[(df["time"] >= TRAIN_START) & (df["time"] < TEST_START - GAP)]
    test = df[(df["time"] >= TEST_START) & (df["time"] < TEST_END)].reset_index(drop=True)
    base_rate = train["big"].mean()
    y = test["big"].to_numpy()

    # 참고: 공부 기간에서만 — 시간대별 장대봉 비율 (다음 봉 시각 기준, UTC / KST)
    nxt = (train["time"].dt.hour + 1) % 24
    by_h = train.groupby(nxt)["big"].mean()
    top = by_h.sort_values(ascending=False).head(4)
    low = by_h.sort_values().head(3)
    fmt = lambda s: ", ".join(f"{h:02d}시 UTC({(h + 9) % 24:02d}시 KST) {v:.1%}" for h, v in s.items())
    print(f"\n[공부 기간만] 장대봉이 잘 나오는 다음 봉 시각: {fmt(top)}")
    print(f"             잘 안 나오는 시각: {fmt(low)}   (전체 {base_rate:.1%})")

    runs = {
        "변동성": BASE,
        "변동성+시간": BASE + HOURS,
        "학생": ALL,
        "학생+시간": ALL + HOURS,
    }
    p, res = {}, {}
    for name, cols in runs.items():
        p[name], coef = fit_predict(train, test, cols)
        auc, sk = scores(y, p[name], base_rate)
        res[name] = {"auc": auc, "skill": sk, "coef": {k: v for k, v in coef.items() if k in ALL}}

    print(f"\n{'예보관':<14}{'향상도':>8}{'AUC':>8}")
    for name in runs:
        print(f"{name:<14}{res[name]['skill']:>8.1%}{res[name]['auc']:>8.3f}")

    # 판정: 시간대를 똑같이 준 상태에서 학생 vs 상대
    ci_auc, ci_sk = bootstrap(test, p["변동성+시간"], p["학생+시간"], base_rate)
    d_auc = res["학생+시간"]["auc"] - res["변동성+시간"]["auc"]
    d_sk = res["학생+시간"]["skill"] - res["변동성+시간"]["skill"]
    ok_auc, ok_sk = ci_auc[0] > 0, ci_sk[0] > 0
    passed = bool(ok_auc and ok_sk)

    print("\n판정: 학생+시간 - 변동성+시간 (날 단위 부트스트랩 500회 95% 범위)")
    print(f"  ① AUC    {d_auc:+.3f}  ({ci_auc[0]:+.3f} ~ {ci_auc[1]:+.3f})  {'✓' if ok_auc else '✗'}")
    print(f"  ② 향상도 {d_sk:+.2%}  ({ci_sk[0]:+.2%} ~ {ci_sk[1]:+.2%})  {'✓' if ok_sk else '✗'}")
    print(f"\n→ {'합격: 거래량·OI는 시간대를 빼고도 남는 정보' if passed else '불합격: c03의 승리는 대부분 시간대 효과'}")

    print("\n시간대를 줘도 거래량을 계속 믿나? (표준화 계수: 학생 → 학생+시간)")
    for k in ["volu_1h", "volu_4h", "vol_4h", "vol_24h", "oi_24h", "oi_1h"]:
        print(f"  {k:<9}{res['학생']['coef'][k]:+.3f} → {res['학생+시간']['coef'][k]:+.3f}")

    print("\n정직함 (말한 확률 → 실제 비율)")
    honesty(y, p["변동성+시간"], "변동성+시간")
    honesty(y, p["학생+시간"], "학생+시간  ")

    out = {"results": res, "diff_auc_ci": ci_auc.tolist(), "diff_skill_ci": ci_sk.tolist(),
           "passed": passed}
    (DATA / "c04_hour_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))
