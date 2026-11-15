"""
c06_multi_mock.py — 32종목 1단계 모의고사

질문   : 다음 1시간봉이 장대봉(몸통 >= 그 코인 평소 x2)일까?
상대   : 변동성 2개 + 시간대
학생   : 힌트 10개 + 시간대
나누기 : 공부 2023-11 ~ 2024-10 / 비움 2시간 / 시험 2024-11 ~ 2025-10 (BTC와 같음)
합격   : ① AUC 차이, ② 향상도 차이 — 둘 다 날 단위 부트스트랩 500회 95% 범위 아래 끝 > 0
         날 단위 = 그날의 모든 코인 시간을 통째로 뽑음   (점수 보기 전 확정, 2026-11-15)

필요: data/multi_features.csv (c05), 같은 폴더의 c03_mock.py, c04_hour_check.py
실행: python c06_multi_mock.py   (부트스트랩 때문에 1~3분)
"""
import json

import pandas as pd

from c03_mock import (ALL, BASE, DATA, GAP, TEST_END, TEST_START, TRAIN_START,
                      bootstrap, fit_predict, honesty, scores)
from c04_hour_check import HOURS, add_hours

if __name__ == "__main__":
    df = add_hours(pd.read_csv(DATA / "multi_features.csv", parse_dates=["time"]))
    train = df[(df["time"] >= TRAIN_START) & (df["time"] < TEST_START - GAP)].reset_index(drop=True)
    test = df[(df["time"] >= TEST_START) & (df["time"] < TEST_END)].reset_index(drop=True)
    base_rate = train["big"].mean()
    y = test["big"].to_numpy()

    seen = set(train["symbol"])
    is_new = ~test["symbol"].isin(seen)
    print(f"\n공부 {len(train):,}시간 · {train['symbol'].nunique()}코인 (장대봉 {base_rate:.1%})")
    print(f"시험 {len(test):,}시간 · {test['symbol'].nunique()}코인 (장대봉 {y.mean():.1%}), "
          f"그중 처음 보는 코인 {test.loc[is_new, 'symbol'].nunique()}개 · {is_new.sum():,}시간")

    runs = {"변동성": BASE, "학생": ALL, "변동성+시간": BASE + HOURS, "학생+시간": ALL + HOURS}
    p, res = {}, {}
    for name, cols in runs.items():
        p[name], coef = fit_predict(train, test, cols)
        auc, sk = scores(y, p[name], base_rate)
        res[name] = {"auc": auc, "skill": sk, "coef": {k: v for k, v in coef.items() if k in ALL}}

    print(f"\n{'예보관':<14}{'향상도':>8}{'AUC':>8}")
    for name in runs:
        mark = "  ← 판정 대상" if name in ("변동성+시간", "학생+시간") else ""
        print(f"{name:<14}{res[name]['skill']:>8.1%}{res[name]['auc']:>8.3f}{mark}")

    # ── 판정
    ci_auc, ci_sk = bootstrap(test, p["변동성+시간"], p["학생+시간"], base_rate)
    d_auc = res["학생+시간"]["auc"] - res["변동성+시간"]["auc"]
    d_sk = res["학생+시간"]["skill"] - res["변동성+시간"]["skill"]
    ok_auc, ok_sk = ci_auc[0] > 0, ci_sk[0] > 0
    passed = bool(ok_auc and ok_sk)
    print("\n판정: 학생+시간 - 변동성+시간 (날 단위 부트스트랩 500회 95% 범위)")
    print(f"  ① AUC    {d_auc:+.3f}  ({ci_auc[0]:+.3f} ~ {ci_auc[1]:+.3f})  {'✓' if ok_auc else '✗'}")
    print(f"  ② 향상도 {d_sk:+.2%}  ({ci_sk[0]:+.2%} ~ {ci_sk[1]:+.2%})  {'✓' if ok_sk else '✗'}")
    print(f"\n판정: {'합격' if passed else '불합격'}")

    # ── 참고 (판정 아님)
    print("\n참고: 본 코인 / 처음 보는 코인 따로 (AUC, 향상도)")
    for label, mask in [("본 코인", ~is_new), ("처음 보는 코인", is_new)]:
        if mask.sum() == 0 or y[mask].sum() == 0:
            print(f"  {label}: 없음")
            continue
        row = []
        for name in ("변동성+시간", "학생+시간"):
            a, s = scores(y[mask], p[name][mask.to_numpy()], base_rate)
            row.append(f"{name} {a:.3f} / {s:+.1%}")
        print(f"  {label:<8}: " + "   ".join(row))

    print("\n학생+시간이 본 것 (표준화 계수)")
    for k, v in sorted(res["학생+시간"]["coef"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:<10}{v:+.3f}")

    print("\n정직함 (말한 확률 → 실제 비율)")
    honesty(y, p["변동성+시간"], "변동성+시간")
    honesty(y, p["학생+시간"], "학생+시간  ")

    print(f"\n2단계 준비: 장대봉 개수 — 공부 {int(train['big'].sum()):,} / 시험 {int(test['big'].sum()):,}")

    out = {"train_hours": len(train), "test_hours": len(test), "base_rate": base_rate,
           "results": res, "diff_auc_ci": ci_auc.tolist(), "diff_skill_ci": ci_sk.tolist(),
           "passed": passed}
    (DATA / "c06_multi_mock.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))
