"""
c07_color_mock.py — 2단계 모의고사 (32종목): "장대봉이라면 양봉일까 음봉일까?"

문제   : 실제로 다음 봉이 장대봉인 시간만 (big = 1)
정답   : big_up (양봉 1 / 음봉 0)
상대   : 찍기 — 매번 "공부 기간 양봉 비율"이라고 말함 (AUC는 항상 0.5)
학생   : 힌트 10개 + 시간대 로지스틱 (1단계와 같은 재료)
나누기 : 1단계와 같음 (공부 2023-11 ~ 2024-10 / 비움 2시간 / 시험 2024-11 ~ 2025-10)
합격   : ① AUC 95% 범위 아래 끝 > 0.5
         ② 향상도(찍기 대비 Brier 개선) 95% 범위 아래 끝 > 0
         날 단위 부트스트랩 500회, 그날 모든 코인 통째로   (점수 보기 전 확정, 2026-11-15)

필요: data/multi_features.csv (c05), 같은 폴더의 c03_mock.py, c04_hour_check.py
실행: python c07_color_mock.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from c03_mock import ALL, DATA, GAP, N_BOOT, SEED, TEST_END, TEST_START, TRAIN_START
from c04_hour_check import HOURS, add_hours

COLS = ALL + HOURS


def fit_predict(train, test, cols, target):
    """공부 기간으로만 빈칸 채우기·표준화·학습"""
    fill = train[cols].median()
    Xtr, Xte = train[cols].fillna(fill), test[cols].fillna(fill)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), train[target])
    return m.predict_proba(sc.transform(Xte))[:, 1], dict(zip(cols, m.coef_[0].round(3)))


def skill(y, p, guess):
    return 1 - np.mean((p - y) ** 2) / np.mean((guess - y) ** 2)


def bootstrap(test, y, p, guess):
    rng = np.random.default_rng(SEED)
    day = test["time"].dt.floor("D").to_numpy()
    days = np.unique(day)
    idx_by_day = [np.flatnonzero(day == d) for d in days]
    aucs, sks = [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_by_day[i] for i in rng.integers(0, len(days), len(days))])
        aucs.append(roc_auc_score(y[idx], p[idx]))
        sks.append(skill(y[idx], p[idx], guess))
    return np.percentile(aucs, [2.5, 97.5]), np.percentile(sks, [2.5, 97.5])


if __name__ == "__main__":
    df = add_hours(pd.read_csv(DATA / "multi_features.csv", parse_dates=["time"]))
    df = df[df["big"] == 1]                       # 장대봉이 나온 시간만
    train = df[(df["time"] >= TRAIN_START) & (df["time"] < TEST_START - GAP)].reset_index(drop=True)
    test = df[(df["time"] >= TEST_START) & (df["time"] < TEST_END)].reset_index(drop=True)
    guess = train["big_up"].mean()                # 찍기 = 공부 기간 양봉 비율
    y = test["big_up"].to_numpy().astype(int)

    print(f"\n공부 장대봉 {len(train):,}개 (양봉 {guess:.1%}) / 시험 장대봉 {len(test):,}개 (양봉 {y.mean():.1%})")

    p, coef = fit_predict(train, test, COLS, "big_up")
    auc = roc_auc_score(y, p)
    sk = skill(y, p, guess)
    ci_auc, ci_sk = bootstrap(test, y, p, guess)
    ok_auc, ok_sk = ci_auc[0] > 0.5, ci_sk[0] > 0
    passed = bool(ok_auc and ok_sk)

    print(f"\n{'예보관':<12}{'향상도':>8}{'AUC':>8}")
    print(f"{'찍기 (상대)':<12}{0:>8.1%}{0.5:>8.3f}")
    print(f"{'학생':<12}{sk:>8.2%}{auc:>8.3f}")
    print(f"\n판정 (날 단위 부트스트랩 {N_BOOT}회 95% 범위)")
    print(f"  ① AUC    {auc:.3f}  ({ci_auc[0]:.3f} ~ {ci_auc[1]:.3f})  기준 > 0.5  {'✓' if ok_auc else '✗'}")
    print(f"  ② 향상도 {sk:+.2%}  ({ci_sk[0]:+.2%} ~ {ci_sk[1]:+.2%})  기준 > 0   {'✓' if ok_sk else '✗'}")
    print(f"\n판정: {'합격' if passed else '불합격'}")

    # ── 참고 (판정 아님)
    hit = ((p >= 0.5) == (y == 1)).mean()
    print(f"\n참고: 색 맞힌 비율 (확률 50% 기준) {hit:.1%}  (찍기로 '항상 {'양봉' if guess >= 0.5 else '음봉'}'이면 "
          f"{max(y.mean(), 1 - y.mean()):.1%})")
    print("참고: 학생이 가장 자신 있던 10%")
    q_hi, q_lo = np.quantile(p, [0.9, 0.1])
    print(f"  '양봉' 자신 (확률 ≥ {q_hi:.1%}): 실제 양봉 {y[p >= q_hi].mean():.1%} ({(p >= q_hi).sum():,}개)")
    print(f"  '음봉' 자신 (확률 ≤ {q_lo:.1%}): 실제 음봉 {1 - y[p <= q_lo].mean():.1%} ({(p <= q_lo).sum():,}개)")

    seen = set(train["symbol"])
    new = ~test["symbol"].isin(seen).to_numpy()
    for label, m in [("본 코인", ~new), ("처음 보는 코인", new)]:
        if m.sum() > 50 and 0 < y[m].mean() < 1:
            print(f"참고: {label} AUC {roc_auc_score(y[m], p[m]):.3f}, 향상도 {skill(y[m], p[m], guess):+.2%} ({m.sum():,}개)")

    print("\n학생이 본 것 (표준화 계수, +는 양봉 쪽 / −는 음봉 쪽)")
    for k, v in sorted(((k, v) for k, v in coef.items() if k in ALL), key=lambda kv: -abs(kv[1])):
        print(f"  {k:<10}{v:+.3f}")

    out = {"train_n": len(train), "test_n": len(test), "guess": guess, "auc": auc, "skill": sk,
           "auc_ci": ci_auc.tolist(), "skill_ci": ci_sk.tolist(), "passed": passed,
           "coef": {k: v for k, v in coef.items() if k in ALL}}
    (DATA / "c07_color_mock.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))
