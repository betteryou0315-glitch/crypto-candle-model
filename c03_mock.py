"""
c03_mock.py — 다음 봉 장대봉 모델 1단계: 모의고사 (BTC만)

질문   : 다음 1시간봉이 장대봉(몸통 >= 평소 x2)일까?
상대   : 변동성 예보관 — vol_4h, vol_24h 로지스틱
학생   : 단서 10개 로지스틱
나누기 : 공부 2023-11 ~ 2024-10 / 비움 2시간 / 시험 2024-11 ~ 2025-10
합격   : 학생 - 상대의 ① AUC 차이, ② 향상도 차이 — 둘 다
         날 단위 부트스트랩 500회 95% 범위 아래 끝 > 0   (점수 보기 전 확정, 2026-11-15)

필요: data/c02_features.csv (c02_features.py 먼저 실행)
실행: python c03_mock.py
결과: 화면 + data/c03_mock.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA = Path("data")
TRAIN_START = pd.Timestamp("2023-11-01", tz="UTC")
TEST_START = pd.Timestamp("2024-11-01", tz="UTC")
TEST_END = pd.Timestamp("2025-11-01", tz="UTC")
GAP = pd.Timedelta(hours=2)
N_BOOT = 500
SEED = 42

BASE = ["vol_4h", "vol_24h"]
ALL = ["vol_4h", "vol_24h", "ret_1h", "ret_4h", "ret_24h",
       "volu_1h", "volu_4h", "taker_buy", "oi_1h", "oi_24h"]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    """공부 기간으로만 표준화·빈칸 채우기·학습 → 시험 기간 예측 (시험 정보는 전혀 안 씀)"""
    fill = train[cols].median()
    Xtr, Xte = train[cols].fillna(fill), test[cols].fillna(fill)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), train["big"])
    coef = dict(zip(cols, m.coef_[0].round(3)))
    return m.predict_proba(sc.transform(Xte))[:, 1], coef


def scores(y, p, base_rate):
    """AUC, 향상도 = '매번 공부 기간 평균으로 찍기' 대비 Brier 개선 비율"""
    brier = np.mean((p - y) ** 2)
    brier_guess = np.mean((base_rate - y) ** 2)
    return roc_auc_score(y, p), 1 - brier / brier_guess


def bootstrap(test, p_base, p_stu, base_rate):
    """날 단위로 다시 뽑기: 같은 날의 시간들은 통째로 같이 뽑힘 (붙어 있는 시간끼리 닮아서)"""
    rng = np.random.default_rng(SEED)
    day = test["time"].dt.floor("D").to_numpy()
    days = np.unique(day)
    idx_by_day = [np.flatnonzero(day == d) for d in days]
    y = test["big"].to_numpy()
    d_auc, d_skill = [], []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(days), len(days))
        idx = np.concatenate([idx_by_day[i] for i in pick])
        a1, s1 = scores(y[idx], p_base[idx], base_rate)
        a2, s2 = scores(y[idx], p_stu[idx], base_rate)
        d_auc.append(a2 - a1)
        d_skill.append(s2 - s1)
    return np.percentile(d_auc, [2.5, 97.5]), np.percentile(d_skill, [2.5, 97.5])


def honesty(y, p, name):
    """정직함: 예보 확률을 5칸으로 나눠서 '말한 확률' vs '실제 비율'"""
    bins = pd.qcut(p, 5, labels=False, duplicates="drop")
    t = pd.DataFrame({"말함": p, "실제": y, "칸": bins}).groupby("칸").mean()
    pairs = ", ".join(f"{a:.1%}→{b:.1%}" for a, b in zip(t["말함"], t["실제"]))
    print(f"  {name}: {pairs}")


if __name__ == "__main__":
    df = pd.read_csv(DATA / "c02_features.csv", parse_dates=["time"])
    # 공부: 다음 봉(정답)이 시험 시작 전에 끝나도록 2시간 비움
    train = df[(df["time"] >= TRAIN_START) & (df["time"] < TEST_START - GAP)]
    test = df[(df["time"] >= TEST_START) & (df["time"] < TEST_END)].reset_index(drop=True)
    base_rate = train["big"].mean()

    p_base, coef_base = fit_predict(train, test, BASE)
    p_stu, coef_stu = fit_predict(train, test, ALL)

    y = test["big"].to_numpy()
    auc_b, sk_b = scores(y, p_base, base_rate)
    auc_s, sk_s = scores(y, p_stu, base_rate)
    ci_auc, ci_sk = bootstrap(test, p_base, p_stu, base_rate)
    ok_auc, ok_sk = ci_auc[0] > 0, ci_sk[0] > 0
    passed = bool(ok_auc and ok_sk)

    print(f"\n공부 {len(train):,}시간 (장대봉 {base_rate:.1%}) / 시험 {len(test):,}시간 (장대봉 {y.mean():.1%})")
    print(f"\n{'예보관':<14}{'향상도':>8}{'AUC':>8}")
    print(f"{'변동성 (상대)':<14}{sk_b:>8.1%}{auc_b:>8.3f}")
    print(f"{'학생 (10개)':<14}{sk_s:>8.1%}{auc_s:>8.3f}")
    print(f"\n차이 (학생 - 상대), 날 단위 부트스트랩 {N_BOOT}회 95% 범위")
    print(f"  ① AUC    {auc_s - auc_b:+.3f}  ({ci_auc[0]:+.3f} ~ {ci_auc[1]:+.3f})  {'✓' if ok_auc else '✗'}")
    print(f"  ② 향상도 {sk_s - sk_b:+.2%}  ({ci_sk[0]:+.2%} ~ {ci_sk[1]:+.2%})  {'✓' if ok_sk else '✗'}")
    print(f"\n판정: {'합격' if passed else '불합격'}")

    print("\n학생이 본 것 (표준화 계수 — 클수록 장대봉 쪽으로 강하게 밂)")
    for k, v in sorted(coef_stu.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:<10}{v:+.3f}")

    print("\n정직함 (말한 확률 → 실제 비율, 낮은 칸부터)")
    honesty(y, p_base, "변동성")
    honesty(y, p_stu, "학생  ")

    out = {
        "train_hours": len(train), "test_hours": len(test), "base_rate": base_rate,
        "base": {"auc": auc_b, "skill": sk_b, "coef": coef_base},
        "student": {"auc": auc_s, "skill": sk_s, "coef": coef_stu},
        "diff_auc_ci": ci_auc.tolist(), "diff_skill_ci": ci_sk.tolist(), "passed": passed,
    }
    (DATA / "c03_mock.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))
