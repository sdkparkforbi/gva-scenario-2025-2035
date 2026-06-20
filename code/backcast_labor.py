# -*- coding: utf-8 -*-
"""취업자 증가율(lab_g) 백캐스팅 — 동적요인 칼만평활 + 홀드아웃 검증.

lab_g(1999Q3~)를 1990Q3까지 복원. 두 예측자 집합을 비교:
  (A) 외생 4종만        : oil_g, trd_g, d_rrate, rfx_g
  (B) + 실질GDP 증가율  : gdp_g 추가 (Okun의 법칙 — 고용의 본질적 동인)
홀드아웃(최근 관측구간을 일부러 결측처리→복원→실측 비교)으로 예측력을 정직하게 평가.
최종 백캐스트는 검증 우수안으로 1990Q3~1999Q2 결측을 복원.

산출: exog_quarterly_filled.csv, _labor_backcast.json
"""
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

SPAN0, SPAN1 = "1990Q3", "2025Q4"
BASE = ["oil_g", "trd_g", "d_rrate", "rfx_g"]


def qkey(q):
    return (int(q[:4]), int(q[5]))


def load_panel():
    ex = pd.read_csv("exog_quarterly.csv", index_col=0, encoding="utf-8-sig")
    g = pd.read_csv("gdp_quarterly_sa.csv", encoding="utf-8-sig")          # 실질 GDP
    g = g.set_index("분기")["실질GDP_십억원"].astype(float)
    gdp_g = 100 * np.log(g).diff()
    idx = sorted([q for q in ex.index if SPAN0 <= q <= SPAN1], key=qkey)
    P = ex.loc[idx, BASE + ["lab_g"]].astype(float)
    P["gdp_g"] = gdp_g.reindex(idx)
    return P, idx


def smooth_labor(panel, cols):
    """동적요인모형 적합 후 lab_g 평활 재구성 (표준화 역변환), (lab_hat, se)."""
    Z0 = panel[cols + ["lab_g"]]
    mu, sd = Z0.mean(), Z0.std()
    Z = (Z0 - mu) / sd
    mod = DynamicFactor(Z, k_factors=2, factor_order=2, error_order=1,
                        enforce_stationarity=True)
    res = mod.fit(disp=False, maxiter=400, method="lbfgs")
    d = res.filter_results.design
    d = d[:, :, 0] if d.ndim == 3 else d
    li = (cols + ["lab_g"]).index("lab_g")
    a, Pc = res.smoothed_state, res.smoothed_state_cov
    zr = d[li]
    yhat = zr @ a
    var = np.clip([zr @ Pc[:, :, t] @ zr for t in range(Pc.shape[2])], 0, None)
    return yhat * sd["lab_g"] + mu["lab_g"], np.sqrt(var) * sd["lab_g"]


def holdout(panel, cols, test=("2019Q1", "2021Q4")):
    """관측구간 일부를 결측처리→복원→실측 비교 (블록 백캐스트 모사)."""
    idx = list(panel.index)
    mask = [(test[0] <= q <= test[1]) for q in idx]
    actual = panel["lab_g"].values.copy()
    pp = panel.copy()
    pp.loc[[q for q, m in zip(idx, mask) if m], "lab_g"] = np.nan
    hat, _ = smooth_labor(pp, cols)
    m = np.array(mask) & ~np.isnan(actual)
    rmse = np.sqrt(np.nanmean((actual[m] - hat[m]) ** 2))
    corr = np.corrcoef(actual[m], hat[m])[0, 1]
    return rmse, corr


def main():
    panel, idx = load_panel()
    obs = panel["lab_g"].values
    missing = np.isnan(obs)

    print("[홀드아웃 검증] 2019Q1~2021Q4(COVID 변동기) 결측처리 후 복원 vs 실측")
    rmse_a, corr_a = holdout(panel, BASE)
    rmse_b, corr_b = holdout(panel, BASE + ["gdp_g"])
    print(f"  (A) 외생4종     : RMSE={rmse_a:.3f}  corr={corr_a:+.3f}")
    print(f"  (B) +실질GDP    : RMSE={rmse_b:.3f}  corr={corr_b:+.3f}")
    better = BASE + ["gdp_g"] if rmse_b < rmse_a else BASE
    print(f"  → 채택: {'(B) +실질GDP' if better == BASE + ['gdp_g'] else '(A) 외생4종'}")

    lab_hat, lab_se = smooth_labor(panel, better)
    lab_filled = np.where(missing, lab_hat, obs)

    out = panel[BASE].copy()
    out["lab_g"] = lab_filled
    out.index.name = "분기"
    out.to_csv("exog_quarterly_filled.csv", encoding="utf-8-sig")

    viz = {"q": idx,
           "obs": [None if m else round(float(v), 3) for m, v in zip(missing, obs)],
           "hat": [round(float(v), 3) for v in lab_hat],
           "lo": [round(float(h - 1.64 * s), 3) for h, s in zip(lab_hat, lab_se)],
           "hi": [round(float(h + 1.64 * s), 3) for h, s in zip(lab_hat, lab_se)],
           "adopt": "B(+GDP)" if better == BASE + ["gdp_g"] else "A(4exog)"}
    json.dump(viz, open("_labor_backcast.json", "w", encoding="utf-8"), ensure_ascii=False)

    print(f"\n최종 백캐스트: 결측 {int(missing.sum())}분기 복원 ({idx[0]}~1999Q2)")
    for q in ["1997Q4", "1998Q1", "1998Q2", "1998Q4", "1999Q1"]:
        i = idx.index(q)
        print(f"  lab_g {q}: {lab_filled[i]:+.2f}%  [90%: {viz['lo'][i]:+.2f}~{viz['hi'][i]:+.2f}] (백캐스트)")
    print("저장: exog_quarterly_filled.csv")


if __name__ == "__main__":
    main()
