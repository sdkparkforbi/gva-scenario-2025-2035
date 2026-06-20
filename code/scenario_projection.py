# -*- coding: utf-8 -*-
"""제안서 S1~S4 시나리오 전망 (2026Q1~2035Q4) — BVARX 조건부 예측 + 팬차트.

제안서 page 13/19 시나리오 설계를 그대로 구현 (Block 2 = 본 BVARX):
  노동공급(생산연령인구 통계청 추계) × 외부환경 2축의 4개 시나리오
    S1 기준    : 노동 중위 / 외부 안정
    S2 제약+충격: 노동 저위 / 외부 충격(교역↓·원화절하·금리↑·유가↓)   [RFP 필수]
    S3 제약완화 : 노동 고위(국제순이동 高=외국인력 보강) / 외부 안정     [RFP 필수]
    S4 기술도약 : 노동 중위 / 외부 상방(교역↑)  (TFP는 DSGE 채널—VARX 외 주석)
핵심 비교쌍 S2 vs S3 → 노동공급 효과 → 산업별 추가 필요 인력의 직접 입력.

노동경로 = 통계청 장래인구추계 생산연령인구(15-64) 연간 log증가율/4 (분기 lab_g).
불확실성 = 켤레 NIW 사후 + 충격 추출 → 70%/90% 팬차트.

산출: scenario_paths.csv(외생 경로 매핑·공개), scenario_gdp_fan.csv(총량 GDP 팬),
      scenario_industry_2035.csv(산업별 2035 수준·S2-S3 차이), _scenario_viz.json
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import invwishart

from bvar_industry import load_growth, ar_sigma
from bvarx_industry import build, prior, EXO, P, Q

H = 40                     # 2026Q1 ~ 2035Q4
NDRAW = 400
SEED = 20260621
SCN = ["S1", "S2", "S3", "S4"]
LAB_ASSUMP = {"S1": "중위", "S2": "저위", "S3": "고위", "S4": "중위"}


def labor_paths():
    """통계청 생산연령인구 → 분기 lab_g (2026~2035, 40분기) 가정별."""
    pop = pd.read_csv("kosis_생산연령인구.csv", encoding="utf-8-sig").set_index("연도")
    col = {"중위": "중위_천명", "고위": "고위_천명", "저위": "저위_천명"}
    out = {}
    for a, c in col.items():
        ann = {y: np.log(pop.loc[y, c] / pop.loc[y - 1, c]) * 100 for y in range(2026, 2036)}
        out[a] = np.array([ann[2026 + h // 4] / 4 for h in range(H)])   # 연→분기 분할
    return out


def scenario_exog(means, sig, labp):
    """4개 시나리오 외생경로 (H x 5).  baseline + 충격 overlay."""
    base = np.tile([means["oil_g"], means["trd_g"], 0.0, 0.0, 0.0], (H, 1))  # rate·fx 변화는 0
    paths = {}
    for s in SCN:
        X = base.copy()
        X[:, 4] = labp[LAB_ASSUMP[s]]                                   # 노동
        if s == "S2":   # 외부 충격: 교역↓·원화절하↑·금리↑·유가↓ (3년 감쇠)
            dec12 = np.clip(1 - np.arange(H) / 12, 0, 1)
            dec8 = np.clip(1 - np.arange(H) / 8, 0, 1)
            X[:, 1] += -1.5 * sig["trd_g"] * dec12
            X[:, 3] += +1.0 * sig["rfx_g"] * dec8
            X[:, 2] += +0.5 * sig["d_rrate"] * dec8
            X[:, 0] += -0.5 * sig["oil_g"] * dec8
        if s == "S4":   # 외부 상방: 교역↑ 지속 (TFP는 VARX 외)
            X[:, 1] += +0.7 * sig["trd_g"]
        paths[s] = X
    return paths


def main():
    rng = np.random.default_rng(SEED)
    g, ex, Y, X, n, ne, qs = build()
    names = list(g.columns)
    sig_y = ar_sigma(g, P)
    sig_x = ex.std().values
    B0, Om0, Psi, d = prior(n, ne, sig_y, sig_x)
    iO0 = np.diag(1 / np.diag(Om0))
    iOb = iO0 + X.T @ X
    Ob = np.linalg.inv(iOb)
    Bb = Ob @ (iO0 @ B0 + X.T @ Y)
    Psib = Psi + Y.T @ Y + B0.T @ iO0 @ B0 - Bb.T @ iOb @ Bb
    Psib = (Psib + Psib.T) / 2
    db = d + Y.shape[0]
    Lu = np.linalg.cholesky(Ob)

    # Block 1 (BOKDPM/DSGE 포트, dsge_block1.py) 생성 외생경로 로드 → BVARX 입력
    dp = pd.read_csv("dsge_exog_paths.csv", encoding="utf-8-sig")
    xpaths = {}
    for s in SCN:
        sub = dp[dp["시나리오"] == s].sort_values("분기")
        assert len(sub) == H, f"{s} 경로 길이 불일치"
        xpaths[s] = sub[EXO].values

    # 시드: 최근 내생 4분기, 외생 2분기 (표본 끝 = 2025Q4)
    y_seed = g.values[-P:][::-1]          # [y_{T},y_{T-1},...] 최신순 길이 P
    x_seed = ex.values[-Q:][::-1]         # [x_T, x_{T-1}]
    # 총량 가중치(2024 명목 VA 비중)
    il = pd.read_csv("gdp_industry_long.csv", encoding="utf-8-sig")
    w24 = il[il["분기"].str[:4] == "2024"].groupby("산업명")["명목_십억원"].sum()
    w = np.array([w24.get(nm, 0.0) for nm in names]); w = w / w.sum()

    def coefs(Bmat):
        A = [Bmat[(l) * n:(l + 1) * n, :].T for l in range(P)]            # A1..A4
        G = [Bmat[n * P + s * ne:n * P + (s + 1) * ne, :].T for s in range(Q + 1)]
        c = Bmat[-1, :]
        return A, G, c

    def simulate(Bmat, Sig, xpath, shock):
        A, G, c = coefs(Bmat)
        yb = [y_seed[i].copy() for i in range(P)]      # yb[0]=가장 최근
        xb = [x_seed[0].copy(), x_seed[1].copy()]      # xb[0]=x_{t-1}, xb[1]=x_{t-2}
        gpath = np.zeros((H, n))
        L = np.linalg.cholesky(Sig) if shock else None
        for h in range(H):
            xt = xpath[h]
            yt = c.copy()
            for l in range(P):
                yt = yt + A[l] @ yb[l]
            yt = yt + G[0] @ xt + G[1] @ xb[0] + G[2] @ xb[1]
            if shock:
                yt = yt + L @ rng.standard_normal(n)
            gpath[h] = yt
            yb = [yt] + yb[:-1]
            xb = [xt, xb[0]]
        return gpath

    # 점추정 (shock 없음, 사후평균)
    point = {s: simulate(Bb, Psib / (db - n - 1), xpaths[s], False) for s in SCN}
    # 팬 (사후+충격)
    agg_draws = {s: np.zeros((NDRAW, H)) for s in SCN}
    ind2035 = {s: np.zeros((NDRAW, n)) for s in SCN}
    for k in range(NDRAW):
        Sig = invwishart.rvs(df=db, scale=Psib, random_state=rng)
        Bs = Bb + Lu @ rng.standard_normal((Bb.shape[0], n)) @ np.linalg.cholesky(Sig).T
        for s in SCN:
            gp = simulate(Bs, Sig, xpaths[s], True)
            agg_g = gp @ w                                    # 총량 증가율(가중)
            agg_draws[s][k] = np.cumsum(agg_g)                # 누적(레벨, 2025Q4=0)
            ind2035[s][k] = np.cumsum(gp, 0)[-1]              # 산업별 2035 누적

    # 총량 GDP 팬 (2025Q4=100 지수)
    fan = {}
    for s in SCN:
        q = np.percentile(agg_draws[s], [5, 15, 50, 85, 95], axis=0)
        fan[s] = 100 * np.exp(q / 100)                        # 지수
    pd.DataFrame({f"{s}_{p}": fan[s][i] for s in SCN
                  for i, p in enumerate(["p05", "p15", "p50", "p85", "p95"])},
                 index=[f"{2026+h//4}Q{h%4+1}" for h in range(H)]).to_csv(
        "scenario_gdp_fan.csv", encoding="utf-8-sig")

    # 산업별 2035 누적반응(중앙값) + S2-S3 차이
    med = {s: np.median(ind2035[s], 0) for s in SCN}
    df_ind = pd.DataFrame({s: med[s] for s in SCN}, index=names)
    df_ind["S2-S3(노동효과)"] = df_ind["S2"] - df_ind["S3"]
    df_ind.sort_values("S2-S3(노동효과)").to_csv("scenario_industry_2035.csv", encoding="utf-8-sig")

    # 외생경로 공개 매핑 (연 평균)
    rows = []
    for s in SCN:
        Xp = xpaths[s]
        for yi, yr in enumerate(range(2026, 2036)):
            seg = Xp[yi * 4:(yi + 1) * 4].mean(0)
            rows.append([s, yr] + [round(v, 3) for v in seg])
    pd.DataFrame(rows, columns=["시나리오", "연도"] + EXO).to_csv(
        "scenario_paths.csv", index=False, encoding="utf-8-sig")

    # 시각화
    qlabels = [f"{2026+h//4}Q{h%4+1}" for h in range(H)]
    viz = {"q": qlabels, "scn": SCN,
           "fan": {s: {p: [round(x, 2) for x in fan[s][i]]
                       for i, p in enumerate(["p05", "p15", "p50", "p85", "p95"])} for s in SCN},
           "gdp2035": {s: round(float(fan[s][2][-1]), 1) for s in SCN}}
    json.dump(viz, open("_scenario_viz.json", "w", encoding="utf-8"), ensure_ascii=False)

    print("시나리오 전망 완료 (2026Q1~2035Q4)")
    print(f"  2035Q4 총량 실질GDP 지수 (2025Q4=100):")
    for s in SCN:
        print(f"    {s}: {fan[s][2][-1]:.1f}  [90% {fan[s][0][-1]:.1f}~{fan[s][4][-1]:.1f}]  (노동:{LAB_ASSUMP[s]})")
    print(f"  S2-S3 산업평균 노동효과(2035 누적): {(df_ind['S2-S3(노동효과)'].mean()):+.2f}%")
    print("저장: scenario_gdp_fan.csv, scenario_industry_2035.csv, scenario_paths.csv")


if __name__ == "__main__":
    main()
