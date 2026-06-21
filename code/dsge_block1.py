# -*- coding: utf-8 -*-
"""Block 1 — BOKDPM 핵심 블록의 Python 포트 (준구조 NK gap 모형).

한국은행 BOKDPM(Kang 2008; Pyoun·Park 2009; Kim·Kim 2010)의 한국 도메스틱 블록을
calibration 파라미터 그대로 이식: 전향적 IS곡선·필립스곡선·테일러준칙·실질금리·
UIP 환율·오쿤 실업식. 전향항(기대)은 Fair-Taylor(Gauss-Seidel 시간반복)로 해.
Dynare 미설치 환경이라 Dynare 대신 동일 구조를 결정론적으로 시뮬레이션.

역할: 시나리오 충격(잠재성장·세계수요·유가·무역비용·인구) → 거시 경로 생성
      → BVARX(Block 2) 외생 입력 5종(oil_g, trd_g, d_rrate, rfx_g, lab_g) 산출.
노동 = 인구추계 노동력증가(KOSIS) + 오쿤 실업갭의 경기 조정(고용=노동력×(1-실업)).

산출: dsge_exog_paths.csv (시나리오×40분기×5 외생), dsge_macro_paths.csv (거시),
      _dsge_viz.json
"""
import json
import numpy as np
import pandas as pd

# --- BOKDPM calibration (.mod에서; 노후 정상상태값은 2025~35용으로 갱신·주석) ---
b1, b2, b3, b4, b6 = 0.50, 0.10, 0.10, 0.30, 0.10      # IS: lag,lead,realrate,world,oil
lam1, lam2, lam3 = 0.55, 0.15, 0.65                     # Phillips
g1, g2, g3 = 0.50, 1.50, 0.50                           # Taylor rule
a1, a2 = 0.80, 0.30                                     # Okun: gap지속, 산출갭→실업
om4, om7, om8 = 0.25, 0.50, 0.20                        # UIP 환율
rrbar = 1.5                                             # 중립실질금리(BOKDPM)
pietar = 2.0                                            # 인플레목표(현행 2%; 원본 3.0)
H, BUF = 40, 24                                         # 2026Q1~2035Q4 + 종단버퍼
N = H + BUF
SCN = ["S1", "S2", "S3", "S4", "S5"]


def labor_force_growth():
    """KOSIS 생산연령인구 → 분기 노동력증가율 (가정별)."""
    pop = pd.read_csv("kosis_생산연령인구.csv", encoding="utf-8-sig").set_index("연도")
    col = {"중위": "중위_천명", "고위": "고위_천명", "저위": "저위_천명"}
    out = {}
    for a, c in col.items():
        ann = {y: np.log(pop.loc[y, c] / pop.loc[y - 1, c]) * 100 for y in range(2026, 2036)}
        q = np.array([ann[2026 + h // 4] / 4 for h in range(H)])
        out[a] = q
    return out


def scenario_inputs(s):
    """시나리오별 외생 충격 경로 (length N).  세계수요갭·유가·무역비용·수요충격."""
    z = np.zeros(N)
    dec = lambda k: np.clip(1 - np.arange(N) / k, 0, 1)
    yus = z.copy(); oil = z.copy(); tcost = z.copy(); dshk = z.copy()
    if s == "S2":   # 외부 충격: 세계수요↓·유가↓(수요)·무역비용↑·국내수요↓
        yus = -1.2 * dec(12); oil = -8.0 * dec(8); tcost = +1.5 * dec(12); dshk = -1.0 * dec(12)
    if s == "S4":   # 기술도약/대외 상방: 세계수요↑·TFP(수요견인)
        yus = +0.8 * dec(20); dshk = +0.6 * np.ones(N) * dec(28)
    if s == "S5":   # 지정학 리스크: 유가 급등(공급충격)·무역비용 급등·세계수요↓·안전자산 선호
        oil = +15.0 * dec(8); tcost = +2.0 * dec(10); yus = -0.6 * dec(10); dshk = -0.8 * dec(10)
    return yus, oil, tcost, dshk


def solve_korea_block(yus, oil, dshk, iters=400):
    """Fair-Taylor: Y(산출갭),PIE(분기인플레,연율),RS(정책금리),S(실질환율갭) 동시 해."""
    Y = np.zeros(N); PIE = np.full(N, pietar); RS = np.full(N, rrbar + pietar); S = np.zeros(N)
    for _ in range(iters):
        Y0, PIE0, RS0, S0 = Y.copy(), PIE.copy(), RS.copy(), S.copy()
        for t in range(N):
            yl = Y[t - 1] if t > 0 else 0.0
            yf = Y[t + 1] if t + 1 < N else 0.0
            rsl = RS[t - 1] if t > 0 else rrbar + pietar
            rgap_l = (RS[t - 1] if t > 0 else rrbar + pietar) - pietar - rrbar   # 실질금리갭
            Y[t] = (b1 * yl + b2 * yf - b3 * rgap_l + b4 * yus[t] - b6 * oil[t] / 10 + dshk[t])
            pif = PIE[t + 1] if t + 1 < N else pietar
            pil = PIE[t - 1] if t > 0 else pietar
            PIE[t] = lam3 * pietar + (1 - lam3) * (lam1 * pif + (1 - lam1) * pil + lam2 * 4 * yl)
            pie_e = PIE[t + 1] if t + 1 < N else pietar
            RS[t] = g1 * rsl + (1 - g1) * (rrbar + pietar + g2 * (pie_e - pietar) + g3 * 4 * Y[t])
            sf = S[t + 1] if t + 1 < N else 0.0
            sl = S[t - 1] if t > 0 else 0.0
            rgap = RS[t] - pietar - rrbar
            S[t] = (1 - 0.1) * (om7 * sf + (1 - om7) * sl - om4 * rgap) + om8 * (-Y[t])
        if max(np.abs(Y - Y0).max(), np.abs(RS - RS0).max(), np.abs(S - S0).max()) < 1e-8:
            break
    RR = RS - PIE                                          # 실질금리(ex-post 근사)
    return Y[:H], PIE[:H], RS[:H], RR[:H], S[:H]


def main():
    lfg = labor_force_growth()
    lab_assump = {"S1": "중위", "S2": "저위", "S3": "고위", "S4": "중위", "S5": "중위"}
    partic = np.linspace(0.05, 0.0, H)                    # 참가율 추세(완만 둔화), 분기%
    rows, macro, viz = [], [], {"q": [f"{2026+h//4}Q{h%4+1}" for h in range(H)], "scn": SCN, "d": {}}

    for s in SCN:
        yus, oil_sh, tcost, dshk = scenario_inputs(s)
        Y, PIE, RS, RR, S = solve_korea_block(yus, oil_sh, dshk)
        UNRgap = np.zeros(H)                               # 오쿤 실업갭
        for t in range(H):
            UNRgap[t] = a1 * (UNRgap[t - 1] if t > 0 else 0) - a2 * Y[t]
        dUNR = np.diff(np.concatenate([[0], UNRgap]))
        # BVARX 외생 5종 매핑
        oil_g = (0.5 + oil_sh[:H] / 8 * 4)                 # 분기 유가증가율(baseline 완만)
        trd_g = (1.0 + 0.8 * yus[:H] - 0.6 * tcost[:H])    # 교역량증가율 ~ 세계수요−무역비용
        d_rrate = np.diff(np.concatenate([[RR[0]], RR]))   # 실질금리 변화
        rfx_g = np.diff(np.concatenate([[0], S]))          # 실질환율 변화(절하+)
        lab_g = lfg[lab_assump[s]] + partic - dUNR         # 고용증가 = 노동력+참가−Δ실업
        for h in range(H):
            rows.append([s, viz["q"][h], round(oil_g[h], 3), round(trd_g[h], 3),
                         round(d_rrate[h], 3), round(rfx_g[h], 3), round(lab_g[h], 3)])
            macro.append([s, viz["q"][h], round(Y[h], 3), round(PIE[h], 3), round(RS[h], 3),
                          round(RR[h], 3), round(S[h], 3), round(UNRgap[h], 3)])
        viz["d"][s] = {"Y": [round(x, 2) for x in Y], "RS": [round(x, 2) for x in RS],
                       "UNRgap": [round(x, 2) for x in UNRgap], "lab_g": [round(x, 3) for x in lab_g]}

    pd.DataFrame(rows, columns=["시나리오", "분기", "oil_g", "trd_g", "d_rrate", "rfx_g", "lab_g"]).to_csv(
        "dsge_exog_paths.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(macro, columns=["시나리오", "분기", "산출갭Y", "인플레PIE", "정책금리RS",
                                 "실질금리RR", "실질환율갭S", "실업갭UNR"]).to_csv(
        "dsge_macro_paths.csv", index=False, encoding="utf-8-sig")
    json.dump(viz, open("_dsge_viz.json", "w", encoding="utf-8"), ensure_ascii=False)

    print("Block 1 (BOKDPM 포트) 시뮬레이션 완료")
    print(f"{'시나리오':<6}{'2030 산출갭':>10}{'2030 실업갭':>10}{'10년 고용Σ':>10}{'10년 실질금리Δ':>12}")
    for s in SCN:
        d = viz["d"][s]; i30 = (2030 - 2026) * 4
        labsum = sum(d["lab_g"]); rrch = pd.read_csv("dsge_exog_paths.csv", encoding="utf-8-sig")
        rr10 = rrch[rrch["시나리오"] == s]["d_rrate"].sum()
        print(f"{s:<6}{d['Y'][i30]:>10.2f}{d['UNRgap'][i30]:>10.2f}{labsum:>10.2f}{rr10:>12.2f}")
    print("저장: dsge_exog_paths.csv, dsge_macro_paths.csv")


if __name__ == "__main__":
    main()
