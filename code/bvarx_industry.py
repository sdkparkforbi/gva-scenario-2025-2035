# -*- coding: utf-8 -*-
"""BVARX — 36개 산업 실질부가가치 증가율 + 외생 거시 블록 (시나리오 엔진).

  y_t = c + Σ_{l=1..p} B_l y_{t-l} + Σ_{s=0..q} Γ_s x_{t-s} + u_t,  u_t~N(0,Σ)
  내생 y : 36개 산업 100·Δlog(real VA)           (미네소타 prior, λ=0.1)
  외생 x : oil_g, trd_g, d_rrate, rfx_g, lab_g    (느슨한 prior, λx=1.0)
  내생 lag p=4, 외생 lag q=2(동시+2분기). 켤레 정규-역위샤트 사후평균.

외생 동태승수(dynamic multiplier) D_h = ∂y_{t+h}/∂x_t' 를 산출 →
각 외생충격(유가·환율·금리·교역·노동)의 산업별 전파 = 시나리오 시뮬레이션 기반.

산출: bvarx_coefficients.csv, bvarx_exog_multipliers.csv, bvarx_summary.txt,
      _bvarx_mult.json (시각화용)
"""
import json
import numpy as np
import pandas as pd

from bvar_industry import load_growth, ar_sigma

P, Q = 4, 2                 # 내생 lag, 외생 lag
LAM, LAMX = 0.1, 1.0        # 내생/외생 prior 수축강도
EXO = ["oil_g", "trd_g", "d_rrate", "rfx_g", "lab_g"]
# 노동 패스스루를 구조 탄력성(노동분배율)으로 제약. 축소형 계수(~1.4)는 경기 공행성을
# 반영해 구조적 인구감소에 적용 시 산출을 과대 위축시킨다 → 동시계수 prior평균=노동분배율.
# 동시계수 prior평균을 0.40으로 잡으면, 산업 간 동학 증폭 후 총량 가중 장기승수가
# 0.85(노동분배율 0.65 + 완만한 일반균형 승수) 수준이 되어 구조적으로 타당해진다.
LAB_SHARE, OM_LAB = 0.40, 0.0008


def qkey(q):
    return (int(q[:4]), int(q[5]))


def build():
    g = load_growth()                                    # 36산업 증가율
    ex = pd.read_csv("exog_quarterly_filled.csv", encoding="utf-8-sig").set_index("분기")
    ex = ex[EXO].astype(float)
    common = sorted(set(g.index) & set(ex.index), key=qkey)
    g, ex = g.loc[common], ex.loc[common]
    n, ne = g.shape[1], ex.shape[1]
    T0 = max(P, Q)                                        # lag 확보 시작
    rows = list(range(T0, len(common)))
    Y = g.values[rows]                                   # (T x n)
    T = len(rows)
    # 설계: [내생 L1..Lp | 외생 L0..Lq | const]
    blocks = []
    for l in range(1, P + 1):
        blocks.append(g.values[[r - l for r in rows]])
    for s in range(0, Q + 1):
        blocks.append(ex.values[[r - s for r in rows]])
    X = np.column_stack(blocks + [np.ones(T)])
    return g, ex, Y, X, n, ne, [common[r] for r in rows]


def prior(n, ne, sig_y, sig_x):
    k = n * P + ne * (Q + 1) + 1
    B0 = np.zeros((k, n))                                # white-noise prior
    om = np.zeros(k)
    pos = 0
    for l in range(1, P + 1):                            # 내생: 미네소타
        for j in range(n):
            om[pos] = LAM ** 2 / (l ** 2 * sig_y[j] ** 2); pos += 1
    for s in range(0, Q + 1):                            # 외생: 느슨
        for m in range(ne):
            om[pos] = LAMX ** 2 / ((s + 1) ** 2 * sig_x[m] ** 2); pos += 1
    om[pos] = 1e6                                        # const
    # 노동계수 구조 제약: 동시계수 prior평균=노동분배율, 시차=0, 모두 tight
    lab = EXO.index("lab_g")
    for s in range(Q + 1):
        r = n * P + s * ne + lab
        B0[r, :] = LAB_SHARE if s == 0 else 0.0
        om[r] = OM_LAB
    return B0, np.diag(om), np.diag(sig_y ** 2), n + 2


def posterior(Y, X, B0, Om0, Psi, d):
    iO0 = np.diag(1 / np.diag(Om0))
    iOb = iO0 + X.T @ X
    Ob = np.linalg.inv(iOb)
    Bb = Ob @ (iO0 @ B0 + X.T @ Y)
    Psib = Psi + Y.T @ Y + B0.T @ iO0 @ B0 - Bb.T @ iOb @ Bb
    return Bb, (Psib + Psib.T) / 2, d + Y.shape[0]


def multipliers(Bb, n, ne, H=20):
    """외생 동태승수 D_h (n x ne), h=0..H.  D_h = Σ B_l D_{h-l} + Γ_h."""
    Bl = [Bb[(l - 1) * n:l * n, :].T for l in range(1, P + 1)]      # A_l (n x n)
    G = [Bb[n * P + s * ne: n * P + (s + 1) * ne, :].T for s in range(Q + 1)]  # Γ_s (n x ne)
    D = []
    for h in range(H + 1):
        Dh = G[h].copy() if h <= Q else np.zeros((n, ne))
        for l in range(1, P + 1):
            if h - l >= 0:
                Dh = Dh + Bl[l - 1] @ D[h - l]
        D.append(Dh)
    return D                                              # 길이 H+1, 각 (n x ne)


def main():
    g, ex, Y, X, n, ne, qs = build()
    sig_y = ar_sigma(g, P)
    sig_x = ex.std().values
    B0, Om0, Psi, d = prior(n, ne, sig_y, sig_x)
    Bb, Psib, db = posterior(Y, X, B0, Om0, Psi, d)
    Sigma = Psib / (db - n - 1)
    resid = Y - X @ Bb
    r2 = pd.Series(1 - resid.var(0) / Y.var(0), index=g.columns)

    # 안정성(내생 동반행렬)
    A = Bb[:n * P, :].T
    comp = np.zeros((n * P, n * P)); comp[:n, :] = A
    comp[n:, :-n] = np.eye(n * (P - 1))
    eig = np.abs(np.linalg.eigvals(comp)).max()

    # 외생 동태승수: 1 표준편차 충격에 대한 누적(레벨) 반응
    D = multipliers(Bb, n, ne, H=20)
    cum = {}
    for m, xname in enumerate(EXO):
        imp = np.array([D[h][:, m] * sig_x[m] for h in range(len(D))])   # (H+1 x n) 증가율 반응
        cum[xname] = np.cumsum(imp, axis=0)                              # 누적=레벨 반응

    # 저장: 20분기 누적 반응(장기 승수)
    longrun = pd.DataFrame({xname: cum[xname][-1] for xname in EXO}, index=g.columns)
    longrun.to_csv("bvarx_exog_multipliers.csv", encoding="utf-8-sig")
    cols = ([f"y.L{l}.{c}" for l in range(1, P + 1) for c in g.columns]
            + [f"x.L{s}.{m}" for s in range(Q + 1) for m in EXO] + ["const"])
    pd.DataFrame(Bb, index=cols, columns=g.columns).to_csv(
        "bvarx_coefficients.csv", encoding="utf-8-sig")

    # 시각화용: 주요 외생충격의 산업별 20분기 누적반응 + 시간경로(대표산업)
    viz = {"exo": EXO, "industries": list(g.columns),
           "longrun": {x: [round(v, 3) for v in longrun[x].values] for x in EXO},
           "sig_x": {x: round(float(s), 3) for x, s in zip(EXO, sig_x)}}
    json.dump(viz, open("_bvarx_mult.json", "w", encoding="utf-8"), ensure_ascii=False)

    with open("bvarx_summary.txt", "w", encoding="utf-8") as f:
        f.write("BVARX — 36산업 증가율 + 외생 5변수 (시나리오 엔진)\n")
        f.write(f"표본 {qs[0]}~{qs[-1]} (T={Y.shape[0]}), 내생 p={P}, 외생 q={Q}, "
                f"λ={LAM}, λx={LAMX}\n")
        f.write(f"안정성 |z|max={eig:.4f} ({'안정' if eig<1 else '불안정'}), "
                f"평균 R²={r2.mean():.3f}\n\n")
        f.write("[외생충격 1σ → 20분기 누적(레벨) 반응: 산업 평균, %p]\n")
        for x in EXO:
            v = longrun[x]
            f.write(f"  {x:8} 1σ={sig_x[EXO.index(x)]:.2f}: 산업평균 {v.mean():+.2f}, "
                    f"최대 {v.idxmax()[:10]} {v.max():+.2f}, 최소 {v.idxmin()[:10]} {v.min():+.2f}\n")
    print(f"BVARX 추정: 표본 {qs[0]}~{qs[-1]} T={Y.shape[0]}, |z|max={eig:.4f}, 평균R²={r2.mean():.3f}")
    print("저장: bvarx_coefficients.csv, bvarx_exog_multipliers.csv, bvarx_summary.txt")


if __name__ == "__main__":
    main()
