# -*- coding: utf-8 -*-
"""36개 산업 실질부가가치 증가율에 대한 Large Bayesian VAR (켤레 미네소타 prior).

- 데이터 : gdp_industry_long.csv 의 실질부가가치(연쇄, 십억원), 1980Q1~2025Q4
- 변수   : g_t = 100 * Δlog(real VA)  (36개 산업, 정상시계열)
- 모형   : VAR(p=4),  y_t = c + B1 y_{t-1} + ... + B4 y_{t-4} + u_t,  u_t~N(0,Σ)
- prior  : 켤레 정규-역위샤트(Minnesota).  자기 1차항 prior평균 δ=0 (white-noise),
           lag l·변수 j 계수 분산 ∝ λ²/(l² σ_j²),  Σ~IW(diag(σ_i²), n+2)
- λ 선택 : 주변우도(marginal likelihood) 격자 최대화 (Giannone-Lenza-Primiceri 2015)

산출:
  bvar_coefficients.csv  : 사후평균 계수 B (k x n)
  bvar_sigma.csv         : 잔차 공분산 Σ
  bvar_summary.txt       : 선택 λ, 안정성, 적합도 등 진단
"""
import numpy as np
import pandas as pd
from scipy.special import multigammaln

P = 4                       # 시차 (요청: max 4)
DELTA = 0.0                 # 자기 1차항 prior 평균 (증가율→0, white-noise)
LAMBDAS = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]


def load_growth():
    df = pd.read_csv("gdp_industry_long.csv", encoding="utf-8-sig")
    w = df.pivot(index="분기", columns="산업명", values="실질_십억원")
    order = sorted(w.index, key=lambda q: (int(q[:4]), int(q[5])))
    w = w.loc[order]
    g = 100.0 * np.log(w).diff().dropna()        # 100*Δlog, 183 x 36
    # 정유 등 극단 분기변동(예: 2020Q1 코크스·석유정제 -128%) 윈저화 → Σ·상수 왜곡 방지.
    # 실질부가가치=거대 산출−거대 투입이라 정유의 차액(부가가치)이 작고 변동이 극심하다.
    g = g.clip(-30, 30)
    return g


def build_xy(g, p):
    """y_t 와 x_t=(y_{t-1}..y_{t-p},1) 구성."""
    Y = g.values[p:]                              # (T x n)
    T, n = Y.shape
    X = np.ones((T, n * p + 1))
    for l in range(1, p + 1):
        X[:, (l - 1) * n:(l) * n] = g.values[p - l:-l]
    X[:, -1] = 1.0                                # 절편 (마지막 열)
    return Y, X


def ar_sigma(g, p):
    """각 변수 단변량 AR(p) 잔차표준편차 σ_i."""
    sig = np.zeros(g.shape[1])
    for i, col in enumerate(g.columns):
        y = g[col].values
        Y = y[p:]
        Z = np.column_stack([y[p - l:-l] for l in range(1, p + 1)] + [np.ones(len(Y))])
        beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
        sig[i] = (Y - Z @ beta).std(ddof=Z.shape[1])
    return sig


def minnesota_prior(n, p, sigma, lam):
    """켤레 미네소타 prior: B0(k x n), Omega0(k x k diag), Psi(n x n), d."""
    k = n * p + 1
    B0 = np.zeros((k, n))
    for i in range(n):
        B0[i, i] = DELTA                          # 자기 1차항
    om = np.zeros(k)
    for l in range(1, p + 1):
        for j in range(n):
            om[(l - 1) * n + j] = (lam ** 2) / (l ** 2 * sigma[j] ** 2)
    om[-1] = 1e6                                   # 절편: 매우 느슨
    Omega0 = np.diag(om)
    Psi = np.diag(sigma ** 2)
    d = n + 2                                      # E[Σ]=Psi/(d-n-1)=diag(σ²)
    return B0, Omega0, Psi, d


def posterior(Y, X, B0, Omega0, Psi, d):
    n = Y.shape[1]
    iO0 = np.diag(1.0 / np.diag(Omega0))
    iOb = iO0 + X.T @ X
    Ob = np.linalg.inv(iOb)
    Bb = Ob @ (iO0 @ B0 + X.T @ Y)
    Psib = Psi + Y.T @ Y + B0.T @ iO0 @ B0 - Bb.T @ iOb @ Bb
    Psib = (Psib + Psib.T) / 2
    db = d + Y.shape[0]
    return Bb, Ob, Psib, db, Omega0


def log_ml(Y, X, B0, Omega0, Psi, d):
    """주변우도 log p(Y) — 켤레 N-IW 정확식."""
    n, T = Y.shape[1], Y.shape[0]
    Bb, Ob, Psib, db, _ = posterior(Y, X, B0, Omega0, Psi, d)
    _, ld_O0 = np.linalg.slogdet(Omega0)
    _, ld_Ob = np.linalg.slogdet(Ob)
    _, ld_Psi = np.linalg.slogdet(Psi)
    _, ld_Psib = np.linalg.slogdet(Psib)
    return (-(n * T / 2) * np.log(np.pi)
            + (n / 2) * (ld_Ob - ld_O0)
            + (d / 2) * ld_Psi - (db / 2) * ld_Psib
            + multigammaln(db / 2, n) - multigammaln(d / 2, n))


def companion_max_eig(Bb, n, p):
    """동반행렬 최대 고유값 절댓값 (안정성: <1 이면 정상)."""
    A = Bb[:n * p, :].T                            # (n x n*p), 절편 제외
    comp = np.zeros((n * p, n * p))
    comp[:n, :] = A
    comp[n:, :-n] = np.eye(n * (p - 1))
    return np.max(np.abs(np.linalg.eigvals(comp)))


def main():
    g = load_growth()
    n = g.shape[1]
    Y, X = build_xy(g, P)
    sigma = ar_sigma(g, P)
    T = Y.shape[0]

    # λ 선택 (주변우도 최대)
    grid = []
    for lam in LAMBDAS:
        B0, Om0, Psi, d = minnesota_prior(n, P, sigma, lam)
        grid.append((lam, log_ml(Y, X, B0, Om0, Psi, d)))
    lam_opt = max(grid, key=lambda t: t[1])[0]

    # 최적 λ 추정
    B0, Om0, Psi, d = minnesota_prior(n, P, sigma, lam_opt)
    Bb, Ob, Psib, db, _ = posterior(Y, X, B0, Om0, Psi, d)
    Sigma = Psib / (db - n - 1)                    # 사후평균 Σ
    resid = Y - X @ Bb
    eig = companion_max_eig(Bb, n, P)

    # 적합도 (변수별 in-sample R²)
    r2 = pd.Series(1 - resid.var(0) / Y.var(0), index=g.columns)

    # 저장
    cols = ([f"L{l}.{c}" for l in range(1, P + 1) for c in g.columns] + ["const"])
    pd.DataFrame(Bb, index=cols, columns=g.columns).to_csv(
        "bvar_coefficients.csv", encoding="utf-8-sig")
    pd.DataFrame(Sigma, index=g.columns, columns=g.columns).to_csv(
        "bvar_sigma.csv", encoding="utf-8-sig")

    with open("bvar_summary.txt", "w", encoding="utf-8") as f:
        f.write("Large Bayesian VAR — 36개 산업 실질부가가치 증가율 (100*Δlog)\n")
        f.write(f"표본: {g.index[P]} ~ {g.index[-1]}  (유효관측 T={T}, 변수 n={n}, 시차 p={P})\n")
        f.write(f"prior: 켤레 미네소타, δ={DELTA}(white-noise), Σ~IW(diag(σ²), n+2)\n\n")
        f.write("[λ 선택: 주변우도]\n")
        for lam, ml in grid:
            mark = "  <== 선택" if lam == lam_opt else ""
            f.write(f"  λ={lam:>5}:  logML={ml:14.2f}{mark}\n")
        f.write(f"\n선택 λ* = {lam_opt}\n")
        f.write(f"안정성: 동반행렬 최대고유값 |z|max = {eig:.4f}  "
                f"({'정상(안정)' if eig < 1 else '비정상'})\n")
        f.write(f"평균 in-sample R² = {r2.mean():.3f}  "
                f"(min {r2.min():.3f}, max {r2.max():.3f})\n\n")
        f.write("[변수별 in-sample R²]\n")
        for c, v in r2.sort_values(ascending=False).items():
            f.write(f"  {v:5.3f}  {c}\n")

    print(f"BVAR 추정 완료: λ*={lam_opt}, |z|max={eig:.4f}, 평균R²={r2.mean():.3f}, T={T}, n={n}")
    print("저장: bvar_coefficients.csv, bvar_sigma.csv, bvar_summary.txt")


if __name__ == "__main__":
    main()
