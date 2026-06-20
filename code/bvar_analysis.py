# -*- coding: utf-8 -*-
"""BVAR(4) 종합 분석 — 계수·충격반응(GIRF)·일반화FEVD·Diebold-Yilmaz 연결성.

bvar_industry.py 의 켤레 미네소타 BVAR(λ*=0.1) 사후분포에서 직접 추출하여
모든 통계량에 신용구간(credible interval)을 부여한다. 식별은 순서불변(GIRF/GFEVD).

산출:
  bvar_coef_persistence.csv  : 산업별 자기지속성(자기 lag합) 사후평균·신용구간
  bvar_top_spillovers.csv    : 강한 교차산업 스필오버(계수) 상위
  bvar_connectedness.csv     : 산업별 FROM/TO/NET 연결성(GFEVD 기반) + 총연결성
  bvar_girf_<shock>.csv       : 선택 산업 충격에 대한 GIRF(전 산업 반응, 신용구간)
  bvar_analysis_summary.txt
"""
import numpy as np
import pandas as pd
from scipy.stats import invwishart

from bvar_industry import load_growth, build_xy, ar_sigma, minnesota_prior, posterior, P

LAM = 0.1            # bvar_industry.py 에서 주변우도로 선택된 값
NDRAW = 600          # 사후 추출 수
H_IRF = 20           # 충격반응 시계(분기)
H_FEVD = 12          # FEVD/연결성 시계(3년)
SEED = 20260621
SHOCKS = ["컴퓨터, 전자 및 광학기기 제조업", "금융 및 보험업"]


def ma_theta(Bb, n, p, H):
    """MA(∞) 계수 Θ_0..Θ_H (각 n x n). 동반행렬 거듭제곱."""
    A = Bb[:n * p, :].T                       # [A1|A2|A3|A4]  (n x np)
    comp = np.zeros((n * p, n * p))
    comp[:n, :] = A
    if p > 1:
        comp[n:, :-n] = np.eye(n * (p - 1))
    J = np.zeros((n, n * p)); J[:n, :n] = np.eye(n)
    Th, Apow = [np.eye(n)], np.eye(n * p)
    for _ in range(H):
        Apow = comp @ Apow
        Th.append(J @ Apow @ J.T)
    return Th                                  # 길이 H+1


def gfevd(Th, Sigma, H):
    """일반화 FEVD (Pesaran-Shin), 행 정규화. θ̃[i,j] = i의 예측오차분산 중 j 기여분."""
    n = Sigma.shape[0]
    M = [Th[h] @ Sigma for h in range(H + 1)]   # Θ_h Σ
    sig = np.diag(Sigma)
    num = np.zeros((n, n)); den = np.zeros(n)
    for h in range(H + 1):
        num += M[h] ** 2
        den += np.einsum("ik,ik->i", M[h], Th[h])   # diag(Θ_h Σ Θ_h')
    theta = (num / sig[None, :]) / den[:, None]
    return theta / theta.sum(1, keepdims=True)       # 행합=1로 정규화


def connectedness(theta):
    """Diebold-Yilmaz: FROM(받는), TO(주는), NET, 총연결성 TCI(%)."""
    n = theta.shape[0]
    frm = 100 * (1 - np.diag(theta))                 # 행 비대각 합
    to = 100 * (theta.sum(0) - np.diag(theta))        # 열 비대각 합
    net = to - frm
    tci = frm.mean()
    return frm, to, net, tci


def girf(Th, Sigma, j, H):
    """변수 j 에 1 표준편차 일반화충격 → 전 변수 반응 (H+1 x n)."""
    sj = np.sqrt(Sigma[j, j])
    return np.array([(Th[h] @ Sigma[:, j]) / sj for h in range(H + 1)])


def main():
    rng = np.random.default_rng(SEED)
    g = load_growth(); names = list(g.columns); n = len(names)
    Y, X = build_xy(g, P); sigma = ar_sigma(g, P)
    B0, Om0, Psi, d = minnesota_prior(n, P, sigma, LAM)
    Bb, Ob, Psib, db, _ = posterior(Y, X, B0, Om0, Psi, d)

    Lu = np.linalg.cholesky(Ob)                       # row cov
    idx = {nm: i for i, nm in enumerate(names)}

    # 누적 저장소
    persist = np.zeros((NDRAW, n))                     # 자기 lag합
    A1diag = np.zeros((NDRAW, n))                      # 자기 1차항
    FROM = np.zeros((NDRAW, n)); TO = np.zeros((NDRAW, n))
    NET = np.zeros((NDRAW, n)); TCI = np.zeros(NDRAW)
    A1_sum = np.zeros((n, n))                          # 1차 계수 사후평균(스필오버용)
    girf_store = {s: np.zeros((NDRAW, H_IRF + 1, n)) for s in SHOCKS}
    n_stable = 0

    for s in range(NDRAW):
        Sig = invwishart.rvs(df=db, scale=Psib, random_state=rng)
        Lv = np.linalg.cholesky(Sig)
        Bs = Bb + Lu @ rng.standard_normal((Bb.shape[0], n)) @ Lv.T

        A_l = [Bs[(l) * n:(l + 1) * n, :].T for l in range(P)]   # A_{l+1}[c,j]
        persist[s] = sum(np.diag(Al) for Al in A_l)
        A1diag[s] = np.diag(A_l[0])
        A1_sum += A_l[0]

        Th = ma_theta(Bs, n, P, max(H_IRF, H_FEVD))
        # 안정성
        comp_eig = np.abs(np.linalg.eigvals(
            np.block([[Bs[:n * P, :].T],
                      [np.eye(n * (P - 1)), np.zeros((n * (P - 1), n))]])
            if P > 1 else Bs[:n, :].T)).max()
        n_stable += comp_eig < 1

        theta = gfevd(Th, Sig, H_FEVD)
        FROM[s], TO[s], NET[s], TCI[s] = connectedness(theta)
        for sh in SHOCKS:
            girf_store[sh][s] = girf(Th, Sig, idx[sh], H_IRF)

    A1_mean = A1_sum / NDRAW
    qs = lambda a, ax=0: np.percentile(a, [5, 16, 50, 84, 95], axis=ax)

    # 1) 자기지속성
    pq = qs(persist)
    pd.DataFrame({"산업": names, "지속성_p50": pq[2], "p05": pq[0], "p16": pq[1],
                  "p84": pq[3], "p95": pq[4]}).sort_values(
        "지속성_p50", ascending=False).to_csv(
        "bvar_coef_persistence.csv", index=False, encoding="utf-8-sig")

    # 2) 강한 스필오버 (1차 계수, 비대각, |평균|/사후표준편차 기준)
    A1_draws = None  # 메모리 절약: 표준편차는 평균제곱 누적 대신 근사 생략, 평균크기로 선정
    sp = []
    for c in range(n):
        for j in range(n):
            if c != j:
                sp.append((names[j], names[c], A1_mean[c, j]))
    sp = pd.DataFrame(sp, columns=["from_충격산업", "to_반응산업", "A1계수"])
    sp["abs"] = sp["A1계수"].abs()
    sp.sort_values("abs", ascending=False).head(25).drop(columns="abs").to_csv(
        "bvar_top_spillovers.csv", index=False, encoding="utf-8-sig")

    # 3) 연결성
    fq, tq, nq = qs(FROM), qs(TO), qs(NET)
    conn = pd.DataFrame({"산업": names,
                         "FROM_받는": fq[2], "TO_주는": tq[2],
                         "NET": nq[2], "NET_p05": nq[0], "NET_p95": nq[4]})
    conn = conn.sort_values("NET", ascending=False)
    conn.to_csv("bvar_connectedness.csv", index=False, encoding="utf-8-sig")

    # 4) GIRF 저장
    for sh in SHOCKS:
        gq = np.percentile(girf_store[sh], [5, 50, 95], axis=0)   # (3,H+1,n)
        out = {"h": np.arange(H_IRF + 1)}
        for i, nm in enumerate(names):
            out[f"{nm}_p50"] = gq[1, :, i]
            out[f"{nm}_p05"] = gq[0, :, i]
            out[f"{nm}_p95"] = gq[2, :, i]
        safe = sh.split(",")[0].split(" ")[0]
        pd.DataFrame(out).to_csv(f"bvar_girf_{safe}.csv", index=False, encoding="utf-8-sig")

    with open("bvar_analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write("BVAR(4) 종합 분석 — 36개 산업 실질부가가치 증가율\n")
        f.write(f"사후추출 {NDRAW}회 (안정 draw 비율 {n_stable/NDRAW:.1%}), "
                f"GIRF/GFEVD 순서불변, FEVD 시계 H={H_FEVD}\n\n")
        f.write(f"[총연결성 TCI]  {TCI.mean():.1f}%  "
                f"(90% 신용구간 {np.percentile(TCI,5):.1f}~{np.percentile(TCI,95):.1f})\n")
        f.write("  → 전체 산업 예측오차분산의 이 비율이 '다른 산업'에서 전이됨\n\n")
        f.write("[순(NET) 연결성 상위 — 순(net) 충격 전파자(허브)]\n")
        for _, r in conn.head(8).iterrows():
            f.write(f"  {r['NET']:+6.1f}  {r['산업']}  (주는 {r['TO_주는']:.0f}, 받는 {r['FROM_받는']:.0f})\n")
        f.write("\n[순(NET) 연결성 하위 — 순 충격 수용자]\n")
        for _, r in conn.tail(6).iterrows():
            f.write(f"  {r['NET']:+6.1f}  {r['산업']}  (주는 {r['TO_주는']:.0f}, 받는 {r['FROM_받는']:.0f})\n")
        f.write("\n[자기지속성 상위/하위 (자기 lag 합)]\n")
        ps = pd.DataFrame({"산업": names, "p": pq[2]}).sort_values("p", ascending=False)
        for _, r in ps.head(5).iterrows():
            f.write(f"  {r['p']:+.2f}  {r['산업']}\n")
        f.write("  ...\n")
        for _, r in ps.tail(3).iterrows():
            f.write(f"  {r['p']:+.2f}  {r['산업']}\n")

    print(f"분석 완료: TCI={TCI.mean():.1f}%, 안정draw={n_stable/NDRAW:.1%}, draws={NDRAW}")
    print("저장: bvar_connectedness.csv, bvar_coef_persistence.csv, "
          "bvar_top_spillovers.csv, bvar_girf_*.csv, bvar_analysis_summary.txt")


if __name__ == "__main__":
    main()
