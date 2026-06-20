# -*- coding: utf-8 -*-
"""VARX 외생변수 구축 — 정상성 변환된 거시 외생 블록 (분기).

변수와 변환 (모두 정상성 지향):
  유가     oil_g   : 100·Δlog(Brent, USD/bbl, 분기평균)        [FRED MCOILBRENTEU]
  노동     lab_g   : 100·Δlog(취업자수, 분기평균)              [ECOS 901Y027 I61BA]
  교역량   trd_g   : 100·Δlog((수출물량+수입물량)/2, 분기평균) [ECOS 403Y002/403Y004 *AA]
  실질금리 rrate   : 회사채(3년,AA-) − CPI 전년동기 인플레이션  [ECOS 721Y001 7020000, 901Y009]
  실질환율 rfx_g   : 100·Δlog(실질 원/달러),
                     실질원달러 = 명목(원/달러) × 미국CPI / 한국CPI [731Y006, 902Y008, 901Y009]

월자료는 분기평균으로 집계. 정상성은 ADF로 검정해 레벨/차분을 선택.
산출: exog_quarterly.csv (분기 x 변수), exog_coverage_adf.csv
"""
import io
import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.stattools import adfuller

KEY = open("API_KEY_BOK.txt", encoding="utf-8").read().strip()
BASE = "http://ecos.bok.or.kr/api/StatisticSearch/{k}/json/kr/1/100000/{t}/M/190001/209912/{it}"


def ecos_m(table, *items):
    """월자료 → pandas Series (index=Timestamp, 분기말 아님)."""
    url = BASE.format(k=KEY, t=table, it="/".join(items))
    rows = requests.get(url, timeout=120).json().get("StatisticSearch", {}).get("row", [])
    s = {}
    for r in rows:
        v = r.get("DATA_VALUE")
        if v in (None, "", "-"):
            continue
        t = r["TIME"]
        s[pd.Timestamp(int(t[:4]), int(t[4:6]), 1)] = float(v)
    return pd.Series(s).sort_index()


def to_q(s):
    """월 → 분기평균, index를 'YYYYQq' 문자열로."""
    q = s.resample("QE").mean()
    return pd.Series(q.values, index=[f"{d.year}Q{(d.month-1)//3+1}" for d in q.index])


def fred_brent():
    txt = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MCOILBRENTEU", timeout=30).text
    df = pd.read_csv(io.StringIO(txt))
    df.columns = ["date", "v"]
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["v"] != "."]
    return pd.Series(df["v"].astype(float).values,
                     index=df["date"]).sort_index()


def qspan(start="1980Q1", end="2025Q4"):
    return [f"{y}Q{q}" for y in range(int(start[:4]), int(end[:4]) + 1) for q in range(1, 5)
            if f"{y}Q{q}" >= start and f"{y}Q{q}" <= end]


def adf_p(x):
    x = x.dropna()
    return adfuller(x, autolag="AIC")[1] if len(x) > 12 else np.nan


def main():
    # 원자료(분기) 적재
    oil = to_q(fred_brent())                                   # Brent USD
    lab = to_q(ecos_m("901Y027", "I61BA"))                     # 취업자수
    xq = to_q(ecos_m("403Y002", "*AA"))                        # 수출물량
    mq = to_q(ecos_m("403Y004", "*AA"))                        # 수입물량
    trd = (xq + mq) / 2                                        # 교역량(평균)
    nrate = to_q(ecos_m("721Y001", "7020000"))                # 회사채3년 명목
    krcpi = to_q(ecos_m("901Y009", "0"))                      # 한국 CPI
    uscpi = to_q(ecos_m("902Y008", "US"))                     # 미국 CPI
    nfx = to_q(ecos_m("731Y006", "0000003", "0000100"))       # 명목 원/달러(월평균)

    idx = qspan()
    R = lambda s: s.reindex(idx)

    # 변환
    g = lambda s: 100 * np.log(R(s)).diff()                    # 100·Δlog
    cpi_yoy = 100 * (R(krcpi) / R(krcpi).shift(4) - 1)         # 전년동기 인플레이션
    rrate = R(nrate) - cpi_yoy                                 # 실질금리(ex-post)
    rfx = R(nfx) * R(uscpi) / R(krcpi)                         # 실질 원/달러
    rfx_g = 100 * np.log(rfx).diff()                           # 실질환율 변화율

    out = pd.DataFrame({
        "oil_g": g(oil), "lab_g": g(lab), "trd_g": g(trd),
        "d_rrate": rrate.diff(),        # 실질금리 변화 (레벨이 비정상이라 차분)
        "rfx_g": rfx_g,                 # 실질환율 변화율
        "rrate_lvl": rrate, "rfx_lvl": np.log(rfx) * 100,   # 참고용(비정상)
    }, index=idx)

    # 커버리지 + ADF (레벨/차분 후보 모두)
    cov = []
    for c in out.columns:
        s = out[c].dropna()
        cov.append([c, s.index[0] if len(s) else "", s.index[-1] if len(s) else "",
                    len(s), round(adf_p(out[c]), 4)])
    cov = pd.DataFrame(cov, columns=["변수", "시작", "종료", "관측수", "ADF_p(레벨/현재변환)"])

    out.to_csv("exog_quarterly.csv", encoding="utf-8-sig")
    cov.to_csv("exog_coverage_adf.csv", index=False, encoding="utf-8-sig")

    print(cov.to_string(index=False))
    # 공통표본(노동 포함 / 제외)
    core = ["oil_g", "trd_g", "d_rrate", "rfx_g"]
    allv = core + ["lab_g"]
    cs_all = out[allv].dropna().index
    cs_core = out[core].dropna().index
    print(f"\n공통표본(5변수, 노동포함): {cs_all[0]} ~ {cs_all[-1]}  ({len(cs_all)}분기)")
    print(f"공통표본(4변수, 노동제외): {cs_core[0]} ~ {cs_core[-1]}  ({len(cs_core)}분기)")
    print("저장: exog_quarterly.csv, exog_coverage_adf.csv")


if __name__ == "__main__":
    main()
