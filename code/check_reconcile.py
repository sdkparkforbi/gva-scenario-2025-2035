# -*- coding: utf-8 -*-
"""산업 합 = 총부가가치 = GDP 일치 검증 (명목·실질, 1980Q1~2025Q4).

명목: 36개 산업 합 == 총부가가치(1200), + 순생산물세(1300) == GDP(1400)  → 정확히 일치 기대.
실질: 연쇄가중(2020 기준) 볼륨지수는 가법성이 없어 합 != 총부가가치 (연쇄불일치, chain residual).
       → 2020년 부근에서 0에 수렴, 멀어질수록 괴리 커지는지 확인.
"""
import requests
import pandas as pd

API_KEY = open("API_KEY_BOK.txt", encoding="utf-8").read().strip()
URL = "http://ecos.bok.or.kr/api/StatisticSearch/{k}/json/kr/1/30000/{t}/Q/1980Q1/2025Q4"
LEAF = {  # 36개 말단 산업 코드
    "1101", "1102", "110301", "110302", "110303", "110304", "110305", "110306",
    "110307", "110308", "110309", "110310", "110311", "110312", "110313",
    "110401", "110402", "110403", "11051", "11052", "11053", "11054",
    "110601", "110602", "1107", "1108", "1109", "11141", "11142",
    "111501", "111502", "1110", "1111", "1112", "11131", "11132",
}


def load(table):
    rows = requests.get(URL.format(k=API_KEY, t=table), timeout=120) \
        .json().get("StatisticSearch", {}).get("row", [])
    rec = {}
    for r in rows:
        v = r.get("DATA_VALUE")
        if v in (None, "", "-"):
            continue
        rec.setdefault(r["ITEM_CODE1"], {})[r["TIME"]] = float(v)
    return rec


def reconcile(table, label):
    rec = load(table)
    leaf_sum = pd.DataFrame({c: rec[c] for c in LEAF if c in rec}).sort_index().sum(axis=1)
    gva = pd.Series(rec.get("1200", {}))          # 총부가가치(기초가격)
    tax = pd.Series(rec.get("1300", {}))          # 순생산물세
    gdp = pd.Series(rec.get("1400", {}))          # GDP(시장가격)
    df = pd.DataFrame({"leaf_sum": leaf_sum, "GVA": gva, "tax": tax, "GDP": gdp}).dropna()
    df["합-GVA"] = df["leaf_sum"] - df["GVA"]
    df["GVA+세-GDP"] = df["GVA"] + df["tax"] - df["GDP"]
    df["합대비_GVA_괴리%"] = (df["합-GVA"] / df["GVA"] * 100)
    print(f"\n===== {label} ({table}) =====")
    print(f"  [산업합 - 총부가가치]  최대절대오차 = {df['합-GVA'].abs().max():.3f} 십억원")
    print(f"  [총부가가치+순생산물세 - GDP]  최대절대오차 = {df['GVA+세-GDP'].abs().max():.4f} 십억원")
    print(f"  연쇄불일치(합/GVA 괴리%)  2020 부근 vs 양끝:")
    for q in ["1980Q1", "2000Q1", "2019Q4", "2020Q1", "2020Q4", "2025Q4"]:
        if q in df.index:
            print(f"      {q}: {df.loc[q, '합대비_GVA_괴리%']:+.3f}%")
    return df


if __name__ == "__main__":
    reconcile("200Y103", "명목")
    reconcile("200Y104", "실질(연쇄, 2020=100)")
