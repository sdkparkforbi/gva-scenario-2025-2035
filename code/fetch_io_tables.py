# -*- coding: utf-8 -*-
"""한국은행 산업연관표(I-O) 계수행렬을 전 구간 받아온다 (대분류, 2015~2023).

VAR 계수의 prior 정보로 쓰기 위한 산업 간 연계강도 행렬:
  - 투입계수표      : a_ij = 산업 j 한 단위 산출에 직접 투입되는 산업 i 산출 (직접 연계)
  - 생산유발계수표  : (I-A)^-1, 산업 j 최종수요 1단위가 유발하는 산업 i 총산출 (직·간접)

빈티지(기준년)별로 코드가 다름:
  생산유발계수: 271Y010(2015기준,'15~'19) + 271Y112(2020기준,'20~'23)
  투입계수    : 271Y009(2015기준,'15~'19) + 271Y111(2020기준,'20~'23)
ITEM_CODE1/NAME1 = 행(유발되는·투입하는 산업 i), ITEM_CODE2/NAME2 = 열(수요·산출 산업 j).

산출:
  io_생산유발계수_long.csv, io_투입계수_long.csv  (연도,행코드,행산업,열코드,열산업,값,기준년)
  io_sectors.csv  (대분류 산업 목록)
"""
import csv
import requests

API_KEY = open("API_KEY_BOK.txt", encoding="utf-8").read().strip()
URL = "http://ecos.bok.or.kr/api/StatisticSearch/{k}/json/kr/1/100000/{t}/A/1900/2099"
TABLES = {
    "생산유발계수": [("271Y010", "2015기준"), ("271Y112", "2020기준")],
    "투입계수":     [("271Y009", "2015기준"), ("271Y111", "2020기준")],
}


def fetch(table):
    r = requests.get(URL.format(k=API_KEY, t=table), timeout=120).json()
    return r.get("StatisticSearch", {}).get("row", []) or []


def main():
    sectors = {}
    for kind, vintages in TABLES.items():
        out = []
        for code, base in vintages:
            for r in fetch(code):
                v = r.get("DATA_VALUE")
                if v in (None, "", "-"):
                    continue
                out.append([r["TIME"], r["ITEM_CODE1"], r["ITEM_NAME1"],
                            r["ITEM_CODE2"], r["ITEM_NAME2"], float(v), base])
                sectors[r["ITEM_CODE1"]] = r["ITEM_NAME1"]
        fn = f"io_{kind}_long.csv"
        with open(fn, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["연도", "행코드_i", "행산업_i", "열코드_j", "열산업_j", "값", "기준년"])
            w.writerows(out)
        yrs = sorted(set(r[0] for r in out))
        print(f"{kind}: {len(out)}행, {yrs[0]}~{yrs[-1]} ({len(yrs)}개년), "
              f"산업 {len(set(r[1] for r in out))}개 → {fn}")

    with open("io_sectors.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["코드", "산업명"])
        for c in sorted(sectors):
            w.writerow([c, sectors[c]])
    print(f"대분류 산업 {len(sectors)}개 → io_sectors.csv")


if __name__ == "__main__":
    main()
