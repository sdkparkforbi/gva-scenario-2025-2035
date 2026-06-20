# -*- coding: utf-8 -*-
"""한국은행 ECOS API로 경제활동별(산업별) GDP를 가장 세부 단위로 받아온다.

- 통계표: 200Y103 = 경제활동별 GDP 및 GNI(계절조정, 명목, 분기)
          200Y104 = 경제활동별 GDP 및 GNI(계절조정, 실질, 분기)
- 가장 작은 세부 단위(leaf) 36개 산업만 추출 (상위 집계·GDP/GNI 등 소득 항목 제외).
- 단위: 십억원. 기간: API 보유 전 구간(분기 GDP와 동일하게 1960Q1~ 가능 범위).
- 산출물:
    gdp_industry_long.csv : 분기 x 산업 long 포맷 (분기,산업코드,산업명,명목_십억원,실질_십억원)
    gdp_industry_coverage.csv : 산업별 데이터 보유 구간(시작/종료/개수)
"""
import csv
import requests

API_KEY = open("API_KEY_BOK.txt", encoding="utf-8").read().strip()
ITEM_URL = "http://ecos.bok.or.kr/api/StatisticItemList/{k}/json/kr/1/500/{t}"
DATA_URL = "http://ecos.bok.or.kr/api/StatisticSearch/{k}/json/kr/1/30000/{t}/Q/1900Q1/2099Q4"
NOM_TBL, REAL_TBL = "200Y103", "200Y104"
# 산업이 아닌 집계·소득 항목 (제외)
AGG = {"1200", "1300", "1400", "1500", "1600", "1700", "1800"}
CUTOFF = "2025Q4"  # 분기 GDP 시계열과 동일하게 최신 신뢰 분기까지만 사용 (2026Q1 제외)
START = "1980Q1"   # 36개 말단 산업이 모두 존재하는 첫 분기 → 결측 없는 균형 패널


def leaf_industries():
    """200Y103 항목 계층에서 말단(자식 없는) 산업 코드만 반환. 표시 순서 유지."""
    rows = requests.get(ITEM_URL.format(k=API_KEY, t=NOM_TBL), timeout=30) \
        .json().get("StatisticItemList", {}).get("row", [])
    order, name, parents = [], {}, set()
    for r in rows:
        ic = r["ITEM_CODE"]
        if ic not in name:
            order.append(ic)
            name[ic] = r["ITEM_NAME"]
        if r.get("P_ITEM_CODE"):
            parents.add(r["P_ITEM_CODE"])
    return [(ic, name[ic]) for ic in order if ic not in parents and ic not in AGG]


def fetch_all(table):
    """표 전체를 {item_code: {TIME: value}} 로 반환."""
    rows = requests.get(DATA_URL.format(k=API_KEY, t=table), timeout=120) \
        .json().get("StatisticSearch", {}).get("row", [])
    out = {}
    for r in rows:
        v = r.get("DATA_VALUE")
        if v in (None, "", "-") or r["TIME"] > CUTOFF or r["TIME"] < START:
            continue
        out.setdefault(r["ITEM_CODE1"], {})[r["TIME"]] = float(v)
    return out


def main():
    leaves = leaf_industries()
    nom, real = fetch_all(NOM_TBL), fetch_all(REAL_TBL)

    # long 포맷 저장
    with open("gdp_industry_long.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["분기", "산업코드", "산업명", "명목_십억원", "실질_십억원"])
        for code, nm in leaves:
            n, rl = nom.get(code, {}), real.get(code, {})
            for t in sorted(set(n) | set(rl)):
                w.writerow([t, code, nm, n.get(t, ""), rl.get(t, "")])

    # 산업별 커버리지
    with open("gdp_industry_coverage.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["산업코드", "산업명", "명목_시작", "명목_종료", "명목_개수",
                    "실질_시작", "실질_종료", "실질_개수"])
        for code, nm in leaves:
            n, rl = sorted(nom.get(code, {})), sorted(real.get(code, {}))
            w.writerow([code, nm,
                        n[0] if n else "", n[-1] if n else "", len(n),
                        rl[0] if rl else "", rl[-1] if rl else "", len(rl)])

    print(f"산업 {len(leaves)}개 저장 완료")
    print(f"{'산업명':<28}{'명목구간':>22}{'실질구간':>22}")
    for code, nm in leaves:
        n, rl = sorted(nom.get(code, {})), sorted(real.get(code, {}))
        ns = f"{n[0]}~{n[-1]}({len(n)})" if n else "없음"
        rs = f"{rl[0]}~{rl[-1]}({len(rl)})" if rl else "없음"
        print(f"{nm[:26]:<28}{ns:>22}{rs:>22}")


if __name__ == "__main__":
    main()
