# -*- coding: utf-8 -*-
"""한국은행 ECOS API로 계절조정 분기 GDP 시계열을 받아온다.

확보 시계열 (1960Q1 ~ 최신, 단위: 십억원 / 디플레이터는 지수):
  - 명목 GDP        : 200Y107 (국내총생산에 대한 지출, 계절조정, 명목, 분기)
  - 실질 GDP        : 200Y108 (국내총생산에 대한 지출, 계절조정, 실질, 분기)
  - GDP 디플레이터  : 명목/실질 x 100 으로 산출 (계절조정 시계열 일관 / 2020=100)
  항목코드 10601 = 국내총생산에 대한 지출(= GDP 총계)
"""
import csv
import requests

API_KEY = open("API_KEY_BOK.txt", encoding="utf-8").read().strip()
BASE = "http://ecos.bok.or.kr/api/StatisticSearch"
GDP_ITEM = "10601"
START, END = "1900Q1", "2099Q4"  # API 조회 범위 (실제 보유분만 반환)
# 2026Q1 계절조정 명목 GDP는 비정상값(직전 분기 대비 +10.5%, 디플레이터 급등)이라 제외.
# 신뢰 가능한 최신 분기까지만 사용.
CUTOFF = "2025Q4"
# 산업별 균형 패널(36개 말단 산업이 모두 존재하는 첫 분기)과 분석 기간을 일치.
START_USE = "1980Q1"


def fetch(table):
    url = f"{BASE}/{API_KEY}/json/kr/1/9999/{table}/Q/{START}/{END}/{GDP_ITEM}"
    rows = requests.get(url, timeout=60).json().get("StatisticSearch", {}).get("row", [])
    return {r["TIME"]: float(r["DATA_VALUE"]) for r in rows}


def build():
    nominal = fetch("200Y107")   # 계절조정 명목
    real = fetch("200Y108")      # 계절조정 실질
    rows = []
    for t in sorted(nominal):
        if t > CUTOFF or t < START_USE:
            continue
        nom, rl = nominal[t], real.get(t)
        defl = round(nom / rl * 100, 4) if rl else None  # GDP 디플레이터(2020=100)
        rows.append((t, nom, rl, defl))
    return rows


if __name__ == "__main__":
    rows = build()
    with open("gdp_quarterly_sa.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["분기", "명목GDP_십억원", "실질GDP_십억원", "GDP디플레이터_2020=100"])
        w.writerows(rows)
    print(f"{len(rows)}개 분기 저장: gdp_quarterly_sa.csv ({rows[0][0]} ~ {rows[-1][0]})")
    print("\n최근 8분기:")
    print(f"{'분기':<8}{'명목GDP':>14}{'실질GDP':>14}{'디플레이터':>12}")
    for t, nom, rl, defl in rows[-8:]:
        print(f"{t:<8}{nom:>14,.1f}{rl:>14,.1f}{defl:>12,.2f}")
