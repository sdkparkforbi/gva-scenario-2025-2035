# -*- coding: utf-8 -*-
"""index.html을 Playwright(chromium)로 렌더링하여 PDF로 변환. 하단에 페이지번호."""
from playwright.sync_api import sync_playwright

URL = "http://localhost:8731/index.html"
OUT = "gva-repo/gva-scenario-paper.pdf"
FOOTER = ('<div style="font-size:9px;width:100%;text-align:center;color:#888;'
          'padding-top:2px;">- <span class="pageNumber"></span> / '
          '<span class="totalPages"></span> -</div>')

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 900, "height": 1200})
    pg.goto(URL, wait_until="networkidle", timeout=60000)
    pg.wait_for_function("window.__chartsReady===true", timeout=30000)
    # MathJax 렌더 완료 대기 (수식 컨테이너 생성 확인)
    try:
        pg.wait_for_function(
            "document.querySelectorAll('mjx-container').length > 25", timeout=20000)
    except Exception:
        pass
    pg.wait_for_timeout(2500)
    pg.pdf(path=OUT, format="A4", print_background=True,
           margin={"top": "16mm", "bottom": "18mm", "left": "14mm", "right": "14mm"},
           display_header_footer=True,
           header_template="<div></div>",
           footer_template=FOOTER)
    b.close()
print("PDF 생성:", OUT)
