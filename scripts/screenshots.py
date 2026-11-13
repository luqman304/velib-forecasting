#!/usr/bin/env python3
"""Capture the README screenshots from a running app (streamlit run app/app.py).

    pip install playwright && playwright install chromium-headless-shell
    python scripts/screenshots.py --station "Mairie du 11ème" --origin 2021-03-09T07:00
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def settle(page, ms: int = 2500) -> None:
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=60_000)
    page.wait_for_timeout(ms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--station", default="Mairie du 11ème")
    ap.add_argument("--origin", default="2021-03-09T07:00")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    url = f"{args.url}/?{urlencode({'station': args.station, 'origin': args.origin})}"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=2,
                                color_scheme="light")
        page.goto(url)
        page.wait_for_selector("text=Forecasts from one origin", timeout=90_000)
        settle(page, 6000)  # map tiles
        page.screenshot(path=OUT / "forecast.png", full_page=False)
        # Streamlit scrolls an inner container, so use a tall viewport and screenshot elements instead.
        page.set_viewport_size({"width": 1440, "height": 4000})
        settle(page, 3000)
        page.locator('[data-testid="stColumn"]').filter(has_text="Rolling view").last.screenshot(path=OUT / "forecast-station.png")
        main = page.locator('[data-testid="stMainBlockContainer"]')
        for tab, name in (("Evaluation", "evaluation"), ("Data quality", "data-quality"), ("Live feed", "live-feed")):
            page.get_by_role("tab", name=tab).click()
            settle(page)
            main.screenshot(path=OUT / f"{name}.png")
        browser.close()
    print(f"screenshots written to {OUT}")


if __name__ == "__main__":
    main()
