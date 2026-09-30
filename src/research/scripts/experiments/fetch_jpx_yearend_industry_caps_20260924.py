#!/usr/bin/env python3
"""Fetch and parse JPX annual December industry market-cap PDFs (research only).

Requires pypdf in the project research environment. This is a data-preparation
utility and does not add a runtime dependency to the production package.
"""
from __future__ import annotations

from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.request import urlopen, urlretrieve

import pandas as pd
import yaml
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[4]
BASE = "https://www.jpx.co.jp"
PDF_DIR = ROOT / "var/research/sensitivity_jpx33/yearend_pdfs"
CSV_PATH = ROOT / "configs/research/jpx33_yearend_market_cap_2014_2025.csv"
SOURCE_PATH = ROOT / "configs/research/jpx33_yearend_market_cap_2014_2025.sources.json"
BASE_CONFIG = ROOT / "configs/research/jpx33_sensitivity_prior_2017.yaml"
SOURCE_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/monthly/"
ARCHIVES = {year: f"/markets/statistics-equities/monthly/00-archives-{2026-year:02d}.html" for year in range(2016, 2026)}
FIXED_URLS = {
    2014: "https://www.jpx.co.jp/files/tse/english/market/data/geppo/b7gje6000002aeyz-att/04_jikaso1412.pdf",
    2015: "https://www.jpx.co.jp/markets/statistics-equities/monthly/nlsgeu000001ekco-att/04_jikaso1512-.pdf",
    2016: "https://www.jpx.co.jp/markets/statistics-equities/monthly/nlsgeu00000280ko-att/04_jikaso1612.pdf",
    2017: "https://www.jpx.co.jp/markets/statistics-equities/misc/nlsgeu000002vgd0-att/201712.pdf",
}


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        href = dict(attrs).get("href")
        if tag == "a" and href:
            self.links.append(href)


def _source_url(year: int) -> str:
    if year in FIXED_URLS:
        return FIXED_URLS[year]
    with urlopen(BASE + ARCHIVES[year], timeout=30) as response:
        html = response.read().decode("utf-8", "replace")
    parser = _LinkParser()
    parser.feed(html)
    needle = f"04_jikaso{year % 100:02d}12"
    matches = [href for href in parser.links if needle in href.lower()]
    if len(matches) != 1:
        raise ValueError(f"{year}: expected one December market-cap PDF, found {matches}")
    href = matches[0]
    return href if href.startswith("http") else BASE + href


def _normalize_industry_prefix(value: str) -> str:
    # Some older JPX PDFs insert ASCII spaces between each Japanese character.
    value = re.sub(r"(?<=[ぁ-ゖァ-ヺ一-龯・、ー])\s+(?=[ぁ-ゖァ-ヺ一-龯・、ー])", "", value)
    return re.sub(r"[\u3000]+", "", value).strip()


def _parse_pdf(path: Path, year: int, names: list[str]) -> dict[str, int]:
    reader = PdfReader(str(path))
    table_text = None
    for page in reader.pages:
        text = page.extract_text() or ""
        if "Market Capitalization by Industry Sector" in text or "株式時価総額" in text:
            table_text = text
            break
    if table_text is None:
        raise ValueError(f"{year}: industry market-cap table page not found in {path}")
    values: dict[str, int] = {}
    for raw_line in table_text.splitlines():
        compact = _normalize_industry_prefix(raw_line)
        for name in names:
            normalized_name = _normalize_industry_prefix(name)
            if not compact.startswith(normalized_name):
                continue
            tail = compact[len(normalized_name) :]
            tokens = re.findall(r"\d[\d,]*|[－—―ｰー-]", tail)
            # 2017's summary table has count then market cap. Older/later
            # monthly formats include listed shares between those two fields.
            token_index = 1 if year == 2017 else 2
            if len(tokens) <= token_index:
                raise ValueError(f"{year}: market-cap column missing: {raw_line}")
            token = tokens[token_index]
            values[name] = int(token.replace(",", "")) if token not in {"－", "—", "―", "ｰ", "ー", "-"} else 0
            break
    missing = sorted(set(names) - set(values))
    if missing:
        raise ValueError(f"{year}: missing industry rows: {missing}")
    return values


def main() -> None:
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(BASE_CONFIG)
    raw = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    industry_table = raw["industries"]
    names = [str(row[0]) for row in industry_table["rows"]]
    if len(names) != 33 or len(set(names)) != 33:
        raise ValueError("Reference config must contain 33 unique industries")
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"source_page": SOURCE_PAGE, "universe_by_year": {}, "files": {}}
    records = []
    for year in range(2014, 2026):
        url = _source_url(year)
        path = PDF_DIR / f"{year}.pdf"
        urlretrieve(url, path)
        data = path.read_bytes()
        reader = PdfReader(str(path))
        caps = _parse_pdf(path, year, names)
        records.extend({"snapshot_year": year, "industry": name, "market_cap_million_yen": caps[name]} for name in names)
        manifest["universe_by_year"][str(year)] = "TSE First Section" if year <= 2021 else "TSE Prime"
        manifest["files"][str(year)] = {
            "url": url,
            "sha256": sha256(data).hexdigest(),
            "bytes": len(data),
            "pages": len(reader.pages),
        }
    reference_rows = [dict(zip(industry_table["columns"], row, strict=True)) for row in industry_table["rows"]]
    expected_2017 = {str(row["name"]): int(row["market_cap"]) for row in reference_rows}
    parsed_2017 = {row["industry"]: int(row["market_cap_million_yen"]) for row in records if row["snapshot_year"] == 2017}
    if parsed_2017 != expected_2017:
        differences = {name: (expected_2017[name], parsed_2017.get(name)) for name in names if expected_2017[name] != parsed_2017.get(name)}
        raise ValueError(f"2017 parser validation failed: {differences}")
    pd.DataFrame(records).to_csv(CSV_PATH, index=False)
    SOURCE_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} rows to {CSV_PATH}")
    print(f"Source manifest: {SOURCE_PATH}; 2017 validation: {len(parsed_2017)}/33 rows matched")


if __name__ == "__main__":
    main()
