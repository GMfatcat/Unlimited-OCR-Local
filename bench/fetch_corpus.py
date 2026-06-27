"""下載可公開取得的測試文件到 bench/corpus/。抓不到/需特定用途的標 provided。

使用方式：
    python -m bench.fetch_corpus
"""
import os
import re
import urllib.parse
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS = os.path.join(REPO, "bench", "corpus")

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_url(url: str) -> str:
    """Percent-encode non-ASCII / unsafe characters in the URL path only."""
    p = urllib.parse.urlsplit(url)
    safe_path = urllib.parse.quote(p.path, safe="/:@!$&'()*+,;=.-_~")
    return urllib.parse.urlunsplit((p.scheme, p.netloc, safe_path, p.query, p.fragment))


def _download(url: str, dest: str) -> None:
    """Stream-download *url* → *dest* with desktop User-Agent.  Cleans up the
    partial file if an error occurs mid-download."""
    encoded_url = _safe_url(url)
    req = urllib.request.Request(encoded_url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as f:
            while chunk := resp.read(1 << 16):  # 64 KB chunks
                f.write(chunk)
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)
        raise


def _fetch_patent_pdf(category: str, name: str, patent_url: str, patent_number: str) -> None:
    """Parse a Google Patents HTML page, extract the patentimages.storage.googleapis.com
    PDF link via regex, then download it.  Falls back to the USPTO direct-download
    endpoint if no patentimages link is found in the HTML."""
    d = os.path.join(CORPUS, category)
    os.makedirs(d, exist_ok=True)
    dest = os.path.join(d, name)

    # 1. Fetch Google Patents HTML with desktop UA
    req = urllib.request.Request(patent_url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    # 2. Regex for patentimages PDF URL embedded anywhere in the page source
    m = re.search(
        r'https://patentimages\.storage\.googleapis\.com/[^\s"\'<>]+\.pdf',
        html,
    )
    if m:
        pdf_url = m.group(0)
    else:
        # Fallback: USPTO direct download (best-effort; streamed, not always complete)
        num = patent_number.upper()
        pdf_url = (
            f"https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/{num}"
        )
        print(f"  [patent] no patentimages link in HTML, trying USPTO fallback: {num}")

    _download(pdf_url, dest)


# ── Direct-download list: (category, filename, url) ───────────────────────────
#
# Sources verified in .superpowers/sdd/corpus-sources.md (2026-06-27).
# Update URLs here if they rot; never abort the whole run on a single failure.

PUBLIC_DIRECT = [
    # HK CSB 政府公文寫作手冊 — born-digital 繁中, text-heavy → pseudo-GT possible
    (
        "cjk",
        "hk_csb_circulars.pdf",
        "https://www.csb.gov.hk/tc_chi/publications_stat/publication/files/circulars_2ed.pdf",
    ),
    # 台灣主計總處 國情統計通報 040 — 繁中＋統計表＋圖混排
    (
        "cjk",
        "tw_stat_040.pdf",
        "https://www.civil.taichung.gov.tw/media/656044/國情統計通報-第040號.pdf",
    ),
    # US Treasury FSOC 2025 Annual Report — dense financial tables, large file
    (
        "tables",
        "fsoc2025.pdf",
        "https://home.treasury.gov/system/files/261/FSOC2025AnnualReport.pdf",
    ),
    # Internet Archive scanned annual report — follows 302 redirect; urllib handles automatically
    (
        "scanned",
        "ia_annual_report.pdf",
        "https://archive.org/download/annualreportofse00unit/annualreportofse00unit.pdf",
    ),
]

# ── Patent downloads (HTML → patentimages PDF link → download) ────────────────
#
# Per corpus-sources.md: do NOT construct hash paths manually — parse from HTML.
# Format: (category, filename, google_patents_url, patent_number)

PATENT_PAGES = [
    # US6556710B2 — 多欄多圖、閱讀順序複雜
    (
        "patents",
        "us6556710b2.pdf",
        "https://patents.google.com/patent/US6556710B2/en",
        "US6556710B2",
    ),
    # US8110241B2 — 表格＋示意圖
    (
        "tables",
        "us8110241b2.pdf",
        "https://patents.google.com/patent/US8110241B2/en",
        "US8110241B2",
    ),
    # US3930271A — 1976 掃描打字稿; 無乾淨文字層、歪斜雜訊
    (
        "scanned",
        "us3930271a.pdf",
        "https://patents.google.com/patent/US3930271A/en",
        "US3930271A",
    ),
]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    fetched: list[str] = []
    failed: list[str] = []

    # Direct downloads ─────────────────────────────────────────────────────────
    for cat, name, url in PUBLIC_DIRECT:
        d = os.path.join(CORPUS, cat)
        os.makedirs(d, exist_ok=True)
        dest = os.path.join(d, name)
        label = f"{cat}/{name}"
        if os.path.exists(dest):
            print(f"skip (exists): {label}")
            fetched.append(f"{label} (cached)")
            continue
        print(f"fetching: {label} …")
        try:
            _download(url, dest)
            print(f"  OK: {label}")
            fetched.append(label)
        except Exception as exc:
            print(f"  FAILED {label}: {exc}")
            failed.append(label)

    # Patent downloads ─────────────────────────────────────────────────────────
    for cat, name, patent_url, patent_number in PATENT_PAGES:
        d = os.path.join(CORPUS, cat)
        os.makedirs(d, exist_ok=True)
        dest = os.path.join(d, name)
        label = f"{cat}/{name}"
        if os.path.exists(dest):
            print(f"skip (exists): {label}")
            fetched.append(f"{label} (cached)")
            continue
        print(f"fetching patent: {label} ({patent_number}) …")
        try:
            _fetch_patent_pdf(cat, name, patent_url, patent_number)
            print(f"  OK: {label}")
            fetched.append(label)
        except Exception as exc:
            print(f"  FAILED {label}: {exc}")
            failed.append(label)

    # Summary ──────────────────────────────────────────────────────────────────
    newly_fetched = [s for s in fetched if "(cached)" not in s]
    cached = [s for s in fetched if "(cached)" in s]

    print("\n=== 下載摘要 ===")
    print(f"成功 {len(newly_fetched)} / 失敗 {len(failed)} / 已快取 {len(cached)}")
    for s in newly_fetched:
        print(f"  ✓ {s}")
    for s in cached:
        print(f"  ⏭  {s}")
    for f in failed:
        print(f"  ✗ {f}")

    print(
        "\n提醒：以下類別需使用者自備（無公開穩定直連）："
        "\n  • scanned-real  ：手機翻拍/嚴重歪斜的真實辦公掃描"
        "\n  • cjk-presentation：中文簡報/投影片版面（含多欄+圖框）"
        "\n  • tables-10K    ：上市公司 10-K/年報密集表格頁"
        "\n請放到 bench/corpus/<類別>/ 並在 bench/corpus/manifest.yaml 登記。"
    )


if __name__ == "__main__":
    main()
