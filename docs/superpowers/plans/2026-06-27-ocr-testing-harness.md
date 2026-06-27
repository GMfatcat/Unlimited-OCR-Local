# OCR Testing Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-track (quality + stability) testing harness for Unlimited-OCR with a curated corpus and reports, per the approved spec.

**Architecture:** Pure, unit-tested helper modules (`metrics`, `corpus`) plus integration runners (`runner`, `quality`, `stability`) that drive the live SGLang server through the existing `ocr_backends`. Quality is difficulty-tiered (auto pseudo-GT for `simple`, human review bundle for `hard`); stability covers no-crash / loop-guard / long-run / recovery, with a tok/s + GPU-mem line chart.

**Tech Stack:** Python 3.12 (`.venv-sglang`), PyMuPDF (fitz), rapidfuzz, PyYAML, matplotlib, pytest; reuses `ocr_backends.py`.

## Global Constraints

- All commands run **inside WSL Ubuntu-24.04** using the SGLang venv python: `~/uocr/.venv-sglang/bin/python`.
- Repo root in WSL: `/mnt/c/Users/User/Desktop/project/unlimited-ocr` (referred to as `$REPO`).
- Integration tasks require the **SGLang server running** (`triton` backend) on `http://127.0.0.1:10000`; start via `scripts/wsl/03_start_sglang_server.sh` with `ATTN_BACKEND=triton`.
- Reuse `ocr_backends` functions: `sglang_stream`, `strip_markup`, `parse_dets`, `draw_dets`, `load_display_image`, `to_image_paths`. Do **not** reimplement them.
- Quality thresholds are **informational, not hard pass/fail gates** (CER < 0.15, coverage > 0.9 for `simple`).
- Defaults: `image_mode` from manifest (PDF→`base`), `max_tokens=4096`, `page_timeout=30`, `ngram_size=35`, `ngram_window=1024`.
- New deps go into `.venv-sglang`: `rapidfuzz`, `pyyaml`, `matplotlib`, `pytest`.
- `bench/corpus/*.pdf` and `bench/reports/` are **gitignored**; `manifest.yaml` and harness code are committed.
- Tests import `ocr_backends` from repo root → tests must `sys.path.insert(0, <repo root>)`.

---

### Task 1: Bench skeleton + metrics module (pure, TDD)

**Files:**
- Create: `bench/__init__.py` (empty), `bench/metrics.py`, `bench/tests/__init__.py` (empty), `bench/tests/conftest.py`, `bench/tests/test_metrics.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `bench/metrics.py` with
  - `normalize(text: str, lower: bool=False) -> str`
  - `cer(gt: str, pred: str) -> float`
  - `coverage(gt: str, pred: str) -> float`
  - `order_insensitive_sim(gt: str, pred: str) -> float`  (0..1)

- [ ] **Step 1: Install deps + create dirs**

```bash
wsl -d Ubuntu-24.04 bash -lc 'export PATH=$HOME/.local/bin:$PATH; export UV_HTTP_TIMEOUT=600; uv pip install --python ~/uocr/.venv-sglang/bin/python rapidfuzz pyyaml matplotlib pytest'
mkdir -p /mnt/c/Users/User/Desktop/project/unlimited-ocr/bench/tests
touch /mnt/c/Users/User/Desktop/project/unlimited-ocr/bench/__init__.py /mnt/c/Users/User/Desktop/project/unlimited-ocr/bench/tests/__init__.py
```

- [ ] **Step 2: Add gitignore entries**

Append to `.gitignore`:
```
# bench corpus + reports (large / generated)
bench/corpus/**/*.pdf
bench/corpus/**/*.png
bench/reports/
```

- [ ] **Step 3: conftest.py to put repo root on sys.path**

Create `bench/tests/conftest.py`:
```python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
```

- [ ] **Step 4: Write the failing test**

Create `bench/tests/test_metrics.py`:
```python
from bench import metrics


def test_normalize_strips_markup_and_collapses_space():
    out = metrics.normalize("<|det|>title [1, 2, 3, 4]<|/det|>Hello   World\n\n")
    assert "det" not in out
    assert out == "Hello World"


def test_cer_identical_is_zero():
    assert metrics.cer("hello world", "hello world") == 0.0


def test_cer_one_substitution_quarter():
    assert abs(metrics.cer("abcd", "abxd") - 0.25) < 1e-9


def test_cer_empty_gt_empty_pred_zero():
    assert metrics.cer("", "") == 0.0


def test_coverage_half():
    assert metrics.coverage("a b c d", "a b zzz") == 0.5


def test_order_insensitive_high_when_reordered():
    assert metrics.order_insensitive_sim("alpha beta gamma", "gamma beta alpha") > 0.99
```

- [ ] **Step 5: Run test to verify it fails**

Run:
```bash
wsl -d Ubuntu-24.04 bash -lc 'cd /mnt/c/Users/User/Desktop/project/unlimited-ocr && ~/uocr/.venv-sglang/bin/python -m pytest bench/tests/test_metrics.py -q'
```
Expected: FAIL (ModuleNotFoundError: bench.metrics).

- [ ] **Step 6: Implement metrics.py**

Create `bench/metrics.py`:
```python
"""純函式品質指標（可單元測試）。GT 為 PyMuPDF 抽出的文字層 pseudo-GT。"""
import os
import re
import sys
import unicodedata

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ocr_backends import strip_markup  # noqa: E402


def normalize(text, lower=False):
    t = strip_markup(text or "")
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("-\n", "")            # 去連字斷行
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower() if lower else t


def cer(gt, pred):
    g, p = normalize(gt), normalize(pred)
    if not g:
        return 0.0 if not p else 1.0
    return Levenshtein.distance(g, p) / len(g)


def _tokens(text):
    return [w for w in normalize(text).split(" ") if w]


def coverage(gt, pred):
    gt_toks = _tokens(gt)
    if not gt_toks:
        return 1.0
    pred_set = set(_tokens(pred))
    return sum(1 for w in gt_toks if w in pred_set) / len(gt_toks)


def order_insensitive_sim(gt, pred):
    return fuzz.token_sort_ratio(normalize(gt), normalize(pred)) / 100.0
```

- [ ] **Step 7: Run test to verify it passes**

Run:
```bash
wsl -d Ubuntu-24.04 bash -lc 'cd /mnt/c/Users/User/Desktop/project/unlimited-ocr && ~/uocr/.venv-sglang/bin/python -m pytest bench/tests/test_metrics.py -q'
```
Expected: PASS (6 passed).

- [ ] **Step 8: Commit**

```bash
git add bench/ .gitignore && git commit -m "test(bench): metrics module (CER/coverage/order-insensitive)"
```

---

### Task 2: Corpus module — manifest + page parsing + pseudo-GT (pure, TDD)

**Files:**
- Create: `bench/corpus_io.py`, `bench/tests/test_corpus_io.py`

**Interfaces:**
- Consumes: PyMuPDF (`fitz`).
- Produces: `bench/corpus_io.py` with
  - `DocEntry` dataclass: `id, file, category, tier, source, pages, image_mode, gt, notes`
  - `load_manifest(path: str) -> list[DocEntry]`  (applies defaults: `image_mode="base"`, `gt="pdf_text"`, `notes=""`, `source=""`, `pages="all"`)
  - `parse_pages(spec: str, total: int) -> list[int]`  (0-based page indices)
  - `pdf_pseudo_gt(pdf_path: str, pages: str) -> str`

- [ ] **Step 1: Write the failing test**

Create `bench/tests/test_corpus_io.py`:
```python
import os

from bench import corpus_io

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_parse_pages_all():
    assert corpus_io.parse_pages("all", 3) == [0, 1, 2]


def test_parse_pages_range():
    assert corpus_io.parse_pages("1-3", 10) == [0, 1, 2]


def test_parse_pages_single():
    assert corpus_io.parse_pages("4", 10) == [3]


def test_load_manifest_applies_defaults(tmp_path):
    m = tmp_path / "manifest.yaml"
    m.write_text(
        "- id: doc1\n"
        "  file: academic/x.pdf\n"
        "  category: academic\n"
        "  tier: simple\n",
        encoding="utf-8",
    )
    entries = corpus_io.load_manifest(str(m))
    assert entries[0].id == "doc1"
    assert entries[0].image_mode == "base"
    assert entries[0].gt == "pdf_text"
    assert entries[0].pages == "all"


def test_pdf_pseudo_gt_extracts_text():
    gt = corpus_io.pdf_pseudo_gt(os.path.join(REPO, "Unlimited-OCR.pdf"), "1")
    assert "Abstract" in gt
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
wsl -d Ubuntu-24.04 bash -lc 'cd /mnt/c/Users/User/Desktop/project/unlimited-ocr && ~/uocr/.venv-sglang/bin/python -m pytest bench/tests/test_corpus_io.py -q'
```
Expected: FAIL (ModuleNotFoundError: bench.corpus_io).

- [ ] **Step 3: Implement corpus_io.py**

Create `bench/corpus_io.py`:
```python
"""語料 manifest 載入、頁碼解析、pseudo-GT 抽取。"""
from dataclasses import dataclass

import fitz  # PyMuPDF
import yaml

_DEFAULTS = {"source": "", "pages": "all", "image_mode": "base", "gt": "pdf_text", "notes": ""}


@dataclass
class DocEntry:
    id: str
    file: str
    category: str
    tier: str
    source: str = ""
    pages: str = "all"
    image_mode: str = "base"
    gt: str = "pdf_text"
    notes: str = ""


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []
    return [DocEntry(**{**_DEFAULTS, **d}) for d in raw]


def parse_pages(spec, total):
    spec = str(spec).strip()
    if spec == "all":
        return list(range(total))
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a) - 1, min(int(b), total)))
    return [int(spec) - 1]


def pdf_pseudo_gt(pdf_path, pages):
    doc = fitz.open(pdf_path)
    idxs = parse_pages(pages, doc.page_count)
    text = "\n".join(doc[i].get_text() for i in idxs)
    doc.close()
    return text
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
wsl -d Ubuntu-24.04 bash -lc 'cd /mnt/c/Users/User/Desktop/project/unlimited-ocr && ~/uocr/.venv-sglang/bin/python -m pytest bench/tests/test_corpus_io.py -q'
```
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add bench/corpus_io.py bench/tests/test_corpus_io.py && git commit -m "test(bench): corpus_io manifest + pages + pseudo-GT"
```

---

### Task 3: Shared runner — run one doc through SGLang (integration)

**Files:**
- Create: `bench/runner.py`

**Interfaces:**
- Consumes: `ocr_backends.sglang_stream/strip_markup/parse_dets/draw_dets/load_display_image`, PyMuPDF, `bench.corpus_io.parse_pages`.
- Produces: `bench/runner.py` with
  - `gpu_mem_mib() -> int`
  - `render_pages(pdf_path: str, pages: str, dpi: int=300) -> list[str]`  (PNG paths)
  - `PageResult` dataclass: `idx, tokens, decode_s, clean, raw, overlay_path, flag` (`flag` in `"", "timeout", "capped"`)
  - `run_page(image_path, server_url, image_mode, max_tokens, page_timeout, ngram_window, ngram_size, out_dir, idx) -> PageResult`

- [ ] **Step 1: Implement runner.py**

Create `bench/runner.py`:
```python
"""把單頁/單文件丟 SGLang server 並存產出（品質軌與穩定軌共用）。"""
import os
import subprocess
import time
from dataclasses import dataclass

import fitz

from bench.corpus_io import parse_pages
from ocr_backends import draw_dets, load_display_image, parse_dets, sglang_stream, strip_markup


def gpu_mem_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def render_pages(pdf_path, pages, dpi=300):
    doc = fitz.open(pdf_path)
    idxs = parse_pages(pages, doc.page_count)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    out_dir = os.path.join(os.path.dirname(pdf_path), "_pages_" + os.path.basename(pdf_path))
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in idxs:
        p = os.path.join(out_dir, f"page_{i + 1:04d}.png")
        doc[i].get_pixmap(matrix=mat).save(p)
        paths.append(p)
    doc.close()
    return paths


@dataclass
class PageResult:
    idx: int
    tokens: int
    decode_s: float
    clean: str
    raw: str
    overlay_path: str
    flag: str


def run_page(image_path, server_url, image_mode, max_tokens, page_timeout,
             ngram_window, ngram_size, out_dir, idx):
    raw, tokens, timed_out = "", 0, False
    t0 = time.time()
    gen = sglang_stream(server_url, image_path, image_mode, ngram_window, ngram_size,
                        max_tokens=max_tokens)
    for delta in gen:
        raw += delta
        tokens += 1
        if time.time() - t0 > page_timeout:
            timed_out = True
            gen.close()
            break
    decode_s = time.time() - t0
    capped = (not timed_out) and tokens >= max_tokens
    flag = "timeout" if timed_out else ("capped" if capped else "")

    clean = strip_markup(raw)
    note = {"timeout": f"\n\n[TIMEOUT >{page_timeout}s]",
            "capped": f"\n\n[CAPPED max_tokens={max_tokens}]"}.get(flag)
    if note:
        clean += note
        raw += note

    os.makedirs(out_dir, exist_ok=True)
    base = load_display_image(image_path, max_width=1000)
    overlay = draw_dets(base, parse_dets(raw))
    overlay_path = os.path.join(out_dir, f"overlay_p{idx:04d}.png")
    overlay.save(overlay_path)
    return PageResult(idx, tokens, decode_s, clean, raw, overlay_path, flag)
```

- [ ] **Step 2: Start the SGLang server (if not running)**

Run:
```bash
wsl -d Ubuntu-24.04 bash -lc "curl -sf http://127.0.0.1:10000/health >/dev/null && echo UP || (tr -d '\r' < /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/wsl/03_start_sglang_server.sh | ATTN_BACKEND=triton bash >/tmp/sgsrv.log 2>&1 &) "
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/wsl/wait_health.sh | bash"
```
Expected: `READY after N polls (code 200)`.

- [ ] **Step 3: Smoke-validate runner against live server**

Run:
```bash
wsl -d Ubuntu-24.04 bash -lc 'cd /mnt/c/Users/User/Desktop/project/unlimited-ocr && ~/uocr/.venv-sglang/bin/python -c "
from bench.runner import render_pages, run_page, gpu_mem_mib
imgs = render_pages(\"Unlimited-OCR.pdf\", \"1\")
r = run_page(imgs[0], \"http://127.0.0.1:10000\", \"base\", 4096, 30, 1024, 35, \"bench/reports/_smoke\", 1)
print(\"tokens\", r.tokens, \"flag\", repr(r.flag), \"clean_chars\", len(r.clean), \"overlay\", r.overlay_path)
assert r.tokens > 50 and len(r.clean) > 100
print(\"gpu_mem\", gpu_mem_mib())
print(\"RUNNER_OK\")
" 2>&1 | grep -v "Failed to get device capability"'
```
Expected: prints `RUNNER_OK` with tokens/clean_chars > thresholds and an overlay path.

- [ ] **Step 4: Commit**

```bash
git add bench/runner.py && git commit -m "feat(bench): shared SGLang runner (run_page + render + gpu_mem)"
```

---

### Task 4: Quality track (`quality.py`) + report + hard review bundles

**Files:**
- Create: `bench/quality.py`

**Interfaces:**
- Consumes: `bench.corpus_io`, `bench.runner`, `bench.metrics`.
- Produces: CLI `python -m bench.quality [--manifest PATH] [--server URL] [--out DIR]`; writes `reports/quality_report.md`, `reports/quality.csv`, and per-doc `reports/<id>/`.

- [ ] **Step 1: Implement quality.py**

Create `bench/quality.py`:
```python
"""品質軌：simple 層自動算 pseudo-GT 指標；hard 層產人工審查包。"""
import argparse
import csv
import os

from bench import metrics
from bench.corpus_io import load_manifest, pdf_pseudo_gt
from bench.runner import render_pages, run_page

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HARD_RUBRIC = ["text_correctness", "table_structure", "formula", "figure_boxes",
               "reading_order", "missing_or_repeat"]


def process_doc(entry, corpus_dir, out_root, server_url):
    pdf = os.path.join(corpus_dir, entry.file)
    out_dir = os.path.join(out_root, entry.id)
    imgs = render_pages(pdf, entry.pages)
    results = [run_page(p, server_url, entry.image_mode, 4096, 30, 1024, 35, out_dir, i + 1)
               for i, p in enumerate(imgs)]
    clean = "\n".join(r.clean for r in results)
    raw = "\n".join(r.raw for r in results)
    with open(os.path.join(out_dir, "clean.txt"), "w", encoding="utf-8") as f:
        f.write(clean)
    with open(os.path.join(out_dir, "raw.txt"), "w", encoding="utf-8") as f:
        f.write(raw)

    row = {"id": entry.id, "category": entry.category, "tier": entry.tier,
           "pages": len(results), "flags": ",".join(sorted({r.flag for r in results if r.flag}))}
    if entry.tier == "simple" and entry.gt == "pdf_text":
        gt = pdf_pseudo_gt(pdf, entry.pages)
        row["cer"] = round(metrics.cer(gt, clean), 4)
        row["coverage"] = round(metrics.coverage(gt, clean), 4)
        row["order_sim"] = round(metrics.order_insensitive_sim(gt, clean), 4)
    else:
        _write_review_bundle(entry, results, out_dir)
        row["cer"] = row["coverage"] = row["order_sim"] = ""
    return row


def _write_review_bundle(entry, results, out_dir):
    lines = [f"# 人工審查：{entry.id}（{entry.category}）", "", f"備註：{entry.notes}", ""]
    for r in results:
        lines += [f"## 第 {r.idx} 頁  flag={r.flag or '-'}",
                  f"![overlay](overlay_p{r.idx:04d}.png)", "",
                  "```", r.clean[:4000], "```", ""]
    with open(os.path.join(out_dir, "review.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    if not os.path.exists(os.path.join(out_dir, "review.yaml")):
        ru = [f"{k}: 0   # 0-5" for k in HARD_RUBRIC]
        with open(os.path.join(out_dir, "review.yaml"), "w", encoding="utf-8") as f:
            f.write("\n".join([f"id: {entry.id}"] + ru + ["note: \"\""]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(REPO, "bench/corpus/manifest.yaml"))
    ap.add_argument("--server", default="http://127.0.0.1:10000")
    ap.add_argument("--out", default=os.path.join(REPO, "bench/reports"))
    args = ap.parse_args()

    corpus_dir = os.path.dirname(args.manifest)
    entries = load_manifest(args.manifest)
    rows = [process_doc(e, corpus_dir, args.out, args.server) for e in entries]

    os.makedirs(args.out, exist_ok=True)
    cols = ["id", "category", "tier", "pages", "flags", "cer", "coverage", "order_sim"]
    with open(os.path.join(args.out, "quality.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    md = ["# 品質報告", "", "參考線：simple 層 CER<0.15、coverage>0.9 算好（非硬關卡）。", "",
          "| id | 類別 | 層 | 頁 | flags | CER | coverage | order_sim |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| {id} | {category} | {tier} | {pages} | {flags} | {cer} | {coverage} | {order_sim} |".format(**r))
    hard = [r["id"] for r in rows if r["tier"] != "simple"]
    if hard:
        md += ["", "## 待人工審查（hard）", ""] + [f"- {h} → `reports/{h}/review.md`，填 `review.yaml`" for h in hard]
    with open(os.path.join(args.out, "quality_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"quality: {len(rows)} docs -> {args.out}/quality_report.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Seed a minimal manifest for smoke test**

Create `bench/corpus/manifest.yaml` (academic docs already in repo root; symlink/copy into corpus for the run):
```bash
mkdir -p /mnt/c/Users/User/Desktop/project/unlimited-ocr/bench/corpus/academic
cp /mnt/c/Users/User/Desktop/project/unlimited-ocr/Unlimited-OCR.pdf /mnt/c/Users/User/Desktop/project/unlimited-ocr/bench/corpus/academic/unlimited-ocr.pdf
```
Create `bench/corpus/manifest.yaml`:
```yaml
- id: academic_unlimited_ocr
  file: academic/unlimited-ocr.pdf
  category: academic
  tier: simple
  pages: "1-2"
  image_mode: base
  gt: pdf_text
  notes: "born-digital paper, text-heavy first pages"
```

- [ ] **Step 3: Smoke-run quality (server must be up)**

Run:
```bash
wsl -d Ubuntu-24.04 bash -lc 'cd /mnt/c/Users/User/Desktop/project/unlimited-ocr && ~/uocr/.venv-sglang/bin/python -m bench.quality 2>&1 | grep -v "Failed to get device capability"'
wsl -d Ubuntu-24.04 bash -lc 'cat /mnt/c/Users/User/Desktop/project/unlimited-ocr/bench/reports/quality_report.md'
```
Expected: report lists `academic_unlimited_ocr` with a CER value (likely < 0.3) and coverage; `quality.csv` exists.

- [ ] **Step 4: Commit**

```bash
git add bench/quality.py bench/corpus/manifest.yaml && git commit -m "feat(bench): quality track (pseudo-GT metrics + hard review bundles)"
```

---

### Task 5: Stability track (`stability.py`) — S1–S4 + speed chart

**Files:**
- Create: `bench/stability.py`

**Interfaces:**
- Consumes: `bench.runner`, `ocr_backends.server_healthy`, matplotlib, PyMuPDF.
- Produces: CLI `python -m bench.stability [--server URL] [--out DIR] [--longrun-pages N]`; writes `reports/stability_report.md`, `reports/speed_curve.png`, `reports/speed_timeseries.csv`.

- [ ] **Step 1: Implement stability.py**

Create `bench/stability.py`:
```python
"""穩定軌：S1 長跑+速度圖、S2 迴圈頁、S3 不 crash、S4 回復。"""
import argparse
import csv
import os
import time

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bench.runner import gpu_mem_mib, render_pages, run_page  # noqa: E402
from ocr_backends import server_healthy  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SERVER = "http://127.0.0.1:10000"


def _cycle_pages(pdf_paths, n):
    imgs = []
    for p in pdf_paths:
        imgs += render_pages(p, "all")
    if not imgs:
        return []
    return [imgs[i % len(imgs)] for i in range(n)]


def s1_longrun(server, out, n_pages):
    pdfs = [os.path.join(REPO, f) for f in ("Unlimited-OCR.pdf", "DeepSeek-OCR.pdf")
            if os.path.exists(os.path.join(REPO, f))]
    pages = _cycle_pages(pdfs, n_pages)
    rows, t0 = [], time.time()
    for i, img in enumerate(pages):
        r = run_page(img, server, "base", 4096, 30, 1024, 35, os.path.join(out, "_s1"), i + 1)
        tps = r.tokens / r.decode_s if r.decode_s > 0 else 0
        rows.append((round(time.time() - t0, 1), i + 1, round(tps, 1), round(r.decode_s, 2), gpu_mem_mib()))

    with open(os.path.join(out, "speed_timeseries.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "page_idx", "tok_per_s", "decode_s", "gpu_mem_mib"])
        w.writerows(rows)

    xs = [r[0] for r in rows]
    tps = [r[2] for r in rows]
    mem = [r[4] for r in rows]
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(xs, tps, color="tab:blue", label="tok/s")
    ax1.set_xlabel("elapsed (s)"); ax1.set_ylabel("tok/s", color="tab:blue")
    ax2 = ax1.twinx(); ax2.plot(xs, mem, color="tab:red", label="GPU mem (MiB)")
    ax2.set_ylabel("GPU mem (MiB)", color="tab:red")
    fig.tight_layout(); fig.savefig(os.path.join(out, "speed_curve.png")); plt.close(fig)

    q = max(1, len(tps) // 4)
    first = sorted(tps[:q])[q // 2]
    last = sorted(tps[-q:])[q // 2]
    mem_growth = max(mem) - mem[0] if mem else 0
    ok = (last >= 0.85 * first) and (mem_growth < 1024)
    return {"name": "S1 long-run", "pass": ok,
            "detail": f"pages={len(rows)} first25%TPS={first:.0f} last25%TPS={last:.0f} mem_growth={mem_growth}MiB"}


def s2_loop_guard(server, out):
    pdf = os.path.join(REPO, "DeepSeek-OCR.pdf")
    if not os.path.exists(pdf):
        return {"name": "S2 loop-guard", "pass": None, "detail": "DeepSeek-OCR.pdf not found (skip)"}
    imgs = render_pages(pdf, "all")
    flagged = []
    for i, img in enumerate(imgs):
        r = run_page(img, server, "base", 4096, 20, 1024, 35, os.path.join(out, "_s2"), i + 1)
        if r.flag:
            flagged.append((i + 1, r.flag, r.tokens))
    mem_ok = gpu_mem_mib() < 15500
    ok = len(flagged) > 0 and mem_ok and server_healthy(server)
    return {"name": "S2 loop-guard", "pass": ok,
            "detail": f"flagged_pages={flagged} mem_ok={mem_ok}"}


def s3_no_crash(server, out):
    tmp = os.path.join(out, "_s3")
    os.makedirs(tmp, exist_ok=True)
    from PIL import Image
    cases = []
    Image.new("RGB", (8, 8), "white").save(os.path.join(tmp, "tiny.png")); cases.append("tiny.png")
    Image.new("RGB", (4000, 50), "white").save(os.path.join(tmp, "wide.png")); cases.append("wide.png")
    Image.new("RGB", (1000, 1400), (0, 0, 0)).save(os.path.join(tmp, "black.png")); cases.append("black.png")
    crashes = []
    for name in cases:
        try:
            run_page(os.path.join(tmp, name), server, "base", 1024, 30, 1024, 35, tmp, 1)
        except Exception as e:
            crashes.append(f"{name}: {e}")
    ok = (len(crashes) == 0) and server_healthy(server)
    return {"name": "S3 no-crash", "pass": ok, "detail": f"cases={cases} crashes={crashes}"}


def s4_recovery(server, out):
    pdf = os.path.join(REPO, "DeepSeek-OCR.pdf")
    if not os.path.exists(pdf):
        return {"name": "S4 recovery", "pass": None, "detail": "DeepSeek-OCR.pdf not found (skip)"}
    imgs = render_pages(pdf, "all")
    loop = imgs[7] if len(imgs) > 7 else imgs[0]   # p8 loops
    run_page(loop, server, "base", 4096, 3, 1024, 35, os.path.join(out, "_s4"), 1)  # force abort at 3s
    nxt = run_page(imgs[0], server, "base", 4096, 30, 1024, 35, os.path.join(out, "_s4"), 2)
    ok = nxt.tokens > 50 and server_healthy(server)
    return {"name": "S4 recovery", "pass": ok, "detail": f"next_tokens={nxt.tokens}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=SERVER)
    ap.add_argument("--out", default=os.path.join(REPO, "bench/reports"))
    ap.add_argument("--longrun-pages", type=int, default=200)
    args = ap.parse_args()
    assert server_healthy(args.server), "SGLang server not healthy — start it first"
    os.makedirs(args.out, exist_ok=True)

    results = [
        s1_longrun(args.server, args.out, args.longrun_pages),
        s2_loop_guard(args.server, args.out),
        s3_no_crash(args.server, args.out),
        s4_recovery(args.server, args.out),
    ]
    md = ["# 穩定性報告", "", "| 子測試 | 結果 | 細節 |", "|---|---|---|"]
    for r in results:
        verdict = "PASS" if r["pass"] else ("SKIP" if r["pass"] is None else "FAIL")
        md.append(f"| {r['name']} | {verdict} | {r['detail']} |")
    md += ["", "速度曲線：`speed_curve.png`；時間序列：`speed_timeseries.csv`。"]
    with open(os.path.join(args.out, "stability_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run stability with a short long-run (server up)**

Run:
```bash
wsl -d Ubuntu-24.04 bash -lc 'cd /mnt/c/Users/User/Desktop/project/unlimited-ocr && ~/uocr/.venv-sglang/bin/python -m bench.stability --longrun-pages 20 2>&1 | grep -v "Failed to get device capability" | tail -12'
wsl -d Ubuntu-24.04 bash -lc 'ls -la /mnt/c/Users/User/Desktop/project/unlimited-ocr/bench/reports/speed_curve.png'
```
Expected: prints the stability table (S1 PASS/FAIL, S2/S4 PASS or SKIP, S3 PASS); `speed_curve.png` exists.

- [ ] **Step 3: Commit**

```bash
git add bench/stability.py && git commit -m "feat(bench): stability track S1-S4 + tok/s+mem speed chart"
```

---

### Task 6: Corpus fetcher + manifest expansion + docs

**Files:**
- Create: `bench/fetch_corpus.py`, `bench/README.md`
- Modify: `bench/corpus/manifest.yaml`

**Interfaces:**
- Produces: CLI `python -m bench.fetch_corpus` that downloads publicly-available docs (US patents, etc.) into `bench/corpus/<category>/` and prints which manifest entries still need user-provided files.

- [ ] **Step 1: Implement fetch_corpus.py**

Create `bench/fetch_corpus.py`:
```python
"""下載可公開取得的測試文件到 bench/corpus/。抓不到/需特定用途的標 provided。"""
import os
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS = os.path.join(REPO, "bench", "corpus")

# (category, filename, url) — 公開可下載來源（執行時若失效需更新 URL）
PUBLIC = [
    # 範例：US 專利 PDF（Google Patents 直連 PDF）。執行時確認 URL 仍有效。
    # ("patents", "us_patent_sample.pdf", "https://patentimages.storage.googleapis.com/.../US....pdf"),
]


def main():
    for cat, name, url in PUBLIC:
        d = os.path.join(CORPUS, cat)
        os.makedirs(d, exist_ok=True)
        dest = os.path.join(d, name)
        if os.path.exists(dest):
            print(f"skip (exists): {cat}/{name}")
            continue
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"fetched: {cat}/{name}")
        except Exception as e:
            print(f"FAILED {cat}/{name}: {e} -> 需改 URL 或改由使用者提供")
    print("提醒：cjk / tables / scanned / 部分 patents 若抓不到，請放到對應 corpus/<類別>/ 並補 manifest。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write bench/README.md (how to run)**

Create `bench/README.md`:
```markdown
# Unlimited-OCR 測試 harness

依 `docs/superpowers/specs/2026-06-27-ocr-testing-plan-design.md`。

## 前置
- 啟動 SGLang server：`scripts/wsl/03_start_sglang_server.sh`（`ATTN_BACKEND=triton`）。
- 相依已在 `.venv-sglang`：rapidfuzz / pyyaml / matplotlib / pytest。

## 語料
- 把文件放 `bench/corpus/<academic|cjk|tables|scanned|patents>/`，在 `bench/corpus/manifest.yaml` 登記。
- 公開可抓的：`python -m bench.fetch_corpus`。其餘由使用者提供。
- 難度分層：有乾淨文字層、文字為主 → `tier: simple`（自動 pseudo-GT）；表格/公式/圖/掃描 → `tier: hard`（人工）。

## 跑測試
```bash
# 單元測試
~/uocr/.venv-sglang/bin/python -m pytest bench/tests -q
# 品質軌
~/uocr/.venv-sglang/bin/python -m bench.quality
# 穩定軌（長跑頁數可調）
~/uocr/.venv-sglang/bin/python -m bench.stability --longrun-pages 200
```

## 產出（`bench/reports/`，gitignore）
- `quality_report.md` / `quality.csv`：simple 指標；hard 待人工清單。
- 各 `<id>/`：clean/raw/overlay；hard 另有 `review.md` + `review.yaml`（填 0–5）。
- `stability_report.md`、`speed_curve.png`、`speed_timeseries.csv`。
```

- [ ] **Step 3: Run unit tests + commit**

Run:
```bash
wsl -d Ubuntu-24.04 bash -lc 'cd /mnt/c/Users/User/Desktop/project/unlimited-ocr && ~/uocr/.venv-sglang/bin/python -m pytest bench/tests -q'
```
Expected: all pass.

```bash
git add bench/fetch_corpus.py bench/README.md && git commit -m "feat(bench): corpus fetcher + bench README"
```

---

## Self-Review

**Spec coverage:**
- §3 structure → Tasks 1–6 create `bench/` with all modules. ✓
- §4 manifest → Task 2 (`corpus_io.load_manifest`) + Task 4 seed. ✓
- §5 quality track → Task 4. ✓
- §6 stability S1–S4 + speed chart → Task 5. ✓
- §7 metrics (CER/coverage/order-insensitive + rubric) → Task 1 (metrics) + Task 4 (`review.yaml` rubric). ✓
- §8 success criteria → encoded in Task 5 (S1 quartile check, mem<1GB) + quality report reference line. ✓
- §9 env/deps → Global Constraints + Task 1 install. ✓
- Corpus fetch/expansion → Task 6. ✓

**Placeholder scan:** No "TBD/handle edge cases" — all code is concrete. `fetch_corpus.PUBLIC` is intentionally an empty seed list with a documented format (URLs are resolved at execution per spec §2 "能公開抓的由實作端抓"); this is data, not a code placeholder.

**Type consistency:** `PageResult.flag` ∈ {"", "timeout", "capped"} used consistently in runner/quality/stability; `run_page(...)` signature identical across call sites; `DocEntry` fields match manifest format and `process_doc` usage. ✓

**Note (deviation from pure TDD):** Tasks 1–2 are classic test-first (pure functions). Tasks 3–5 are integration against a live GPU server; their "test" is a smoke-validation step (run + assert outputs), which is the honest verification for server/GPU-bound code.
