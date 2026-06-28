<p align="center">
  <img src="assets/baidu.png" width="40%" alt="Baidu Inc." />
</p>

<h1 align="center">Unlimited-OCR · Local Run + Real-time OCR UI</h1>

<p align="center">
  <a href="./README.md">中文</a> ｜ <b>English</b>
</p>

<p align="center">
  Run both the <b>Transformers</b> and <b>SGLang</b> inference routes on Windows + WSL2 with <b>uv</b>, plus a <b>Streamlit real-time OCR interface</b>.
  <br>The upstream official documentation is in <a href="./README-Baidu.md">README-Baidu.md</a>.
</p>

<p align="center">
  <img src="assets/ui-demo.gif" width="90%" alt="Streamlit real-time OCR demo" />
  <br><sub>Streamlit UI real-time OCR: layout boxes grow on the left column, plain text scrolls down on the right as it is scanned</sub>
</p>

---

## 📌 What is this project

Building on the [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) weights, this provides a **locally reproducible** environment and toolset:

- 🤗 **Transformers route**: run OCR directly with `transformers` (single image / multi-page / PDF).
- ⚡ **SGLang route**: start an OpenAI-compatible server from a local SGLang wheel, with **per-token streaming**.
- 🖥️ **Streamlit UI**: upload an image / PDF and **see it as it scans** — layout-detection boxes grow live on the left column, plain text scrolls down on the right, with speed metrics, timeout / cap protection, page navigation, and ZIP export.

> Full environment decisions and measured data: [`docs/superpowers/specs/2026-06-25-unlimited-ocr-env-design.md`](docs/superpowers/specs/2026-06-25-unlimited-ocr-env-design.md);
> a one-page command cheat sheet: [`RUN.md`](RUN.md);
> **development gotchas** (Blackwell/SGLang/UI pitfalls and fixes): [`docs/DEVELOPMENT-NOTES.md`](docs/DEVELOPMENT-NOTES.md).

## 🧩 Environment & architecture

| Item | Details |
|---|---|
| Platform | **WSL2 (Ubuntu-24.04)**. SGLang has no native Windows support; WSL2 GPU passthrough is verified |
| GPU | NVIDIA RTX 5070 Ti (**Blackwell sm_120**) / 16GB |
| Package manager | **uv**, two separate venvs (dependencies conflict, so they must be split) |
| `~/uocr/.venv-transformers` | torch 2.10 (cu129) / transformers 4.57.1 |
| `~/uocr/.venv-sglang` | local sglang wheel / torch 2.9.1 (cu128) / transformers 5.3.0 / streamlit |
| Weights | `unlimited-ocr-hf/` (read directly from WSL via `/mnt/c/...`, not copied) |

> **Blackwell note**: SGLang's `fa3` and `flashinfer` attention backends do not work on sm_120; **`triton`** is used in practice.
> It also needs system-level `libnuma / build-essential / python3-dev / CUDA Toolkit 12.8` (runtime JIT requires nvcc ≥ 12.8).
> These are all in `scripts/wsl/04_system_prereqs.sh`.

## 🚀 Installation (one-time)

Run in a Windows terminal (it calls into WSL). `R = /mnt/c/Users/User/Desktop/project/unlimited-ocr`:

```powershell
# 0) Install uv, verify the weights are readable
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < $R/scripts/wsl/00_setup_uv.sh | bash"
# 1) Transformers venv
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < $R/scripts/wsl/01_setup_transformers.sh | bash"
# 2) SGLang venv
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < $R/scripts/wsl/02_setup_sglang.sh | bash"
# 3) System prerequisites (needs root): libnuma / gcc / python headers / CUDA Toolkit 12.8
wsl -d Ubuntu-24.04 -u root bash -lc "tr -d '\r' < $R/scripts/wsl/04_system_prereqs.sh | bash"
# 4) streamlit for the UI (installed into the sglang venv)
wsl -d Ubuntu-24.04 bash -lc "~/.local/bin/uv pip install --python ~/uocr/.venv-sglang/bin/python streamlit"
```

> Above, `$R` is shown as a variable for brevity; use the full path `/mnt/c/Users/User/Desktop/project/unlimited-ocr` in practice.

## ✅ Test both routes

```powershell
# Transformers (single-page gundam + multi-page base)
wsl -d Ubuntu-24.04 bash -lc "~/uocr/.venv-transformers/bin/python /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/test_transformers.py"

# SGLang: start the server first (attention-backend fixed to triton), then run a PDF with infer.py
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/wsl/03_start_sglang_server.sh | ATTN_BACKEND=triton bash"
wsl -d Ubuntu-24.04 bash -lc "cd /mnt/c/Users/User/Desktop/project/unlimited-ocr && ~/uocr/.venv-sglang/bin/python infer.py --pdf ./Unlimited-OCR.pdf --output_dir ./outputs/sglang --image_mode base"
```

Measured (RTX 5070 Ti, the repo's `Unlimited-OCR.pdf`): Transformers single-page gundam ≈ 13.6s; SGLang streaming on normal pages ≈ **200–250 tok/s**.

## 🖥️ Launch the Streamlit UI

> ⚠️ **Only one of the SGLang server and Transformers can run at a time** (16GB VRAM cannot keep both resident).
> The UI's "SGLang streaming" needs the server; for "Transformers batch", stop the server first.

```powershell
# Terminal A: SGLang server
wsl -d Ubuntu-24.04 bash -lc "tr -d '\r' < /mnt/c/Users/User/Desktop/project/unlimited-ocr/scripts/wsl/03_start_sglang_server.sh | ATTN_BACKEND=triton bash"

# Terminal B: UI
wsl -d Ubuntu-24.04 bash -lc "cd /mnt/c/Users/User/Desktop/project/unlimited-ocr && ~/uocr/.venv-sglang/bin/streamlit run app.py"
```

Open **[http://localhost:8501](http://localhost:8501)** in a browser.

> 🎥 **Record the real-time OCR screen**: once the server + UI are up, record automatically with Playwright (separate venv, chromium only):
> `wsl ... ~/uocr/.venv-playwright/bin/python scripts/record_demo.py [pdf] [out_dir]`
> Defaults to `assets/demo_3pages.pdf`, output `outputs/demo/*.webm` (the script prints ffmpeg commands to convert to mp4/gif).

### UI features
- 🟦 **Live detection boxes**: every parsed `<|det|>` box is drawn onto the left-column image (`title` in a thick red box, others colored by category).
- 📝 **Plain text + auto-scroll**: the right column shows only the markup-stripped plain text; a terminal-style tail always stays at the newest line and follows the scan.
- 📊 **Live metrics**: page count / elapsed time / total tokens / average speed (throttled updates during streaming, finalized per page).
- 🧱 **Generation cap `max_tokens` (default 4096)**: some pages "loop" and repeat forever; this cap makes them **end cleanly and early**, preventing abnormal token counts from filling the GPU and dragging down later pages.
- ⏳ **Per-page timeout (default 30s)**: a fallback line of defense — on timeout it aborts and moves to the next page.
- ⚠️ Timed-out / capped pages are marked in the **status bar, end of the plain text, metrics row, navigation marker, and inside the ZIP**, so "the result may be incomplete" is easy to spot.
- 🧭 **Page navigation**: a slider when there are < 16 pages; an input box + Go / Prev / Next when ≥ 16.
- ⬇️ **Download ZIP**: filename `unlimited_ocr_{original_name}_{timestamp_to_seconds}.zip`; one folder per page with `overlay.png` (boxed image), `raw.txt` (raw output), `text.txt` (plain text).
- 📖 **Help**: a sidebar button opens a wide help dialog.

### 🎨 Detection-box color reference

Colored by the `<|det|>` category (label), defined in `LABEL_COLORS` in `ocr_backends.py`. The `title` box is thicker (line width 4), others thinner (line width 2).

| Category (label) | Color | Hex | RGB | Approx. |
|---|---|---|---|---|
| `title` (title, **thick box**) | red | `#DC2828` | (220, 40, 40) | 🟥 |
| `header` (header) | orange | `#E68C14` | (230, 140, 20) | 🟧 |
| `text` (body) | blue | `#286EDC` | (40, 110, 220) | 🟦 |
| `image` / `figure` (figure) | green | `#1EAA5A` | (30, 170, 90) | 🟩 |
| `image_caption` (figure caption) | cyan | `#14A0A0` | (20, 160, 160) | 🟦 |
| `table` (table) | purple | `#963CC8` | (150, 60, 200) | 🟪 |
| `table_caption` (table caption) | light purple | `#7850C8` | (120, 80, 200) | 🟪 |
| `list` (list) | brown | `#A06428` | (160, 100, 40) | 🟫 |
| `formula` (formula) | magenta | `#C83CA0` | (200, 60, 160) | 🟪 |
| `page_number` / `footer` (page number/footer) | gray | `#828282` | (130, 130, 130) | ⬜ |
| any other label (default) | magenta | `#C83CA0` | (200, 60, 160) | 🟪 |

> The "Approx." column is a rough emoji hint; the Hex/RGB values are authoritative. The category label is also drawn on the box.

## 🛡️ Safeguards (avoiding infinite repeat loops)

On some pages the model repeats generation up to `max_length` (32768), stalling for minutes and even filling GPU memory, which **slows down the whole server**. This project uses two lines of defense:

1. **Primary: the `max_tokens` cap** (server side). Problem pages **end cleanly and release memory** at the cap; normal pages (~1–3k tokens) are unaffected. Measured: loop pages stop at 4096 tokens, GPU memory stable throughout.
2. **Fallback: per-page timeout** (client side). Even if the cap isn't reached, a timeout aborts and closes the connection (SGLang then aborts the request and frees memory) and moves to the next page.

> To fully reset after long, heavy use: `wsl -d Ubuntu-24.04 bash -lc "pkill -f sglang.launch_server"`, then re-run `03_start_sglang_server.sh`.

## 📊 Quality & stability test report

Full test report (**self-contained HTML** with model-mechanism explanation, result overlay images, measured data, and a glossary — open it directly in a browser):
**[`docs/Unlimited-OCR-test-report.html`](docs/Unlimited-OCR-test-report.html)**
The test code is in [`bench/`](bench/); the plan is in [`docs/superpowers/specs/2026-06-27-ocr-testing-plan-design.md`](docs/superpowers/specs/2026-06-27-ocr-testing-plan-design.md).
> We deliberately **avoid public benchmarks** (possible training contamination); instead we use a self-built corpus: easy items scored automatically against the PDF text layer, hard items scored manually.

**Test corpus** (7 categories, 9 documents; full source URLs in [`bench/corpus/manifest.yaml`](bench/corpus/manifest.yaml), the PDFs themselves are not committed):

| Category | Document |
|---|---|
| Academic paper | the Unlimited-OCR paper |
| Chinese / CJK | HK CSB *Government Document Writing Manual*, Taiwan DGBAS National Statistics Bulletin 040 |
| Dense tables | US Treasury FSOC 2025 Annual Report, US Patent 8110241B2 |
| Scanned | US Patent 3930271A (1976 typewritten scan) |
| Patent (many figures) | US Patent 6556710B2 |
| Financial report | NVIDIA 10-K |
| Contract / handwriting | stamped contract + handwriting (user-provided) |

**Key results:**

- 🎯 **Excellent on clean print / tables**: English body-text character error rate (CER) ≈ **1.3%**, Chinese official documents ≈ 14% (and mostly whitespace / special-symbol noise in the reference, not real errors).
- 📋 **Hard items (manual 0–5)**: table structure, formulas, and reading order are mostly perfect; dense tables in financial annual reports and the numbers in 10-K filings are read almost flawlessly.
- ⚠️ **Known weaknesses**: stamps / low-quality scans interfere with text; dense table pages occasionally over-generate (triggering `max_tokens`); blank areas are sometimes labeled as text; some figure boxes are incomplete.
- 🛡️ **Stability S1–S4 all pass**: 200-page long run with no speed degradation (251→249 tok/s), no GPU memory leak; loop pages are cleanly cut off by the generation cap; 7 adversarial inputs don't crash; recovers after interruption.
- 🔁 **Multi-page mode, measured**: page-by-page base is **strictly better** than multi-page-in-one (multi-page is slower, and with more pages it drops/repeats content — at 20 pages it stalls on the table of contents and covers only 63%) → **page-by-page deployment** (see [`docs/multipage-test-plan.md`](docs/multipage-test-plan.md)).

> `mem-fraction-static` defaults are now **0.5 (~10GB locally) / 0.18 (~14GB on H100)** — a single-page base service needs no large KV pool; override with `MEM_FRACTION`.

## 🐳 Docker deployment (H100)

The next stage is containerized deployment to **H100 (amd64 / Hopper sm_90)** as a **single container (SGLang server + Streamlit UI)**.

- Dockerfile / entrypoint / instructions: [`docker/`](docker/) (multi-stage slim image: `docker/Dockerfile.h100`, `docker/entrypoint.sh`, `docker/README.md`).
- Overall plan, dependency constraints, and risks: [`docs/DOCKER-DEPLOYMENT-PROPOSAL.md`](docs/DOCKER-DEPLOYMENT-PROPOSAL.md).

Highlights:
- H100 = sm_90 → attention-backend `fa3` (no Blackwell triton-JIT needed).
- Host driver ~R550 (CUDA 12.4), the custom wheel is cu128 → use **CUDA forward-compat** (compat libs baked into the image, zero host-side downloads; controlled by `USE_CUDA_COMPAT`).
- Target host is **air-gapped**: build on a networked machine `docker build` → `docker save` → copy → `docker load` (no registry).
- Verified: image builds, in-container imports work, **transfer tar ~8GB** (multi-stage slim: image 24.1GB). **Not yet verified (needs H100)**: `fa3` and forward-compat behavior on real hardware.
- **DGX Spark (arm64 / sm_121) on hold**: focus on H100; if H100 succeeds that's enough, Spark revisited later.

## 🗂️ Project structure

```
app.py                       # Streamlit UI (interface only)
ocr_backends.py              # backend logic: streaming / subprocess / det parsing / box drawing / ZIP packaging
infer.py                     # upstream: SGLang concurrent batch inference
scripts/
  test_transformers.py       # Transformers route smoke test
  test_ui_stream.py          # UI SGLang streaming test
  test_ui_transformers.py    # UI Transformers batch test
  test_overlay.py            # offline det parsing / plain text / box drawing test
  test_timeout.py            # timeout abort + server recovery test
  test_deepseek.py           # multi-page PDF test for timeout / loop pages
  ocr_once.py                # single-image Transformers OCR (called by the UI subprocess)
  record_demo.py             # Playwright auto-recorder of the UI real-time OCR screen → .webm
  wsl/                       # WSL install / launch scripts (00–04 + wait_health)
bench/                       # test harness: quality track / stability track / multi-page test / report generation
docker/                      # H100 container: Dockerfile.h100 / entrypoint.sh / README
docs/
  Unlimited-OCR-test-report.html # full test report (self-contained HTML)
  multipage-test-plan.md     # multi-page vs page-by-page test
  DEVELOPMENT-NOTES.md       # development gotchas
  DOCKER-DEPLOYMENT-PROPOSAL.md  # Docker deployment plan (H100 / DGX Spark)
  superpowers/specs/         # environment decision docs
unlimited-ocr-hf/            # model weights (gitignored)
README.md                    # Chinese version
README-Baidu.md              # upstream official documentation (English)
LICENSE                      # MIT (original work © GMfatcat; upstream components © Baidu)
```

## 🙏 Acknowledgement / Citation

The model and method come from Baidu Unlimited-OCR; thanks also to DeepSeek-OCR and PaddleOCR. Citation info and the original documentation are in [README-Baidu.md](./README-Baidu.md).

## 📜 License

This project is licensed under **MIT** (see [`LICENSE`](./LICENSE)).
- The **original work** in this repo (local environment/scripts, the Streamlit UI, the test harness `bench/`, Docker, docs) is © 2026 GMfatcat, MIT.
- **Upstream components** (`infer.py`, `README-Baidu.md`, images under `assets/`, etc.) derive from [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR), also MIT, © 2026 Baidu.
- The **model weights** are distributed separately under their own license and are not covered by this repo.
