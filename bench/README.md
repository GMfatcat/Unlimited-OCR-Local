# Unlimited-OCR 測試 harness

依 `docs/superpowers/specs/2026-06-27-ocr-testing-plan-design.md`。

## 前置

- 啟動 SGLang server：`scripts/wsl/03_start_sglang_server.sh`（`ATTN_BACKEND=triton`）。
- 相依已在 `.venv-sglang`：rapidfuzz / pyyaml / matplotlib / pytest。

## 語料

把文件放 `bench/corpus/<academic|cjk|tables|scanned|patents>/`，在 `bench/corpus/manifest.yaml` 登記。

### 公開可下載的文件

```bash
# 在 WSL 中執行（專案 Python）
~/uocr/.venv-sglang/bin/python -m bench.fetch_corpus
```

目前抓取的公開來源（詳見 `bench/fetch_corpus.py`）：

| 類別 | 文件 | tier |
|------|------|------|
| cjk | HK CSB 政府公文寫作手冊 | simple |
| cjk | 台灣主計總處 國情統計通報 040 | hard |
| tables | US Treasury FSOC 2025 年報 | hard |
| tables | 專利 US8110241B2（表格＋示意圖） | hard |
| scanned | Internet Archive 掃描報告 | hard |
| scanned | 專利 US3930271A（1976 掃描打字稿） | hard |
| patents | 專利 US6556710B2（多欄多圖） | hard |

### 需使用者自備（無公開穩定直連）

以下類別需手動提供；放到對應目錄後在 `manifest.yaml` 取消注解並填入欄位：

- **scanned-real**：手機翻拍/嚴重歪斜的真實辦公/公文掃描（最硬的測項）
- **cjk-presentation**：真正中文公文簡報/投影片版面（多欄+圖框）
- **tables-10K**：上市公司 10-K/年報最密集表格頁

### 難度分層規則

| tier | 條件 | GT 方式 |
|------|------|---------|
| `simple` | born-digital、有乾淨文字層、文字為主 | 自動 pseudo-GT（`pdf_text`） |
| `hard` | 含表格/公式/圖/掃描/多欄版面/無文字層 | 人工標注（`manual`） |

`bench/corpus/**/*.pdf` 已加入 `.gitignore`，不會被 commit。

## 跑測試

```bash
# 單元測試
~/uocr/.venv-sglang/bin/python -m pytest bench/tests -q

# 品質軌（需 SGLang server 啟動）
~/uocr/.venv-sglang/bin/python -m bench.quality

# 穩定軌（長跑頁數可調，需 SGLang server 啟動）
~/uocr/.venv-sglang/bin/python -m bench.stability --longrun-pages 200
```

## 產出（`bench/reports/`，gitignore）

- `quality_report.md` / `quality.csv`：simple 文件指標（CER / coverage / order score）；hard 文件輸出待人工審查清單。
- `<id>/`：每份文件的 `clean.txt`、`raw.txt`、`overlay_p*.png`；hard 文件另有 `review.md` + `review.yaml`（填 0–5 評分）。
- `stability_report.md`、`speed_curve.png`、`speed_timeseries.csv`：穩定性與速度指標。

## 目錄結構

```
bench/
├── fetch_corpus.py      # 公開文件下載器
├── corpus_io.py         # manifest 載入 + pseudo-GT 抽取
├── metrics.py           # CER / coverage / order 計算
├── runner.py            # PDF→PNG→OCR 執行器
├── quality.py           # 品質軌主程式
├── stability.py         # 穩定軌主程式
├── corpus/
│   ├── manifest.yaml    # 語料登記表
│   ├── academic/
│   ├── cjk/
│   ├── tables/
│   ├── scanned/
│   └── patents/
├── reports/             # 產出（gitignore）
└── tests/               # 單元測試
```
