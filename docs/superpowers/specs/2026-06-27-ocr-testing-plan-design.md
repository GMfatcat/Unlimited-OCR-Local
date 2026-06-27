# Unlimited-OCR 測試規劃 — 設計文件 (Test Plan Design)

- 日期: 2026-06-27
- 範圍: 對本地可執行的 Unlimited-OCR(SGLang 路線為主)做**品質**與**穩定性**的系統化測試。
- 方案: **A — 雙軌 harness + 策展語料**(品質軌 / 穩定軌 + 策展 corpus + 報告)。
- 相關: 本地環境見 `docs/superpowers/specs/2026-06-25-unlimited-ocr-env-design.md`;踩坑見 `docs/DEVELOPMENT-NOTES.md`。

---

## 1. 目標與非目標

**目標**
1. **OCR 品質/正確性**:依難度分層評估辨識與版面結果是否可信。
2. **穩定性/強健度**:不 crash、迴圈頁防護正確、長跑不退化/不漏記憶體、錯誤可回復。

**非目標(本次不做)**
- 效能/併發吞吐 benchmark(TPS/延遲掃描)——僅在穩定軌順帶記錄速度曲線,不做併發壓測。
- 回歸自動化 / pytest pass-fail 套件 / CI。
- 公開 benchmark 跑分:**刻意不採用**(疑似訓練汙染,數字不可信)。

## 2. 核心決策

- **品質採「難度分層」**:
  - `simple`(有乾淨文字層、以文字為主)→ 用 **PyMuPDF 抽文字層當 pseudo-GT**,自動算指標。
  - `hard`(表格/公式/圖/掃描/多欄複雜順序)→ **人工**依 rubric 評分(自動指標對這類不可信)。
- **語料 5 類**:`academic`(repo 已有 3 份)、`cjk`、`tables`、`scanned`、`patents`(US 專利,公開、含圖表)。
- **語料分工**:能公開抓的由實作端抓(US 專利等),難找/特定用途的由使用者補。
- **路線**:SGLang(部署目標;本機 triton)。Transformers 僅在需要時抽查兩路線一致性。
- **品質指標為參考、非硬性 pass/fail**:OCR 品質本質模糊,報告呈現數字 + 門檻參考線。

## 3. 目錄結構

```
bench/
  corpus/                      # 測試文件（PDF，大檔 gitignore）
    academic/ cjk/ tables/ scanned/ patents/
    manifest.yaml              # committed：每份文件元資料
  fetch_corpus.py              # 下載可公開取得的文件（US 專利等）到 corpus/
  quality.py                   # 品質軌 runner
  stability.py                 # 穩定軌 runner
  metrics.py                   # CER / 覆蓋率 / 順序不敏感相似度（含自我單元測試）
  reports/                     # 產出（gitignore）：報告 + 速度圖 + 每份審查包
```

**commit 策略**:`manifest.yaml` 與 harness 程式碼進 git;`corpus/*.pdf` 與 `reports/` 走 `.dockerignore`/`.gitignore`(避免大檔/版權,US 專利雖公開仍不烤進 repo)。

## 4. 語料 manifest 格式

`bench/corpus/manifest.yaml`,每筆:
```yaml
- id: patent_us_xxxx
  file: patents/us_xxxx.pdf
  category: patents          # academic | cjk | tables | scanned | patents
  tier: hard                 # simple | hard
  source: "https://patents.google.com/..."   # 或 provided
  pages: "1-5"               # 或 all
  image_mode: base           # PDF 預設 base；單張難圖可 gundam
  gt: pdf_text               # simple→pdf_text(PyMuPDF) | hard→manual
  notes: "含圖表，閱讀順序複雜"
```

**分層原則**:有乾淨文字層、以文字為主 → `simple`;含表格/公式/圖/掃描/多欄複雜順序 → `hard`。
**規模**:每類 **3–5 份**(先小而精)。

## 5. 品質軌(`quality.py`)

每份文件流程:
1. PyMuPDF 轉指定頁為 PNG(沿用 `ocr_backends.to_image_paths`)。
2. 丟 SGLang server(沿用 `ocr_backends.sglang_stream`),帶 manifest 的 `image_mode`、`max_tokens=4096`、ngram 預設。
3. 存產出到 `reports/<id>/`:`clean.txt`、`raw.txt`、每頁 `overlay_pXX.png`。
4. 依 `tier`:
   - `simple` → 抽 pseudo-GT(PyMuPDF `page.get_text`)→ 算指標(§7)。
   - `hard` → 產審查包 `review.md`(嵌 overlay 圖 + clean 文字 + 評分清單)、空白 `review.yaml` 待人工填。
5. 彙整 `quality_report.md` + `quality.csv`(simple 指標;hard 待人工清單),依類別/層分組。

## 6. 穩定軌(`stability.py`)

| 子測試 | 內容 | 判定 |
|---|---|---|
| **S1 長跑不退化** | 連續循序送 **目標頁數**（預設 ~200 頁，以循環 corpus 頁面湊足）；逐頁記錄 tok/s、decode 時間、GPU mem、時間戳 | **末 25% 頁的中位 TPS ≥ 首 25% 的 85%** 且全程 GPU mem 增幅 < 1GB |
| **S2 迴圈頁防護** | 已知鬼打牆頁（DeepSeek-OCR p8/p17）+ 掃 corpus | max_tokens 乾淨截斷 / 逾時中止；標註出現；中止後 mem 穩定、server 健康 |
| **S3 不會 crash** | 對抗輸入：空白頁、極小/超大圖、旋轉、毀損/空 PDF、純色圖、反向座標框頁 | server 全程 /health 200；harness 不被例外中斷；`draw_dets` 壞框安全跳過 |
| **S4 錯誤回復** | 串流中途 abort（觸發逾時）→ 立刻送下一份 → 成功；進階：pkill server → 重啟 → 可續 | 中止後下一份完成；重啟後可繼續服務 |

產出 `stability_report.md`(四項 PASS/FAIL + 關鍵數據)。

### 速度線圖(S1 順帶)
- `reports/speed_curve.png`:**x = 經過時間（或累積頁序），y = tok/s**,畫整段速度變化;次座標軸疊 **GPU mem(MiB)** 看「降速 vs 記憶體吃滿」相關性。
- `reports/speed_timeseries.csv`:`time, page_idx, tok_per_s, decode_s, gpu_mem_mib`。
- 以 matplotlib 繪製。

## 7. 指標(`metrics.py`,僅 simple 層)

- `normalize(t)`:去標記(沿用 `ocr_backends.strip_markup`)、NFKC、收斂空白、去連字斷行、可選小寫。
- **CER**:字元級編輯距離 / len(GT)。主指標(以 rapidfuzz 計算)。
- **順序不敏感相似度**:行/詞排序後比對;避免 pseudo-GT 閱讀順序差異灌爆 CER。
- **覆蓋率**:GT 詞有多少出現在輸出;抓漏字/漏段。
- 三者並陳:CER 看整體、順序不敏感看「字對序差」、覆蓋率看「有沒有漏」。
- `metrics.py` 內含對固定字串的自我單元測試(已知輸入→已知值)。

### hard 層人工 rubric(`review.yaml`,每項 0–5）
文字正確性、表格結構、公式、圖/圖說框、閱讀順序、漏字/重複;各項 0–5 + 一句備註。

## 8. 通過標準(彙整)

| 軌 | 指標 | 參考線 |
|---|---|---|
| 品質 simple | CER / 覆蓋率 | CER < 0.15、覆蓋率 > 0.9 算好（**參考、非硬關卡**） |
| 品質 hard | 人工 rubric | 各項 ≥ 4/5 |
| 穩定 S1 | TPS / mem | 末 25% 頁中位 TPS ≥ 首 25% 的 85%、mem 增幅 < 1GB |
| 穩定 S2 | 迴圈頁 | 100% 正確截斷 + 標註 |
| 穩定 S3 | crash | = 0 |
| 穩定 S4 | 回復 | 中止/重啟後可續 |

## 9. 執行環境與相依

- 跑在**本機(WSL2,RTX 5070 Ti / sm_120,SGLang `triton`)**,需先啟動 SGLang server;結果代表「模型品質」,對 H100 部署具代表性(triton vs fa3 對輸出影響可忽略)。
- bench 在 `.venv-sglang` 執行,需補裝:**matplotlib**、**pyyaml**、**rapidfuzz**。
- 沿用 `ocr_backends` 的 `sglang_stream` / `strip_markup` / `to_image_paths` / `parse_dets` / `draw_dets`。

## 10. 風險與緩解

| 風險 | 緩解 |
|---|---|
| pseudo-GT 閱讀順序/表格不準 | 只用於 simple 層(排除表格/多欄);並報順序不敏感相似度 |
| 公開語料抓取失敗/版權 | 抓不到的由使用者補;corpus 不進 repo |
| 本機 sm_120 ≠ H100(fa3) | 品質取決於模型權重,差異可忽略;穩定軌的絕對 TPS 數字僅供本機相對比較 |
| 長跑退化（已知）反而是「待測現象」 | S1 專門量它;若觸發,佐證 max_tokens 防線的必要性 |

## 11. 階段(實作時)

1. 補 bench 相依(matplotlib/pyyaml/rapidfuzz)+ 建 `bench/` 骨架與 `metrics.py`(含自我測試)。
2. 策展 corpus:抓公開文件(US 專利等）+ 寫 manifest;缺的標 `provided` 待補。
3. 實作 `quality.py`(simple 自動 + hard 審查包),對現有 academic 先跑通。
4. 實作 `stability.py`(S1–S4 + 速度圖)。
5. 全量跑 → 產報告 → 人工審 hard 層 → 彙整結論。
