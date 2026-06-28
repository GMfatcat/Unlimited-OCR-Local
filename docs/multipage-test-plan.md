# Multipage 模式測試 — 最小驗證（測試 1）

> 續作起點（使用者將 /compress 後從這裡開始）。日期 2026-06-28。

## 背景 / 現況
- OCR 測試 harness（`bench/`）已完成並合併到 main：品質軌 + 穩定軌 + 報告。
- 測試報告 `docs/Unlimited-OCR-test-report.html` 已完成。其中「多頁模式 vs 逐頁部署」一段目前是**基於程式碼推論**（對照 `unlimited-ocr-hf/modeling_unlimitedocr.py` 的 `infer_multi`）。
- 現有部署/測試都是**逐頁**（每頁一個 SGLang 請求、`image_mode=base`）。
- 結論待驗證：多頁「一次推論」的**每頁保真是否與逐頁 base 相同**，以及**速度/吞吐、VRAM** 差異。

## 目標（測試 1：最小驗證）
拿一份**乾淨多頁 PDF**（建議學術論文內文，如 `Unlimited-OCR.pdf` 第 3–6 頁，born-digital、可抽文字層當基準），用兩種方式各跑一次並比較：

| 比較項 | 逐頁 base（現況） | 多頁一次 base |
|---|---|---|
| 輸出文字 | 每頁分別串流、串接 | 一個請求一次產生整份 |
| **文字一致性** | — | 兩者文字是否一致/差多少（用 `bench.metrics` 比） |
| **總耗時 / 吞吐** | 各頁耗時加總 | 單請求總耗時 |
| **峰值 VRAM** | `nvidia-smi` | `nvidia-smi` |

驗收問題：多頁模式是否「輸出和逐頁相當、但更快（省去重複 prefill/請求往返）」？峰值 VRAM 是否仍在 16GB 內？

## 關鍵技術點（動工前先確認）
1. **SGLang 怎麼下多頁請求**：在**單一請求**的 `messages[].content` 放**多個 `image_url`**（每頁一張），帶 `images_config.image_mode="base"`。需先用 1 個小測試確認這顆客製 wheel 的多圖輸入行為與 `infer_multi` 一致（每頁各自編碼成 ~256 視覺 tokens、串接當參考）。
   - 參考現有單圖呼叫：`ocr_backends.sglang_stream`（`server_url`/`image_mode`/`max_tokens`/ngram）。多頁版需把 content 改成多張圖 + 對應的 prompt（如 `"Multi page parsing."`）。
2. **長度 / VRAM**：多頁 = 視覺 tokens 疊加（N×~256）+ 長輸出，需在 32K context 內。頁數越多越吃 KV/記憶體；消費卡 16GB 要留意，必要時降頁數或 `mem-fraction`。
3. **harness 擴充**：目前 `bench/` 是逐頁設計。測試 1 可寫一支獨立小腳本（如 `bench/test_multipage.py`），不必動既有 harness：
   - 逐頁：對選定頁 render → 逐頁 `sglang_stream` → 串接 clean、記錄總時間、峰值 VRAM。
   - 多頁：同樣的頁 → 一個多圖請求 → clean、總時間、峰值 VRAM。
   - 比較：`metrics.cer` / `coverage` / `order_insensitive_sim`（逐頁 vs 多頁），印出耗時與 VRAM。

## 環境提醒
- 在 WSL 跑：`~/uocr/.venv-sglang/bin/python`；先啟動 SGLang server（`scripts/wsl/03_start_sglang_server.sh`，`ATTN_BACKEND=triton`），用 `scripts/wsl/wait_health.sh` 等就緒。
- stderr 會狂噴 `Failed to get device capability: SM 12.x requires CUDA >= 12.9` → 無害，`| grep -v` 過濾。
- 報告產生器在 `bench/_gen_report.py`（untracked，保留供迭代）；驗證後可把實測數據補進報告的「多頁 vs 逐頁」段。

## 下一步
1. ~~先做「多圖單請求」可行性驗證~~ ✅ 完成（見下方結果）。
2. ~~跑測試 1（逐頁 vs 多頁）~~ ✅ 完成。
3. （可選，測試 2）頁數拉高（10、20 頁）看長度上限/OOM/品質。
4. 把實測補進 HTML 報告。

---

## 測試 1 結果（2026-06-28，academic 第 3–6 頁、base 模式、RTX 5070 Ti）

腳本：`bench/test_multipage.py`。逐頁與多頁皆用標準防重複 `no_repeat_ngram_size=35 / ngram_window=1024`。

| | CER vs GT ↓ | 輸出 | 耗時 | tok/s | 峰值 VRAM |
|---|---|---|---|---|---|
| **逐頁 base**（現況） | **0.078** | 13166 字 / 3553 tok | **12.7s** | 279 | +0（暖機 ~15.5GB） |
| **多頁一次 base** | 0.358 | 16155 字 / 4155 tok | 18.6s | 223 | +0 |

一致性（多頁 vs 逐頁）：CER 0.31、coverage 0.90、order 0.82。

### 關鍵發現
1. **多圖單請求 SGLang server 有支援**：開頭文字與逐頁完全一致，四頁內容都讀到（coverage 0.90）。
2. **沒有防重複會災難性爆走**：未帶 ngram 時，多頁陷入無限重複、打滿 `max_tokens`（CER 3.27、輸出 4× 長）。逐頁因頁面乾淨未爆。
3. **即使有標準 ngram，多頁仍有殘留句級重複**：整句重複 2–3 次（單句 ~28 token < 35，35-gram 來不及擋）→ 多 ~17% token、CER 0.36。
4. **多頁更慢、品質更差**：18.6s vs 12.7s（**0.68×，不是更快**）、CER 0.36 vs 0.08。每 token 也較慢（223 vs 279 tok/s，因每步要注意 4×256=1024 個視覺 tokens）。即使扣掉多餘 token，多頁仍慢於逐頁。
5. **VRAM**：4 頁未撐爆，兩者峰值相同。

### 結論 / 建議
- **維持逐頁部署**。在這顆 SGLang serving wheel 上，多頁模式**沒有速度或品質優勢**（更慢、更差、需更強的防重複）。
- 推測：server 端未完整複製 HF `infer_multi` 的 R-SWA ring-buffer + 專用 `SlidingWindowNoRepeatNgramProcessor`，導致殘留重複。若真要多頁，應走 HF `infer_multi` 路徑而非 SGLang server。
- 報告的「多頁 vs 逐頁」段可從「程式碼推論」升級為「**實測佐證**」。

---

## 測試 2 結果（頁數壓力，2026-06-28）

同腳本，頁數遞增。多頁皆帶標準 ngram 防重複。

| 頁數 / 文件 | 模式 | CER vs GT ↓ | coverage ↑ | 輸出 tok | 耗時 | speed |
|---|---|---|---|---|---|---|
| 4（academic 3–6） | 逐頁 | 0.078 | 0.95 | 3553 | 12.7s | — |
| | 多頁 | 0.358 | 0.91 | 4155 | 18.6s | 0.68× |
| 10（academic 3–12） | 逐頁 | 0.170 | 0.90 | 10521 | 40.7s | — |
| | 多頁 | 0.835 | 0.68 | 14129 | 74.4s | 0.55× |
| 20（FSOC 1–20） | 逐頁 | 0.095 | 0.98 | 17782 | 68.1s | — |
| | 多頁 | 0.719 | **0.63** | 11291 | 65.6s | 1.04× |

### 關鍵發現（退化隨頁數加劇）
- **4 頁**：過度重複（句級重複 2–3 次）→ 輸出偏多、CER 0.36。
- **10 頁**：開始**漏內容**，coverage 掉到 0.68。
- **20 頁**：多頁輸出**卡在第 6 頁的目錄（Table of Contents），後半 14 頁完全沒讀** → coverage 僅 0.63、輸出只有逐頁的 ~35%。
- **無 OOM / 無 context 崩潰**：20 頁仍在預留 KV 池內，VRAM 全程平（idle ~15.6GB 是 `mem-fraction-static 0.8` 預留，與單頁/多頁無關）。
- **逐頁在所有頁數皆優**：CER 0.09–0.17、coverage 0.90–0.98，速度更快或相當。

### 最終結論
**多頁模式在這顆 SGLang serving wheel 不可用** —— 頁數一多就丟失 per-page 結構而卡住/漏頁，且更慢。**逐頁部署嚴格更佳**，維持現狀。若日後真需要「一次多頁」，需走 HF `infer_multi` 路徑（含其專用 R-SWA ring-buffer 與 `SlidingWindowNoRepeatNgramProcessor`），不是 OpenAI 相容 server 的多圖請求。

### 待辦
- [x] 測試 1（4 頁逐頁 vs 多頁）。
- [x] 測試 2（10／20 頁壓力測）。
- [ ] 把測試 1+2 數據與結論補進 `docs/Unlimited-OCR-test-report.html`（「多頁 vs 逐頁」段：程式碼推論 → 實測佐證）。
