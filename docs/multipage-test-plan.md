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
1. 先做「多圖單請求」可行性驗證（1 份 2–3 頁）。
2. 通過後跑測試 1（逐頁 vs 多頁：一致性 / 耗時 / VRAM）。
3. （可選，測試 2）頁數拉高（10、20 頁）看長度上限/OOM/品質。
4. 把實測補進 HTML 報告。
