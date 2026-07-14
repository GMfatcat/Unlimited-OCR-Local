#!/usr/bin/env python3
"""安裝後硬化 vendored sglang wheel：讓「壞掉的 custom_logit_processor 字串」只失敗
該 request，而不會炸掉整個 engine（GPU 脫離）。

背景 / 根因
-----------
`custom_logit_processor` 是一段 `to_str()` 產生的 JSON(內含 dill hex)。若這段字串在
搬運途中被截斷 / 再 escape / 去引號而損壞，送到 SGLang server 後：

  scheduler 行程 SamplingBatchInfo.from_schedule_batch()
    -> CustomLogitProcessor.from_str() -> orjson.loads() 丟例外
    -> 例外冒到 run_scheduler_process 頂層(無 per-request 保護)
    -> parent_process.send_signal(SIGQUIT) 打死整個 engine

亦即單一壞 request 能拖垮整台 server（DoS 級）。tokenizer_manager 的既有驗證只檢查
有沒有開 --enable-custom-logit-processor，完全不驗字串內容。

本 patch 加兩層防護（defense-in-depth）
--------------------------------------
1) tokenizer_manager._validate_one_request：提早反序列化驗證，壞字串以清楚錯誤 reject
   該 request（client 收到有意義的 400，而不是服務暴斃）。這一層即可完全屏蔽 scheduler。
2) sampling_batch_info.from_schedule_batch：最後防線。即使有字串繞過上面驗證抵達
   scheduler，from_str() 失敗也只略過該 processor + log，engine 存活，絕不 SIGQUIT。

好字串行為完全不變。這是對「pinned wheel」做的來源修補，idempotent、可重跑；anchor
找不到會 fail（提醒 wheel 版本變動需重新核對），不會默默不套用。

用法：以「目標 venv 的 python」執行本檔即可（會用該 python 的 sglang 安裝路徑）：
    /opt/uocr/venv/bin/python scripts/patch_sglang_hardening.py
"""
import os
import sys

MARKER = "UOCR-HARDEN"

# ---- Patch 1: scheduler 端最後防線（from_str 失敗只略過該 processor，不炸 engine） ----
SBI_OLD = '''            merged_custom_logit_processor = {
                hash(processor_str): (
                    # The deserialized custom logit processor object
                    CustomLogitProcessor.from_str(processor_str),
                    # The mask tensor for the requests that use this custom logit processor
                    torch.zeros(len(reqs), dtype=torch.bool)
                    .scatter_(0, torch.tensor(true_indices), True)
                    .to(device, non_blocking=True),
                )
                for processor_str, true_indices in processor_dict.items()
            }'''

SBI_NEW = '''            # UOCR-HARDEN: 壞掉的 custom_logit_processor 不可炸掉 scheduler。上游此處若
            # from_str() 丟例外會冒到 run_scheduler_process 頂層 -> SIGQUIT 整個 engine
            # (GPU 脫離)。改逐一 try/except：壞字串只略過該 processor(對應 request 本次不
            # 套用)，engine 存活；正常字串行為不變。
            merged_custom_logit_processor = {}
            for processor_str, true_indices in processor_dict.items():
                try:
                    _uocr_clp_obj = CustomLogitProcessor.from_str(processor_str)
                except Exception as _uocr_e:  # noqa: BLE001
                    logger.error(
                        "UOCR-HARDEN: skip undeserializable custom_logit_processor "
                        "(%d request(s) affected): %r",
                        len(true_indices),
                        _uocr_e,
                    )
                    continue
                merged_custom_logit_processor[hash(processor_str)] = (
                    _uocr_clp_obj,
                    torch.zeros(len(reqs), dtype=torch.bool)
                    .scatter_(0, torch.tensor(true_indices), True)
                    .to(device, non_blocking=True),
                )
            # UOCR-HARDEN-END'''

# ---- Patch 2: 入口驗證（壞字串以清楚錯誤 reject，並屏蔽 scheduler） ----
TM_OLD = '''            if (
                obj.custom_logit_processor
                and not self.server_args.enable_custom_logit_processor
            ):
                raise ValueError(
                    "The server is not configured to enable custom logit processor. "
                    "Please set `--enable-custom-logit-processor` to enable this feature."
                )'''

TM_NEW = TM_OLD + '''

            # UOCR-HARDEN: 提早驗證 custom_logit_processor 能否反序列化。壞字串(搬運途中
            # 被截斷/改寫)在此以清楚錯誤 reject 該 request，避免抵達 scheduler 端 from_str()
            # 失敗 -> SIGQUIT 整個 engine(GPU 脫離)。
            if (
                obj.custom_logit_processor
                and self.server_args.enable_custom_logit_processor
            ):
                from sglang.srt.sampling.custom_logit_processor import (
                    CustomLogitProcessor as _UocrCLP,
                )

                _uocr_clp = obj.custom_logit_processor
                _uocr_list = _uocr_clp if isinstance(_uocr_clp, list) else [_uocr_clp]
                for _uocr_s in _uocr_list:
                    if not _uocr_s:
                        continue
                    try:
                        _UocrCLP.from_str(_uocr_s)
                    except Exception as _uocr_e:  # noqa: BLE001
                        raise ValueError(
                            "Invalid custom_logit_processor: failed to deserialize "
                            f"({type(_uocr_e).__name__}: {_uocr_e}). The string is "
                            "likely truncated or altered in transit; send the exact "
                            "output of CustomLogitProcessor.to_str() unchanged."
                        ) from _uocr_e
            # UOCR-HARDEN-END'''


def _sglang_srt_dir():
    try:
        import sglang
    except ImportError:
        sys.exit("ERROR: 找不到 sglang（請用目標 venv 的 python 執行本檔）")
    return os.path.join(os.path.dirname(sglang.__file__), "srt")


def _apply(path, old, new, label):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if MARKER in src:
        print(f"[skip] {label}: 已含 {MARKER}（idempotent）")
        return False
    if old not in src:
        sys.exit(
            f"ERROR: {label}: 找不到預期的 anchor —— vendored sglang 版本可能變動，"
            f"請重新核對 {path} 後更新本 patch。"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.replace(old, new, 1))
    print(f"[done] {label}: 已套用 {MARKER}")
    return True


def main():
    srt = _sglang_srt_dir()
    _apply(
        os.path.join(srt, "sampling", "sampling_batch_info.py"),
        SBI_OLD, SBI_NEW, "sampling_batch_info(scheduler 最後防線)",
    )
    _apply(
        os.path.join(srt, "managers", "tokenizer_manager.py"),
        TM_OLD, TM_NEW, "tokenizer_manager(入口驗證)",
    )
    print("UOCR-HARDEN: sglang 硬化完成。")


if __name__ == "__main__":
    main()
