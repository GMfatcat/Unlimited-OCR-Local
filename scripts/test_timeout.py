"""
驗證逾時保護：
  1) 對某頁串流，2 秒後提早 break（模擬逾時中止），確認連線關閉、收到部分 token。
  2) 立刻送一個正常請求，確認 server 已從中止的請求復原、能正常完成下一頁。
需 server 已啟動。在 .venv-sglang 內執行。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_backends import server_healthy, sglang_stream, strip_markup, to_image_paths  # noqa: E402

SERVER = "http://127.0.0.1:10000"
PDF = "/mnt/c/Users/User/Desktop/project/unlimited-ocr/Unlimited-OCR.pdf"
TIMEOUT = 2.0


def main():
    assert server_healthy(SERVER), "server not healthy"
    with open(PDF, "rb") as f:
        imgs = to_image_paths(f.read(), "Unlimited-OCR.pdf", dpi=300)

    # 1) 提早中止
    print(f"[1] stream with {TIMEOUT}s cutoff ...")
    raw, n = "", 0
    t0 = time.time()
    gen = sglang_stream(SERVER, imgs[0], "base", 1024, 35)
    for delta in gen:
        raw += delta
        n += 1
        if time.time() - t0 > TIMEOUT:
            break
    gen.close()  # 觸發 finally → resp.close() → server 端 abort
    print(f"    aborted after {time.time()-t0:.1f}s, deltas={n}, chars={len(strip_markup(raw))}")
    assert n > 0, "no tokens before cutoff"

    # 給 server 一點時間處理斷線
    time.sleep(2)
    assert server_healthy(SERVER), "server unhealthy after abort"

    # 2) 中止後立刻送正常請求，確認可完成
    print("[2] normal request after abort ...")
    raw2, n2 = "", 0
    t1 = time.time()
    for delta in sglang_stream(SERVER, imgs[1], "base", 1024, 35):
        raw2 += delta
        n2 += 1
    print(f"    completed: deltas={n2}, chars={len(strip_markup(raw2))}, {time.time()-t1:.1f}s")

    ok = n > 0 and n2 > 50 and server_healthy(SERVER)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}  (server recovered & served next page)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
