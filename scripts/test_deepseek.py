"""
用 DeepSeek-OCR.pdf 實測逾時保護：逐頁串流，單頁超過 TIMEOUT 秒即中止跳下一頁，
回報每頁 token 數與是否逾時，並確認 server 全程持續服務（無限迴圈頁被擋掉）。
需 server 已啟動。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_backends import server_healthy, sglang_stream, strip_markup, to_image_paths  # noqa: E402

SERVER = "http://127.0.0.1:10000"
PDF = os.environ.get("PDF", "/mnt/c/Users/User/Desktop/project/unlimited-ocr/DeepSeek-OCR.pdf")
TIMEOUT = float(os.environ.get("TIMEOUT", "30"))


def main():
    assert server_healthy(SERVER), "server not healthy"
    with open(PDF, "rb") as f:
        imgs = to_image_paths(f.read(), os.path.basename(PDF), dpi=300)
    print(f"pages={len(imgs)}, per-page timeout={TIMEOUT}s\n")

    timeouts = []
    for i, img in enumerate(imgs, 1):
        raw, n = "", 0
        t0 = time.time()
        gen = sglang_stream(SERVER, img, "base", 1024, 35)
        for delta in gen:
            raw += delta
            n += 1
            if time.time() - t0 > TIMEOUT:
                gen.close()
                break
        dt = time.time() - t0
        to = dt > TIMEOUT
        if to:
            timeouts.append(i)
        print(f"  page {i:2d}: {n:5d} tok, {dt:5.1f}s {'  <-- TIMEOUT 中止' if to else ''}"
              f"  | clean chars={len(strip_markup(raw))}")

    print(f"\n逾時頁: {timeouts if timeouts else '無'}")
    healthy = server_healthy(SERVER)
    print(f"server still healthy after all pages: {healthy}")
    print(f"\nRESULT: {'PASS' if healthy else 'FAIL'}")
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
