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
