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
