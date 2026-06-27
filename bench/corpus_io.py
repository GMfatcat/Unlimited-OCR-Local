"""語料 manifest 載入、頁碼解析、pseudo-GT 抽取。"""
from dataclasses import dataclass

import fitz  # PyMuPDF
import yaml

_DEFAULTS = {"source": "", "pages": "all", "image_mode": "base", "gt": "pdf_text", "notes": ""}


@dataclass
class DocEntry:
    id: str
    file: str
    category: str
    tier: str
    source: str = ""
    pages: str = "all"
    image_mode: str = "base"
    gt: str = "pdf_text"
    notes: str = ""


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []
    return [DocEntry(**{**_DEFAULTS, **d}) for d in raw]


def parse_pages(spec, total):
    spec = str(spec).strip()
    if spec == "all":
        return list(range(total))
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a) - 1, min(int(b), total)))
    return [int(spec) - 1]


def pdf_pseudo_gt(pdf_path, pages):
    with fitz.open(pdf_path) as doc:
        idxs = parse_pages(pages, doc.page_count)
        text = "\n".join(doc[i].get_text() for i in idxs)
    return text
