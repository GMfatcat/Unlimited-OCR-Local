"""純函式品質指標（可單元測試）。GT 為 PyMuPDF 抽出的文字層 pseudo-GT。"""
import os
import re
import sys
import unicodedata

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ocr_backends import strip_markup  # noqa: E402


def normalize(text, lower=False):
    t = strip_markup(text or "")
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("-\n", "")            # 去連字斷行
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower() if lower else t


def cer(gt, pred):
    g, p = normalize(gt), normalize(pred)
    if not g:
        return 0.0 if not p else 1.0
    return Levenshtein.distance(g, p) / len(g)


def _tokens(text):
    return [w for w in normalize(text).split(" ") if w]


def coverage(gt, pred):
    gt_toks = _tokens(gt)
    if not gt_toks:
        return 1.0
    pred_set = set(_tokens(pred))
    return sum(1 for w in gt_toks if w in pred_set) / len(gt_toks)


def order_insensitive_sim(gt, pred):
    return fuzz.token_sort_ratio(normalize(gt), normalize(pred)) / 100.0
