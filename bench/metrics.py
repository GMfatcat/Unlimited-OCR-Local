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


# CJK / 假名 / 諺文：這些文字沒有詞間空白，用空白切詞會把整行變成一個巨大 token，
# 使 coverage / order_sim 失準。改成「CJK 每字一個 token、非 CJK 仍照詞」，對中英混排都有意義。
_CJK = "一-鿿㐀-䶿豈-﫿぀-ヿ가-힯"
_TOKEN_RE = re.compile(f"[{_CJK}]|[^\\s{_CJK}]+")


def _tokens(text):
    return _TOKEN_RE.findall(normalize(text))


def coverage(gt, pred):
    gt_toks = _tokens(gt)
    if not gt_toks:
        return 1.0
    pred_set = set(_tokens(pred))
    return sum(1 for w in gt_toks if w in pred_set) / len(gt_toks)


def order_insensitive_sim(gt, pred):
    g = " ".join(sorted(_tokens(gt)))
    p = " ".join(sorted(_tokens(pred)))
    if not g and not p:
        return 1.0
    return fuzz.ratio(g, p) / 100.0
