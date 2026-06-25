"""
單張圖片用 Transformers 路線做 OCR，結果印到 stdout（給 Streamlit 的 transformers 模式以子行程呼叫）。
在 .venv-transformers 內執行。

用法:
  python ocr_once.py --image <png> --mode gundam|base [--model_dir DIR]
"""
import argparse
import os
import sys
import tempfile

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "/mnt/c/Users/User/Desktop/project/unlimited-ocr/unlimited-ocr-hf"

_MODEL = None
_TOK = None


def _load(model_dir):
    global _MODEL, _TOK
    if _MODEL is None:
        _TOK = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        _MODEL = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).eval().cuda()
    return _MODEL, _TOK


def run(image, mode, model_dir):
    model, tok = _load(model_dir)
    out_dir = tempfile.mkdtemp(prefix="ocr_once_")
    if mode == "gundam":
        kwargs = dict(base_size=1024, image_size=640, crop_mode=True, ngram_window=128)
    else:  # base
        kwargs = dict(base_size=1024, image_size=1024, crop_mode=False, ngram_window=1024)
    model.infer(
        tok,
        prompt="<image>document parsing.",
        image_file=image,
        output_path=out_dir,
        max_length=32768,
        no_repeat_ngram_size=35,
        save_results=True,
        **kwargs,
    )
    md_path = os.path.join(out_dir, "result.md")
    if os.path.exists(md_path):
        with open(md_path, encoding="utf-8") as f:
            return f.read()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mode", choices=("gundam", "base"), default="gundam")
    ap.add_argument("--model_dir", default=os.environ.get("MODEL_DIR", DEFAULT_MODEL))
    args = ap.parse_args()
    text = run(args.image, args.mode, args.model_dir)
    # 用明確分隔符包住，方便上層擷取
    sys.stdout.write("\n<<<OCR_RESULT_START>>>\n")
    sys.stdout.write(text)
    sys.stdout.write("\n<<<OCR_RESULT_END>>>\n")


if __name__ == "__main__":
    main()
