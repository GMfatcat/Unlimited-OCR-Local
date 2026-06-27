import os

from bench import corpus_io

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_parse_pages_all():
    assert corpus_io.parse_pages("all", 3) == [0, 1, 2]


def test_parse_pages_range():
    assert corpus_io.parse_pages("1-3", 10) == [0, 1, 2]


def test_parse_pages_single():
    assert corpus_io.parse_pages("4", 10) == [3]


def test_load_manifest_applies_defaults(tmp_path):
    m = tmp_path / "manifest.yaml"
    m.write_text(
        "- id: doc1\n"
        "  file: academic/x.pdf\n"
        "  category: academic\n"
        "  tier: simple\n",
        encoding="utf-8",
    )
    entries = corpus_io.load_manifest(str(m))
    assert entries[0].id == "doc1"
    assert entries[0].image_mode == "base"
    assert entries[0].gt == "pdf_text"
    assert entries[0].pages == "all"


def test_pdf_pseudo_gt_extracts_text():
    gt = corpus_io.pdf_pseudo_gt(os.path.join(REPO, "Unlimited-OCR.pdf"), "1")
    assert "Abstract" in gt
