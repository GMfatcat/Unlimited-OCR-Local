from bench import metrics


def test_normalize_strips_markup_and_collapses_space():
    out = metrics.normalize("<|det|>title [1, 2, 3, 4]<|/det|>Hello   World\n\n")
    assert "det" not in out
    assert out == "Hello World"


def test_cer_identical_is_zero():
    assert metrics.cer("hello world", "hello world") == 0.0


def test_cer_one_substitution_quarter():
    assert abs(metrics.cer("abcd", "abxd") - 0.25) < 1e-9


def test_cer_empty_gt_empty_pred_zero():
    assert metrics.cer("", "") == 0.0


def test_coverage_half():
    assert metrics.coverage("a b c d", "a b zzz") == 0.5


def test_order_insensitive_high_when_reordered():
    assert metrics.order_insensitive_sim("alpha beta gamma", "gamma beta alpha") > 0.99


# ── CJK-aware tokenisation（中文無詞間空白，每字一 token；英數仍照詞）──
def test_tokens_cjk_per_char_latin_per_word():
    assert metrics._tokens("今天 hello 123") == ["今", "天", "hello", "123"]


def test_coverage_cjk_partial():
    # GT 6 字，pred 缺「很好」→ 命中 4/6
    assert abs(metrics.coverage("今天天氣很好", "今天天氣") - 4 / 6) < 1e-9


def test_order_insensitive_cjk_reordered_high():
    assert metrics.order_insensitive_sim("甲乙丙丁", "丁丙乙甲") > 0.99


def test_order_insensitive_cjk_different_low():
    assert metrics.order_insensitive_sim("今天天氣很好", "完全不同的句子內容") < 0.5
