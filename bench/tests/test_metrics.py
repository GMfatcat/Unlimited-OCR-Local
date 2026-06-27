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
