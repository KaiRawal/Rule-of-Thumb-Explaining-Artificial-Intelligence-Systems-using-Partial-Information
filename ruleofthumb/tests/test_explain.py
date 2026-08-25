import numpy as np
import pytest
import torch


@pytest.fixture
def tabular_data():
    rng = np.random.RandomState(0)
    x = rng.randn(64, 5).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)  # black-box labels (CrossEntropy needs int)
    return x, y


@pytest.fixture
def text_data():
    rng = np.random.RandomState(1)
    x = rng.randn(32, 6, 4).astype(np.float32)
    lengths = torch.tensor([6] * 8 + [4] * 12 + [5] * 12)
    y = (x[:, 0, 0] > 0).astype(np.int64)
    return x, lengths, y


def test_tabular_wrapper_explanation_shape(tabular_data):
    from ruleofthumb import RuleOfThumb

    x, y = tabular_data
    rot = RuleOfThumb(y, x, epochs=4, batch_size=32, learning_rate=0.05)
    exp = rot.get_explanation(x)
    assert exp.shape == (64, 5)


def test_text_wrapper_explanation_shape(text_data):
    from ruleofthumb import TextRuleOfThumb

    x, lengths, y = text_data
    rot = TextRuleOfThumb(y, x, epochs=4, batch_size=16, learning_rate=0.01, lengths=lengths)
    exp = rot.get_explanation(x, lengths=lengths)
    assert exp.shape == (32, 6)


def test_text_wrapper_attention_mask_matches_lengths(text_data):
    from ruleofthumb import TextRuleOfThumb

    x, lengths, y = text_data
    mask = torch.arange(6)[None, :] < lengths[:, None]
    torch.manual_seed(0)
    by_lengths = TextRuleOfThumb(y, x, epochs=2, batch_size=16, learning_rate=0.01, lengths=lengths)
    torch.manual_seed(0)
    by_mask = TextRuleOfThumb(y, x, epochs=2, batch_size=16, learning_rate=0.01, attention_mask=mask.numpy())
    exp_by_lengths = by_lengths.get_explanation(x, lengths=lengths)
    exp_by_mask = by_mask.get_explanation(x, attention_mask=mask.numpy())
    assert np.allclose(exp_by_lengths, exp_by_mask)


def test_package_exports():
    import ruleofthumb

    assert ruleofthumb.__version__ == "0.2.3"
    assert hasattr(ruleofthumb, "RoT")
