import numpy as np
import pytest


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
    x[:, -2:, :] = -1.0  # pad last two tokens
    y = (x[:, 0, 0] > 0).astype(np.int64)
    return x, y


def test_tabular_wrapper_explanation_shape(tabular_data):
    from ruleofthumb import RuleOfThumb

    x, y = tabular_data
    rot = RuleOfThumb(y, x, epochs=4, batch_size=32, learning_rate=0.05)
    exp = rot.get_explanation(x)
    assert exp.shape == (64, 5)


def test_text_wrapper_explanation_shape(text_data):
    from ruleofthumb import TextRuleOfThumb

    x, y = text_data
    rot = TextRuleOfThumb(y, x, epochs=4, batch_size=16, learning_rate=0.01)
    exp = rot.get_explanation(x)
    assert exp.shape == (32, 6)
    # padded tokens get zero importance for the class-1 channel
    assert np.all(exp[:, -2:] == 0)


def test_package_exports():
    import ruleofthumb

    assert ruleofthumb.__version__ == "0.1.0"
    assert hasattr(ruleofthumb, "RoT")
