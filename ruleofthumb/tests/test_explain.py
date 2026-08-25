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


def test_tabular_wrapper_signed_explanations(tabular_data):
    from ruleofthumb import RuleOfThumb

    x, y = tabular_data
    rot = RuleOfThumb(y, x, epochs=8, batch_size=32, learning_rate=0.05, seed=0)
    exp = rot.get_explanation(x)

    # signed: both positive and negative contributions exist
    assert (exp > 0).any()
    assert (exp < 0).any()

    # additive decomposition: sum of contributions + class-1 bias = score
    import torch

    score1 = rot._explainer_model.score(torch.from_numpy(x))[:, 1].detach().numpy()
    bias1 = rot._explainer_model.g[1].detach().numpy()
    assert np.allclose(exp.sum(1) + bias1, score1, atol=1e-4)


def test_tabular_wrapper_multiclass_per_class_output(tabular_data):
    from ruleofthumb import RuleOfThumb

    x, _ = tabular_data
    y3 = (x[:, 0] > 0).astype(np.int64) + (x[:, 1] > 0).astype(np.int64)  # labels in {0, 1, 2}
    rot = RuleOfThumb(y3, x, epochs=4, batch_size=32, learning_rate=0.05, n_classes=3)
    exp = rot.get_explanation(x)
    assert exp.shape == (64, 3, 5)
    for k in range(3):
        assert (exp[:, k, :] != 0).any()


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


def test_wrappers_thread_training_hyperparameters(tabular_data, text_data):
    from ruleofthumb import RuleOfThumb, TextRuleOfThumb

    x, y = tabular_data
    rot = RuleOfThumb(
        y, x, epochs=4, batch_size=32, learning_rate=0.05, pretrain_epochs=1, weight_decay=0.1
    )
    assert rot.get_explanation(x).shape == (64, 5)

    tx, lengths, ty = text_data
    rot_text = TextRuleOfThumb(
        ty,
        tx,
        epochs=4,
        batch_size=16,
        learning_rate=0.01,
        lengths=lengths,
        pretrain_epochs=1,
        weight_decay=0.1,
        l1_penalty=0.05,
    )
    assert rot_text.get_explanation(tx, lengths=lengths).shape == (32, 6)


def test_wrapper_seed_reproducibility(text_data):
    from ruleofthumb import TextRuleOfThumb

    x, lengths, y = text_data
    rot_a = TextRuleOfThumb(y, x, epochs=4, batch_size=16, learning_rate=0.01, lengths=lengths, seed=0)
    rot_b = TextRuleOfThumb(y, x, epochs=4, batch_size=16, learning_rate=0.01, lengths=lengths, seed=0)
    exp_a = rot_a.get_explanation(x, lengths=lengths)
    exp_b = rot_b.get_explanation(x, lengths=lengths)
    assert np.allclose(exp_a, exp_b)


def test_tabular_wrapper_n_classes(tabular_data):
    from ruleofthumb import RuleOfThumb

    x, _ = tabular_data
    y3 = (x[:, 0] > 0).astype(np.int64) + (x[:, 1] > 0).astype(np.int64)  # labels in {0, 1, 2}
    rot = RuleOfThumb(y3, x, epochs=4, batch_size=32, learning_rate=0.05, n_classes=3)
    assert rot._explainer_model.classes == 3
    exp = rot.get_explanation(x)
    # K > 2: full per-class output, class axis not collapsed
    assert exp.shape == (64, 3, 5)


def test_text_wrapper_multiclass_per_class_output(text_data):
    from ruleofthumb import TextRuleOfThumb

    x, lengths, _ = text_data
    y3 = (x[:, 0, 0] > 0).astype(np.int64) + (x[:, 1, 1] > 0).astype(np.int64)  # labels in {0, 1, 2}
    rot = TextRuleOfThumb(y3, x, epochs=4, batch_size=16, learning_rate=0.01, lengths=lengths, n_classes=3)
    assert rot._explainer_model.classes == 3
    exp = rot.get_explanation(x, lengths=lengths)
    # K > 2: full per-class output, class axis not collapsed
    assert exp.shape == (32, 3, 6)
    for k in range(3):
        assert (exp[:, k, :] != 0).any()
        assert (exp[:, k, :] > 0).any() and (exp[:, k, :] < 0).any()


def test_package_exports():
    import ruleofthumb

    assert ruleofthumb.__version__ == "0.2.8"
    assert hasattr(ruleofthumb, "RoT")
