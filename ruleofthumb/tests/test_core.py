import numpy as np
import pytest
import torch

from ruleofthumb.core import RoT


@pytest.fixture
def tabular_data():
    rng = np.random.RandomState(0)
    x = rng.randn(64, 5).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)  # simple linear rule (CrossEntropy needs int labels)
    return x, torch.from_numpy(y)


def test_fit_reduces_loss_and_predicts(tabular_data):
    x, y = tabular_data
    model = RoT(2, (5,))
    model.fit(torch.from_numpy(x), y, epochs=8, batch_size=32, lr=0.05)
    assert model.training_loss is not None
    assert len(model.training_loss) == 8
    assert model.training_loss[-1] <= model.training_loss[0]

    score = model.score(torch.from_numpy(x))
    assert score.shape == (64, 2)
    pred = model.predict(torch.from_numpy(x))
    assert pred.shape == (64,)
    accuracy = (pred == y).float().mean().item()
    assert accuracy > 0.7  # the rule is trivially learnable


def test_importance_shape(tabular_data):
    x, _ = tabular_data
    model = RoT(2, (5,))
    imp = model.importance(torch.from_numpy(x))
    assert tuple(imp.shape) == (64, 2, 5)


def test_ordering_pipeline(tabular_data):
    x, y = tabular_data
    model = RoT(2, (5,))
    model.fit(torch.from_numpy(x), y, epochs=4, batch_size=32, lr=0.05)

    order = model.get_order(torch.from_numpy(x))
    assert order.shape == (64, 5)
    # each row is a permutation of 0..4
    for row in order:
        assert sorted(row.tolist()) == list(range(5))

    pred = model.ordered_predict(torch.from_numpy(x), order)
    assert pred.shape == (64, 6)

    labels = torch.randint(0, 2, (64,))
    metric = model.score_ordering(torch.from_numpy(x), labels, order)
    # one fidelity value per prefix of revealed features (0..d features shown)
    assert metric.shape == (6,)


def test_stochastic_importance_dropout():
    torch.manual_seed(0)
    x = torch.randn(16, 3)
    model = RoT(2, (3,), dropout_rate=0.5)
    imp = model.stochastic_importance(x)
    assert tuple(imp.shape) == (16, 2, 3)


def test_mins_maxs_are_instance_attributes():
    model_a = RoT(2, (3,))
    model_b = RoT(2, (3,))
    assert model_a.mins == -np.inf
    assert model_a.maxs == np.inf

    # mutating one instance must not leak into others or the class
    model_a.fit_project(-1.0, 1.0)
    assert model_b.mins == -np.inf
    assert model_b.maxs == np.inf
    assert not hasattr(RoT, "mins")
    assert not hasattr(RoT, "maxs")


def test_get_order_stable_tie_breaking(tabular_data):
    x, _ = tabular_data
    model = RoT(2, (5,))
    # unfitted model: a and b are zero, so every feature has equal importance
    order = model.get_order(torch.from_numpy(x))
    expected = np.arange(5)
    for row in order:
        assert np.array_equal(row, expected)

    # repeated calls are deterministic
    again = model.get_order(torch.from_numpy(x))
    assert np.array_equal(order, again)


def test_training_loop_swa_burn_in_parameter(tabular_data):
    x, y = tabular_data
    model = RoT(2, (5,))
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.05)

    # default burn-in (epochs // 10 + 1 == 1) creates the SWA model
    model.training_loop(model.loss, torch.from_numpy(x), y, optimiser, epochs=2, batch_size=32)
    assert model.swa_model is not None

    # a burn-in beyond the epoch count never creates it
    model = RoT(2, (5,))
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.05)
    model.training_loop(model.loss, torch.from_numpy(x), y, optimiser, epochs=2, batch_size=32, swa_burn_in=5)
    assert model.swa_model is None

    # an explicit zero burn-in works too
    model = RoT(2, (5,))
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.05)
    model.training_loop(model.loss, torch.from_numpy(x), y, optimiser, epochs=2, batch_size=32, swa_burn_in=0)
    assert model.swa_model is not None


def test_fit_hyperparameter_arguments(tabular_data):
    x, y = tabular_data
    model = RoT(2, (5,))
    model.fit(
        torch.from_numpy(x), y, epochs=8, batch_size=32, lr=0.05, pretrain_epochs=2, weight_decay=0.1
    )
    assert len(model.training_loss) == 8
    assert model.training_loss[-1] <= model.training_loss[0]
