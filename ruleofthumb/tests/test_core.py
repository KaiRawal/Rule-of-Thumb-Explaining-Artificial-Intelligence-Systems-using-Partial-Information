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


def test_linear_regression_projects_b_to_zero():
    from ruleofthumb.models import Linear_regression

    model = Linear_regression(2, (4,))
    model.b.data += 1.0
    model.project()
    assert torch.allclose(model.b.data, torch.zeros_like(model.b.data))


def test_rand_order_is_permutation():
    from ruleofthumb.models import rand_order

    points = torch.randn(8, 4)
    order = rand_order(points)
    assert order.shape == points.shape
    for row in order:
        assert sorted(row.tolist()) == list(range(4))
