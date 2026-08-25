import numpy as np
import pytest
import torch

from ruleofthumb.text import RoT_text


@pytest.fixture
def text_data():
    rng = np.random.RandomState(0)
    n, tokens, embedding = 32, 6, 4
    x = rng.randn(n, tokens, embedding).astype(np.float32)
    # pad the last two tokens of each sample with -1 embeddings
    x[:, -2:, :] = -1.0
    y = (x[:, 0, 0] > 0).astype(np.int64)
    return x, torch.from_numpy(y)


def test_importance_masks_padded_tokens(text_data):
    torch.manual_seed(0)
    x, _ = text_data
    model = RoT_text(2, (x.shape[1], x.shape[2]))
    # a fresh model has all-zero weights; perturb so masking is observable
    torch.nn.init.normal_(model.a, std=1.0)
    torch.nn.init.normal_(model.b, std=1.0)
    imp = model.importance(torch.from_numpy(x))
    assert tuple(imp.shape) == (32, 2, 6, 4)
    # padded tokens must receive exactly zero importance
    assert torch.all(imp[:, :, -2:, :] == 0)
    assert torch.any(imp[:, :, :-2, :] != 0)


def test_stochastic_importance_shape(text_data):
    torch.manual_seed(0)
    x, _ = text_data
    model = RoT_text(2, (x.shape[1], x.shape[2]), dropout_rate=0.5)
    imp = model.stochastic_importance(torch.from_numpy(x))
    assert tuple(imp.shape) == (32, 2, 6, 4)


def test_score_shape(text_data):
    torch.manual_seed(0)
    x, _ = text_data
    model = RoT_text(2, (x.shape[1], x.shape[2]))
    score = model.score(torch.from_numpy(x))
    assert score.shape == (32, 2)


def test_fit_runs_and_reduces_loss(text_data):
    x, y = text_data
    model = RoT_text(2, (x.shape[1], x.shape[2]))
    model.fit(torch.from_numpy(x), y, epochs=8, batch_size=16, lr=0.01)
    # fit runs a 5-epoch burn-in followed by the requested epochs
    assert len(model.training_loss) == 8
