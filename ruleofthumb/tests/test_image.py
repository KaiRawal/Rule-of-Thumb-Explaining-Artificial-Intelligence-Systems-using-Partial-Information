import numpy as np
import pytest
import torch

from ruleofthumb.image import RoT_image, RoT_image_mixed


@pytest.fixture
def image_data():
    rng = np.random.RandomState(0)
    # N x channels x width x height
    x = rng.randn(8, 3, 4, 4).astype(np.float32)
    y = torch.randint(0, 2, (8,))
    return x, y


def test_rot_image_importance_shape(image_data):
    x, _ = image_data
    model = RoT_image(2, (x.shape[1],))
    imp = model.importance(torch.from_numpy(x))
    assert tuple(imp.shape) == (8, 2, 3, 4, 4)


@pytest.mark.xfail(
    reason="Original RoT_image_mixed.importance does not broadcast for generic "
    "channel/spatial sizes; kept verbatim for fidelity (see ToDo.md).",
    strict=True,
)
def test_rot_image_mixed_importance_shape():
    rng = np.random.RandomState(0)
    x = rng.randn(8, 3, 3, 3).astype(np.float32)
    model = RoT_image_mixed(2, (x.shape[1],))
    imp = model.importance(torch.from_numpy(x))
    assert tuple(imp.shape) == (8, 2, 3, 3, 3)


def test_rot_image_fit_and_score(image_data):
    x, y = image_data
    model = RoT_image(2, (x.shape[1],))
    model.fit(torch.from_numpy(x), y, epochs=4, batch_size=8, lr=0.05)
    score = model.score(torch.from_numpy(x))
    assert score.shape == (8, 2)
