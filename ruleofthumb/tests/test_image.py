import numpy as np
import pytest
import torch

from ruleofthumb.image import RoTImage, pad_images


@pytest.fixture
def image_data():
    rng = np.random.RandomState(0)
    # N x channels x width x height
    x = rng.randn(8, 3, 4, 4).astype(np.float32)
    y = torch.randint(0, 2, (8,))
    return x, y


def test_rot_image_importance_shape(image_data):
    x, _ = image_data
    model = RoTImage(2, (x.shape[1],))
    imp = model.importance(torch.from_numpy(x))
    assert tuple(imp.shape) == (8, 2, 3, 4, 4)


def test_rot_image_fit_and_score(image_data):
    x, y = image_data
    model = RoTImage(2, (x.shape[1],))
    model.fit(torch.from_numpy(x), y, epochs=4, batch_size=8, lr=0.05)
    score = model.score(torch.from_numpy(x))
    assert score.shape == (8, 2)


def test_mask_zeroes_padded_pixels():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8)
    mask = torch.zeros(2, 8, 8, dtype=torch.bool)
    mask[:, :4, :5] = True
    model = RoTImage(2, (3,))
    torch.nn.init.normal_(model.a, std=1.0)
    torch.nn.init.normal_(model.b, std=1.0)
    imp = model.importance(x, mask=mask)
    assert torch.all(imp[:, :, :, 5:, :] == 0)
    assert torch.all(imp[:, :, 4:, :, :] == 0)
    assert torch.any(imp != 0)


def test_mixed_size_batch_matches_per_sample_loop():
    """Padded+masked scoring equals scoring each unpadded sample individually."""
    torch.manual_seed(0)
    rng = np.random.RandomState(1)
    small = rng.randn(1, 3, 4, 4).astype(np.float32)
    big = rng.randn(1, 3, 6, 7).astype(np.float32)

    padded, mask = pad_images([small[0], big[0]], pad_value=0.0)
    model = RoTImage(2, (3,))
    torch.nn.init.normal_(model.a, std=1.0)
    torch.nn.init.normal_(model.b, std=1.0)

    batch_scores = model.score(padded, mask=mask)
    loop_small = model.score(torch.from_numpy(small))[0]
    loop_big = model.score(torch.from_numpy(big))[0]

    assert torch.allclose(batch_scores[0], loop_small, atol=1e-5)
    assert torch.allclose(batch_scores[1], loop_big, atol=1e-5)


def test_fit_with_padded_mask():
    rng = np.random.RandomState(0)
    imgs = [rng.randn(3, h, w).astype(np.float32) for h, w in [(4, 4), (6, 5), (5, 6)]]
    labels = torch.randint(0, 2, (3,))
    padded, mask = pad_images(imgs)
    model = RoTImage(2, (3,))
    model.fit(padded, labels, epochs=4, batch_size=3, lr=0.05, mask=mask)
    assert len(model.training_loss) == 4

    scores = model.score(padded, mask=mask)
    assert scores.shape == (3, 2)


def test_pad_images_utility():
    rng = np.random.RandomState(0)
    imgs = [rng.randn(3, 4, 4).astype(np.float32), rng.randn(3, 6, 5).astype(np.float32)]
    padded, mask = pad_images(imgs, pad_value=9.0)
    assert tuple(padded.shape) == (2, 3, 6, 5)
    assert tuple(mask.shape) == (2, 6, 5)
    assert torch.all(mask[0, :4, :4])
    assert not mask[0, 4:, :].any()
    assert torch.all(padded[0, :, 4:, :] == 9.0)
    assert torch.equal(padded[1, :, :6, :5], torch.from_numpy(imgs[1]))


def test_facade_binary_pipeline_with_untrained_conv_black_box():
    """Binary image explanations against a real (untrained) conv black box."""
    from ruleofthumb import fit_image

    torch.manual_seed(0)
    black_box = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(8 * 8 * 8, 2),
    )
    x = torch.randn(16, 1, 8, 8)
    with torch.no_grad():
        y = black_box(x).argmax(1)

    exp = fit_image(y.numpy(), x.numpy(), epochs=4, batch_size=8, learning_rate=0.05, device="cpu")
    assert exp.modality == "image"
    assert exp.model.classes == 2
    assert exp.model.a.device.type == "cpu"  # device passthrough

    imp = exp.get_explanation(x.numpy())
    assert imp.shape == (16, 8, 8)
    assert np.isfinite(imp).all()

    # ordering is deterministic and covers every pixel exactly once
    order_a = exp.get_order(x)
    order_b = exp.get_order(x)
    assert np.array_equal(order_a, order_b)
    assert np.array_equal(np.sort(order_a.reshape(16, -1), axis=1), np.tile(np.arange(64), (16, 1)))


def test_facade_binary_padded_batch_respects_mask():
    from ruleofthumb import fit_image
    from ruleofthumb.image import pad_images as _pad_images

    torch.manual_seed(1)
    black_box = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 6 * 5, 2),
    )
    small = torch.randn(2, 1, 4, 4)
    big = torch.randn(2, 1, 6, 5)
    padded_probe, _ = _pad_images([small[0], big[0]])
    with torch.no_grad():
        labels = black_box(padded_probe).argmax(1)

    padded, mask = _pad_images([small[0], big[0]])
    exp = fit_image(labels.numpy(), padded.numpy(), mask=mask.numpy(), epochs=4, batch_size=2, learning_rate=0.05)
    imp = exp.get_explanation(padded.numpy(), mask=mask.numpy())
    assert imp.shape == (2, 6, 5)
    assert (imp[0, 4:, :] == 0).all()  # sample 0's bottom rows are padding
    assert (imp != 0).any()
