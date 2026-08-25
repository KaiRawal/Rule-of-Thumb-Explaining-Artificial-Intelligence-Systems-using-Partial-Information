"""Tests pinning v0.2 mask-aware behaviour (replaces the v0.1 limitation pins).

Since v0.2 padding is explicit: masks are passed by callers and no fill value
has special meaning. These tests pin:

- text/image models accept rectangular batches plus validity masks;
- ragged inputs are handled by ``pad_sequences`` / ``pad_images`` utilities;
- the incremental-reveal pipeline (``get_order`` / ``ordered_predict`` /
  ``score_ordering``) is mask-aware;
- legacy ``-1`` sentinel workflows migrate via ``sentinel_mask``.
"""

import numpy as np
import pytest
import torch

from ruleofthumb.core import RoT
from ruleofthumb.image import RoTImage, pad_images
from ruleofthumb.text import RoTText, pad_sequences, sentinel_mask


def lengths_to_bool(lengths, max_len):
    return torch.arange(max_len)[None, :] < lengths[:, None]


@pytest.fixture
def legacy_sentinel_text():
    rng = np.random.RandomState(3)
    x = rng.randn(8, 7, 4).astype(np.float32)
    x[:, -3:, :] = -1.0  # v0.1-style sentinel padding
    lengths = torch.full((8,), 4, dtype=torch.long)
    y = torch.from_numpy((x[:, 0, 0] > 0).astype(np.int64))
    return torch.from_numpy(x), lengths, y


def test_ragged_text_via_pad_sequences():
    seqs = [np.random.RandomState(i).randn(t, 4).astype(np.float32) for i, t in enumerate([3, 7, 5])]
    padded, lengths = pad_sequences(seqs)
    model = RoTText(2, (padded.shape[1], padded.shape[2]))
    imp = model.importance(padded, mask=lengths_to_bool(lengths, padded.shape[1]))
    assert tuple(imp.shape) == (3, 2, 7, 4)
    assert torch.all(imp[0, :, 3:, :] == 0)


def test_v01_sentinel_scores_reproduce_with_explicit_mask(legacy_sentinel_text):
    """Explicit ``sentinel_mask`` + mask-aware scoring matches v0.1 semantics."""
    torch.manual_seed(0)
    x, _, _ = legacy_sentinel_text
    model = RoTText(2, (7, 4))
    torch.nn.init.normal_(model.a, std=1.0)
    torch.nn.init.normal_(model.b, std=1.0)

    mask = sentinel_mask(x)
    score = model.score(x, mask=mask)

    # v0.1 formula: mean over non-padded tokens of per-token importance sums,
    # summed over embedding dims, plus g.
    imp = model.a[None, :, None, :] * (x[:, None] + model.b[None, :, None, :])
    expected = imp[:, :, :4, :].sum(dim=(2, 3)) / 4 + model.g[None, :]
    assert torch.allclose(score, expected)
    # without the mask, padded tokens would leak into the score (v0.1 quirk gone)
    assert not torch.allclose(score, model.score(x))


def test_image_mixed_sizes_fit_and_reveal(legacy_sentinel_text):
    rng = np.random.RandomState(0)
    imgs = [rng.randn(3, h, w).astype(np.float32) for h, w in [(4, 4), (6, 7), (5, 5)]]
    labels = torch.randint(0, 2, (3,))
    padded, mask = pad_images(imgs)

    model = RoTImage(2, (3,))
    model.fit(padded, labels, epochs=4, batch_size=3, lr=0.05, mask=mask)

    order = model.get_order(padded, mask=mask)
    flat = order.reshape(3, -1)
    # reveal steps operate at feature-element granularity (channels x pixels)
    true_counts = np.array([3 * h * w for h, w in [(4, 4), (6, 7), (5, 5)]])
    assert (np.asarray(flat != -1).sum(1) == true_counts).all()
    # real positions form a permutation prefix per sample
    for i, (h, w) in enumerate([(4, 4), (6, 7), (5, 5)]):
        valid = np.repeat(mask[i].reshape(-1).numpy(), 3)  # expand pixels to channel elements
        expected_elements = set(np.flatnonzero(valid).tolist())
        assert sorted(flat[i][: true_counts[i]].tolist()) == sorted(expected_elements)
        assert (flat[i][true_counts[i] :] == -1).all()


def test_ordered_predict_truncates_by_default():
    torch.manual_seed(0)
    x = torch.randn(4, 6, 3)
    lengths = torch.tensor([6, 3, 5, 2])
    mask = lengths_to_bool(lengths, 6)
    model = RoTText(2, (6, 3))
    torch.nn.init.normal_(model.a, std=1.0)
    torch.nn.init.normal_(model.b, std=1.0)

    order = model.get_order(x, mask=mask)

    # the reveal pipeline operates at feature-element granularity
    # (tokens x embedding dims), as in v0.1
    elements = lengths * 3
    pred = model.ordered_predict(x, order)
    assert pred.shape == (4, int(elements.max()) + 1)
    assert (pred[:, 0] != -1).all()
    for i, elem in enumerate(elements.tolist()):
        assert (pred[i, : elem + 1] != -1).all()
        if elem + 1 < pred.shape[1]:
            assert (pred[i, elem + 1 :] == -1).all()

    full = model.ordered_predict(x, order, include_padded=True)
    assert full.shape == (4, 19)  # all 6 x 3 rectangular steps retained
    assert (full != -1).all()  # legacy-style constant tail retained


def test_score_ordering_curve_lengths():
    torch.manual_seed(0)
    x = torch.randn(4, 6, 3)
    lengths = torch.tensor([6, 3, 5, 2])
    mask = lengths_to_bool(lengths, 6)
    labels = torch.randint(0, 2, (4,))
    model = RoTText(2, (6, 3))

    order = model.get_order(x, mask=mask)
    metric = model.score_ordering(x, labels, order)
    assert metric.shape == (19,)  # max valid feature elements + 1

    full = model.score_ordering(x, labels, order, include_padded=True)
    assert full.shape == (19,)
    assert torch.isfinite(full).all()


def test_unmasked_behaviour_is_unchanged_from_v01():
    """Without masks the ordering pipeline behaves exactly as before."""
    rng = np.random.RandomState(0)
    x = torch.from_numpy(rng.randn(16, 5).astype(np.float32))
    model = RoT(2, (5,))
    model.fit(x, torch.randint(0, 2, (16,)), epochs=4, batch_size=8, lr=0.05)

    order = model.get_order(x)
    for row in order:
        assert sorted(row.tolist()) == list(range(5))  # full permutation, no -1
    assert (order >= 0).all()

    pred = model.ordered_predict(x, order)
    assert pred.shape == (16, 6)
    assert (pred != -1).all()


def test_legacy_imports_are_gone():
    import ruleofthumb

    assert not hasattr(ruleofthumb, "RoT_text")
    assert not hasattr(ruleofthumb, "RoT_image")


@pytest.mark.parametrize("util", ["pad_sequences", "sentinel_mask", "pad_images"])
def test_padding_utils_exported(util):
    import ruleofthumb

    assert hasattr(ruleofthumb, util)
