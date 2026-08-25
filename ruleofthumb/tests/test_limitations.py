"""Characterisation tests pinning limitations preserved from the original code.

The original research code only handles rectangular batches. By design
decision, ruleofthumb keeps exactly this behaviour (see ToDo.md):

- text inputs are ``(N, T, E)`` tensors where padding **must** be encoded as
  all ``-1`` embedding values (the sentinel mask);
- image batches require equal spatial sizes across samples (the RoT weights
  themselves are spatially shared and size-agnostic, batching is not).
"""

import numpy as np
import pytest
import torch

from ruleofthumb.image import RoT_image
from ruleofthumb.text import RoT_text


def test_text_ragged_input_list_is_rejected():
    seqs = [np.random.RandomState(i).randn(t, 4).astype(np.float32) for i, t in enumerate([3, 5, 4])]
    model = RoT_text(2, (max(len(s) for s in seqs), 4))
    with pytest.raises((TypeError, ValueError, AttributeError)):
        # ragged lists cannot be turned into a tensor: rectangular batches only
        model.score([torch.from_numpy(s) for s in seqs])


def test_text_non_sentinel_padding_is_not_masked():
    """Only all ``-1`` embeddings act as padding; other fill values leak through."""
    torch.manual_seed(0)
    x = torch.randn(4, 3, 4)
    x[:, -1:, :] = 0.0  # zero-padding instead of the -1 sentinel
    model = RoT_text(2, (3, 4))
    torch.nn.init.normal_(model.a, std=1.0)
    torch.nn.init.normal_(model.b, std=1.0)
    imp = model.importance(x)
    assert torch.any(imp[:, :, -1, :] != 0)  # not masked — documented quirk


def test_image_weights_are_size_agnostic_per_sample():
    """The same fitted model can score any spatial size, one sample at a time."""
    rng = np.random.RandomState(0)
    small = rng.randn(2, 3, 4, 4).astype(np.float32)
    big = rng.randn(1, 3, 8, 9).astype(np.float32)
    y = torch.randint(0, 2, (2,))
    model = RoT_image(2, (3,))
    model.fit(torch.from_numpy(small), y, epochs=4, batch_size=2, lr=0.05)

    imp_big = model.importance(torch.from_numpy(big))
    assert tuple(imp_big.shape) == (1, 2, 3, 8, 9)


def test_image_batch_with_mixed_sizes_cannot_be_built():
    a = np.random.RandomState(0).randn(3, 4, 4).astype(np.float32)
    b = np.random.RandomState(1).randn(3, 8, 8).astype(np.float32)
    with pytest.raises(ValueError):
        np.stack([a, b])  # rectangular batches only — pad/bucket externally


def test_text_fit_accepts_only_rectangular_padded_batches():
    rng = np.random.RandomState(2)
    x = rng.randn(16, 6, 4).astype(np.float32)
    x[:, -2:, :] = -1.0  # sentinel padding
    y = torch.from_numpy((x[:, 0, 0] > 0).astype(np.int64))
    model = RoT_text(2, (6, 4))
    model.fit(torch.from_numpy(x), y, epochs=4, batch_size=8, lr=0.01)
    assert len(model.training_loss) == 4
