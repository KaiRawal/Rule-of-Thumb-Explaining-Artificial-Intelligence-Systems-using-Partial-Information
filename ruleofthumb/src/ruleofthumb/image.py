"""Image RoT variants.

Faithful port of ``RoT_image`` and ``RoT_image_mixed`` from the original
experiment code (``ExplanationExampleRemote/rot_class.py`` and
``AdversarialAttack/rot_class.py``).
"""

import torch
from torch import nn

from ruleofthumb.core import RoT


class RoT_image(RoT):
    """Share importance between spatial locations.

    Assumes datapoints are in the form channel x width x height.
    """

    def __init__(self, classes, sample_shape, dropout_rate=0.5, use_BCE_loss=False):
        super().__init__(classes, sample_shape, dropout_rate, use_BCE_loss, no_a_b=True)
        self.a = nn.Parameter(torch.zeros((classes, sample_shape[0]), requires_grad=True))
        self.b = nn.Parameter(torch.zeros((classes, sample_shape[0]), requires_grad=True))
        self.weights = (self.a, self.b, self.g)

    def importance(self, points):
        # Convolutional form.
        # Treat all spatial locations given by last two axis the same
        return self.a[None, :, :, None, None] * (points[:, None] + self.b[None, :, :, None, None])

    def fit_project(self, mins, maxs):
        mins = mins.min(-1)[0]
        mins = mins.min(-1)[0]
        maxs = maxs.max(-1)[0]
        maxs = maxs.max(-1)[0]
        print(mins.shape, maxs.shape, self.b.shape)
        assert mins.shape == self.b.shape[1:]
        self.mins = mins
        self.maxs = maxs


class RoT_image_mixed(RoT_image):
    """Compute importance as the product of spatial locations with channels.

    Assumes datapoints are in the form channel x width x height.
    """

    def __init__(self, classes, sample_shape, dropout_rate=0.5, use_BCE_loss=False):
        super().__init__(classes, sample_shape, dropout_rate, use_BCE_loss)
        self.a_spatial = nn.Parameter(torch.zeros((classes, sample_shape[0]), requires_grad=True))

    def importance(self, points):
        return (
            self.a[None, :, None, None]
            * self.a_spatial[None, None, :, :]
            * (points[:, None] + self.b[None, :, None, None])
        )
