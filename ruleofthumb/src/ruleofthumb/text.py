"""Text RoT variant.

Faithful port of ``RoT_text`` from the original experiment code
(``ExplanationExampleRemote/rot_class.py``). Assumes datapoints are in the
form: token x embedding, with padding encoded as all ``-1`` embedding values.
Known hard-coded limitations are tracked in the repository-level ``ToDo.md``.
"""

import torch
from torch import nn

from ruleofthumb.core import RoT


class RoT_text(RoT):
    """Assumes datapoints are in the form: token x embedding."""

    def __init__(self, classes, sample_shape, dropout_rate=0.5, use_BCE_loss=False, l1_penalty=0.01, sgd=False):
        super().__init__(classes, sample_shape, dropout_rate, use_BCE_loss, no_a_b=True)
        self.a = nn.Parameter(torch.zeros((classes, sample_shape[1]), requires_grad=True))
        self.b = nn.Parameter(torch.zeros((classes, sample_shape[1]), requires_grad=True))
        # self.objective = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.weights = (self.a, self.b, self.g)
        self.l1_penalty = l1_penalty
        self.use_sgd = sgd

    def score(self, points):
        imp = self.importance(points).detach()

        # imp=imp.mean(dim=2)  # mean per essay - normalise by number of tokens (length)
        # Calculate the sum over the 512-element dimension
        response_sum = imp.sum(dim=2)
        # Calculate DELTA: the number of all-zero rows in the 512-element dimension
        zero_rows = (imp == 0).all(dim=3).sum(dim=2)
        # Calculate the modified length
        modified_length = imp.shape[2] - zero_rows
        # Avoid division by zero by ensuring modified_length is at least 1
        modified_length = torch.clamp(modified_length, min=1)
        # Calculate the mean using the modified length
        response_mean = response_sum / modified_length.unsqueeze(-1)

        score = response_mean.reshape(response_mean.shape[0], response_mean.shape[1], -1).sum(-1)
        score += self.g[None, :]
        return score

    def loss(self, points, target):
        response = self.stochastic_importance(points)

        # response=response.mean(dim=2)  # mean per essay - normalise by number of tokens (length)
        # Calculate the sum over the 512-element dimension
        response_sum = response.sum(dim=2)
        # Calculate DELTA: the number of all-zero rows in the 512-element dimension
        zero_rows = (response == 0).all(dim=3).sum(dim=2)
        # Calculate the modified length
        modified_length = response.shape[2] - zero_rows
        # Avoid division by zero by ensuring modified_length is at least 1
        modified_length = torch.clamp(modified_length, min=1)
        # Calculate the mean using the modified length
        response_mean = response_sum / modified_length.unsqueeze(-1)  # noqa: F841 (kept verbatim from source)

        response = response.reshape(points.shape[0], self.classes, -1).sum(-1)
        response = response + self.g[None]
        base_loss = self.objective(response, target)

        l1_loss = (
            self.l1_penalty
            * points.shape[0]
            * self.classes
            * (self.a.abs().sum() + self.b.abs().sum() + self.g.abs().sum())
        )
        return base_loss + l1_loss

    def importance(self, points):
        # Create a deterministic mask that ensures only those points remain which don't have all features == -1 for a given token
        imp = self.a[None, :, None, :] * (points[:, None] + self.b[None, :, None, :])
        feature_dim = 2
        mask = (points.sum(dim=feature_dim) != -points.shape[feature_dim]).float()
        return mask[:, None, :, None] * imp

    def stochastic_importance(self, points):
        imp = self.importance(points)
        # importance is 4d (classes is always added as the 1th dimension), because points became 3d
        # points[index, features] -> points[index, token, features]
        token_dim = 1
        mask = (torch.rand(points.shape[0], points.shape[token_dim]) > self.dropout_rate).float()
        return mask[:, None, :, None] * imp  # put none in correct dimension to match importance shape

    def fit(self, points, classifier_response, epochs, batch_size, lr=1e-4):
        # Overridden relative to the tabular ``RoT.fit``: no projection of ``b``
        # onto the data range, zero initialisation instead of mean-centering,
        # as in the original text-experiment copies.
        self.b[None, :].data = torch.zeros(points.shape[2])
        optimiser = torch.optim.AdamW(self.parameters(), lr=lr)
        drop_out = self.dropout_rate
        self.dropout_rate = 0
        self.training_loop(self.loss, points, classifier_response, optimiser, 5, batch_size)
        self.dropout_rate = drop_out
        optimiser = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        self.training_loop(self.loss, points, classifier_response, optimiser, epochs, batch_size)

    def continue_fit(self, points, classifier_response, epochs, batch_size, lr=1e-4):
        optimiser = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        if self.use_sgd:
            optimiser = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=0.01)
        return self.training_loop(self.loss, points, classifier_response, optimiser, epochs, batch_size)
