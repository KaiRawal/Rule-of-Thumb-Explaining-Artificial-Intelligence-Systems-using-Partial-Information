"""Core Rule-of-Thumb explainer module.

Faithful port of the ``RoT`` base class from the original experiment code
(e.g. ``AdversarialAttack/rot_class.py``). Known hard-coded limitations are
tracked in the repository-level ``ToDo.md``.
"""

import numpy as np
import torch
from numpy import prod
from torch import nn
from torch.optim.swa_utils import AveragedModel


class RoT(torch.nn.Module):
    def __init__(self, classes, sample_shape, dropout_rate=0.5, use_BCE_loss=False, no_a_b=False):
        super().__init__()
        if not no_a_b:
            self.a = nn.Parameter(torch.zeros((classes,) + sample_shape, requires_grad=True))
            self.b = nn.Parameter(torch.zeros((classes,) + sample_shape, requires_grad=True))
        self.g = nn.Parameter(torch.zeros(classes, requires_grad=True))
        self.classes = classes
        if use_BCE_loss is False:
            self.objective = torch.nn.CrossEntropyLoss(reduction="sum")
        else:
            self.objective = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.dropout_rate = dropout_rate
        if not no_a_b:
            self.weights = (self.a, self.b, self.g)
        self.training_loss = None
        self.use_BCE_loss = use_BCE_loss
        self.swa_model = None
        self.mins = -np.inf
        self.maxs = np.inf

    def forward(self, x):
        "Warning. Forward should only be used at eval, at training use stochastic importance"
        return self.importance(x)

    def _set_swa_model(self, model):
        # Bypass ``nn.Module.__setattr__`` so the SWA copy is NOT registered as
        # a submodule. Registering it would double the reported parameter count
        # mid-training, which crashes ``update_parameters`` on torch >= 2.x
        # (older torch silently tolerated this via non-strict zips).
        object.__setattr__(self, "swa_model", model)

    def importance(self, points, mask=None):
        """Per-feature importance.

        ``mask`` (optional) marks real (non-padding) feature positions; padded
        positions receive zero importance. The tabular variant has no padding
        concept and ignores the argument; text/image subclasses interpret it
        as ``(N, tokens)`` / ``(N, H, W)`` validity masks respectively.
        """
        return self.a[None] * (points[:, None] + self.b[None])

    def stochastic_importance(self, points, mask=None):
        imp = self.importance(points, mask=mask)
        mask_shape = (points.shape[0],) + tuple(imp.shape[2:])
        keep = (torch.rand(mask_shape) > self.dropout_rate).float()
        return keep[:, None] * imp

    def pretrain_loss(self, points, classifier_response):
        imp = self.importance(points)
        imp = imp.reshape(imp.shape[0], imp.shape[1], -1) + self.g[:, None]
        cl = classifier_response.repeat(imp.shape[2], 1).T
        loss = self.objective(imp, cl)
        loss /= prod(self.b.shape[1:])
        loss += self.objective(self.g.repeat(imp.shape[0], 1), classifier_response)
        return loss / 2

    def loss(self, points, target, mask=None):
        response = self.stochastic_importance(points, mask=mask)
        response = response.reshape(points.shape[0], self.classes, -1).sum(-1)
        response = response + self.g[None]
        return self.objective(response, target)

    def fit_project(self, mins, maxs):
        self.mins = mins
        self.maxs = maxs

    def project(self):
        # Equivalent to the original ``min(max(b, mins), maxs)``; ``torch.clamp``
        # accepts either scalars or tensors so it also works before ``fit_project``.
        self.b.data = torch.clamp(self.b.data, self.mins, self.maxs)

    def training_loop(self, loss, points, classifier_response, optimiser, epochs=1, batch_size=200, mask=None):
        self.training_loss = np.zeros(epochs)
        burn_in = epochs // 10 + 1
        features = points
        for e in range(epochs):
            shuff = torch.randperm(classifier_response.shape[0])
            for i in range(points.shape[0] // batch_size + 1):
                upper = min(points.shape[0], batch_size * (i + 1))
                lower = batch_size * i
                if lower == upper:
                    break

                shuff_inner = shuff[lower:upper]
                target = classifier_response[shuff_inner]
                features = points[shuff_inner]
                mask_inner = mask[shuff_inner] if mask is not None else None

                l = loss(features, target, mask=mask_inner)
                l /= batch_size
                if self.use_BCE_loss:
                    l /= self.classes
                self.training_loss[e] += l.item()
                l.backward()
                optimiser.step()
                optimiser.zero_grad()
                self.project()
            if e > burn_in:
                self.swa_model.update_parameters(self)
            elif e == burn_in:
                self._set_swa_model(AveragedModel(self))

        self.training_loss /= features.shape[0] / batch_size

    def fit(self, points, classifier_response, epochs, batch_size, lr=1e-4):
        assert points.shape[0] == classifier_response.shape[0]
        assert points.shape[1] == self.a.shape[1]
        self.fit_project(-points.max(0)[0], -points.min(0)[0])
        with torch.no_grad():
            self.b.copy_(-points.mean(0))
        optimiser = torch.optim.AdamW(self.parameters(), lr=lr)
        drop_out = self.dropout_rate
        self.dropout_rate = 0
        self.training_loop(self.loss, points, classifier_response, optimiser, 5, batch_size)
        self.dropout_rate = drop_out
        optimiser = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        self.training_loop(self.loss, points, classifier_response, optimiser, epochs, batch_size)

    def averaged_explainer(self):
        return list(self.swa_model.children())[0]  # noqa: RUF015 (kept verbatim from source)

    def score(self, points, mask=None):
        imp = self.importance(points, mask=mask).detach()
        score = imp.reshape(imp.shape[0], imp.shape[1], -1).sum(-1)
        score += self.g[None, :]
        return score

    def predict(self, points, mask=None):
        score = self.score(points, mask=mask)
        return score.argmax(1)

    def _reduce_to_units(self, imp):
        """Collapse per-element importances to reveal-pipeline units.

        Input and output are signed importance tensors of shape
        ``(N, classes, *sample_shape)`` -> ``(N, classes, *unit_shape)``.
        The tabular variant is a no-op (unit == feature element); text
        aggregates over embedding dims, images over channels.
        """
        return imp

    def ordered_predict(self, points, order, include_padded=False, granularity="unit"):
        """Predict after revealing units best-first according to ``order``.

        ``granularity`` must match how ``order`` was produced by
        :meth:`get_order`: ``"unit"`` (default) reveals whole tokens (text) /
        pixels (images) per step; ``"element"`` reveals individual feature
        elements. There is no auto-detection — mixing them silently misaligns.

        ``order`` may contain ``-1`` entries marking padded positions. With
        ``include_padded=False`` (default) reveal steps where every sample has
        exhausted its real features are dropped, and each sample's predictions
        past its own reveal end are filled with ``-1``; the result spans
        ``max_true_units + 1`` steps. ``include_padded=True`` keeps every
        step of the rectangular input order.
        """
        imp = self.importance(points).detach()
        n = imp.shape[0]
        if granularity == "unit":
            imp = self._reduce_to_units(imp)
        elif granularity != "element":
            raise ValueError(f"unknown granularity: {granularity!r}")
        positions = int(prod(imp.shape[2:]))

        flat_imp = imp.reshape(n, self.classes, positions).permute(0, 2, 1)

        flat_order = np.asarray(order).reshape(n, -1)
        assert flat_order.min() >= -1
        assert flat_order.max() <= positions - 1

        if include_padded:
            kept = np.arange(flat_order.shape[1])
        else:
            kept = np.flatnonzero((flat_order != -1).any(axis=0))
        reveal_counts = (flat_order != -1).sum(axis=1)

        pred = torch.full((n, len(kept) + 1), -1, dtype=int)
        acc = torch.zeros(n, self.classes)
        acc += self.g[None].detach()
        pred[:, 0] = acc.argmax(1)

        for j, i in enumerate(kept):
            idx = flat_order[:, i]
            valid = idx != -1
            if valid.any():
                contribution = torch.zeros(n, self.classes)
                contribution[torch.from_numpy(valid)] = flat_imp[
                    torch.from_numpy(valid.nonzero()[0]), torch.from_numpy(idx[valid])
                ]
                acc = acc + contribution
            pred[:, j + 1] = acc.argmax(1)

        if not include_padded:
            cols = np.arange(pred.shape[1])[None, :]
            exhausted = cols > np.asarray(reveal_counts)[:, None]
            pred[torch.from_numpy(exhausted)] = -1
        return pred

    def get_order(self, points, mask=None, granularity="unit"):
        """Rank reveal units by absolute importance, most important first.

        With ``granularity="unit"`` (default) one unit is a token (text),
        pixel (image) or feature (tabular); with ``granularity="element"``
        every individual feature element is ranked separately.

        With a validity ``mask``, padded positions are ranked last and
        reported as ``-1`` in the returned order; real positions form a
        permutation prefix per sample.
        """
        imp = self.importance(points, mask=mask).detach()
        if granularity == "unit":
            imp = self._reduce_to_units(imp)
        elif granularity != "element":
            raise ValueError(f"unknown granularity: {granularity!r}")
        imp = np.abs(imp.numpy()).sum(1)  # abs-sum over classes: n x *unit_shape
        old_shape = imp.shape
        imp = imp.reshape(imp.shape[0], -1)
        flat_mask = None
        if mask is not None:
            flat_mask = np.asarray(mask.reshape(mask.shape[0], -1)).astype(bool)
            repeat, remainder = divmod(imp.shape[1], flat_mask.shape[1])
            if remainder:
                raise ValueError("mask does not align with the input feature shape")
            if repeat > 1:
                # element granularity: expand unit-level mask to elements
                flat_mask = np.repeat(flat_mask, repeat, axis=1)
            imp = np.where(flat_mask, imp, -np.inf)
            counts = flat_mask.sum(axis=1)
        else:
            counts = np.full(imp.shape[0], imp.shape[1])
        order = np.argsort(imp, 1)[:, ::-1].copy()
        cols = np.arange(order.shape[1])[None, :]
        order[cols >= counts[:, None]] = -1
        return order.reshape(old_shape)

    def score_ordering(self, points, labels, order, metric=None, include_padded=False, granularity="unit"):
        """Fidelity metric at each incremental-reveal step.

        ``granularity`` must match how ``order`` was produced (see
        :meth:`get_order`). Steps where no sample reveals a real feature are
        trimmed. Per-step metrics aggregate only samples that still have real
        features left, so denominators can shrink along the curve when reveal
        lengths differ.
        """
        if metric is None:
            metric = lambda tp, fp, fn, tn: (tp + tn) / (tp + fp + fn + tn)
        pred = self.ordered_predict(points, order, include_padded=include_padded, granularity=granularity)
        tp = ((pred == 1).float() * (labels == 1).float()[:, None]).sum(0)
        tn = ((pred == 0).float() * (labels == 0).float()[:, None]).sum(0)
        fp = ((pred == 1).float() * (labels == 0).float()[:, None]).sum(0)
        fn = ((pred == 0).float() * (labels == 1).float()[:, None]).sum(0)
        denom = tp + tn + fp + fn
        last = int((denom > 0).nonzero().max()) if bool((denom > 0).any()) else 0
        return metric(tp[: last + 1], fp[: last + 1], fn[: last + 1], tn[: last + 1])
