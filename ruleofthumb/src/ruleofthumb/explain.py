"""High-level Rule-of-Thumb explainer facade.

One entry point for all three modalities:

- :func:`fit` — auto-detects the modality from the input shape;
- :func:`fit_tabular` / :func:`fit_text` / :func:`fit_image` — explicit,
  each accepting only its own keyword arguments.

All factories return a fitted :class:`Explainer` whose ``get_explanation``
returns signed, SHAP-comparable importances: class-1 contributions for
binary tasks, full per-class output for ``n_classes > 2``.
"""

import numpy as np
import torch

from ruleofthumb.core import RoT
from ruleofthumb.image import RoTImage
from ruleofthumb.text import RoTText, lengths_to_mask


def _as_token_mask(mask_or_lengths, n_tokens):
    """Normalise an attention mask given as a ``(N, T)`` array or lengths."""
    if mask_or_lengths is None:
        return None
    m = torch.as_tensor(np.asarray(mask_or_lengths))
    if m.ndim == 1:
        return lengths_to_mask(m, n_tokens)
    return m.to(torch.bool)


def _as_float_inputs(x_inputs):
    return torch.from_numpy(np.asarray(x_inputs)).to(torch.float)


def _as_labels(y_outputs):
    return torch.as_tensor(np.asarray(y_outputs)).flatten()


class Explainer:
    """Fitted Rule-of-Thumb explainer.

    Create instances through :func:`fit`, :func:`fit_tabular`,
    :func:`fit_text` or :func:`fit_image` rather than constructing directly.

    The underlying model is available as :attr:`model` (a :class:`~ruleofthumb.core.RoT`
    subclass) for full manual control; the reveal-pipeline methods
    (:meth:`get_order`, :meth:`ordered_predict`, :meth:`score_ordering`, ...)
    delegate to it verbatim.
    """

    def __init__(self, model, modality):
        self._model = model
        self._modality = modality

    @property
    def modality(self):
        """Which pipeline this explainer runs: ``"tabular"``, ``"text"`` or ``"image"``."""
        return self._modality

    @property
    def model(self):
        """The fitted underlying RoT model."""
        return self._model

    def get_explanation(self, x_numpy, *, mask=None, attention_mask=None, lengths=None) -> np.ndarray:
        """Return signed importances, comparable to SHAP values.

        For binary tasks (``n_classes == 2``) the result holds the class-1
        ("positive class") contributions: positive values are evidence toward
        class 1, negative toward class 0. For ``n_classes > 2`` the full
        per-class importances are returned; the class axis is never collapsed.

        Shapes: tabular ``(N, d)`` / ``(N, K, d)``; text ``(N, tokens)`` /
        ``(N, K, tokens)`` (embedding dims summed per token); image
        ``(N, H, W)`` / ``(N, K, H, W)`` (channels summed per pixel). Padded
        positions receive exactly zero importance when a mask/lengths is
        supplied.

        Padding arguments depend on the modality: tabular takes none, text
        accepts at most one of ``mask`` / ``attention_mask`` / ``lengths``,
        image accepts ``mask`` only.
        """
        x = torch.from_numpy(np.asarray(x_numpy)).to(self._model.device)
        given = [
            name
            for name, value in (("mask", mask), ("attention_mask", attention_mask), ("lengths", lengths))
            if value is not None
        ]
        if self._modality == "tabular":
            if given:
                raise ValueError(f"tabular explanations take no padding arguments, got {given}")
            imp = self._model.importance(x)
        elif self._modality == "text":
            if len(given) > 1:
                raise ValueError(f"pass at most one of mask / attention_mask / lengths, got {given}")
            padding = lengths if attention_mask is None else attention_mask
            if padding is None:
                padding = mask
            imp = self._model.importance(x, mask=_as_token_mask(padding, x.shape[1]))
        else:
            if attention_mask is not None or lengths is not None:
                raise ValueError("image explanations take mask= only")
            if mask is not None:
                mask = torch.as_tensor(np.asarray(mask)).to(torch.bool)
            imp = self._model.importance(x, mask=mask)
        imp = self._model._reduce_to_units(imp).detach().cpu().numpy()
        if self._model.classes == 2:
            imp = imp[:, 1]
        return imp

    def get_order(self, *args, **kwargs):
        """See :meth:`ruleofthumb.core.RoT.get_order`."""
        return self._model.get_order(*args, **kwargs)

    def ordered_predict(self, *args, **kwargs):
        """See :meth:`ruleofthumb.core.RoT.ordered_predict`."""
        return self._model.ordered_predict(*args, **kwargs)

    def score(self, *args, **kwargs):
        """See :meth:`ruleofthumb.core.RoT.score`."""
        return self._model.score(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """See :meth:`ruleofthumb.core.RoT.predict`."""
        return self._model.predict(*args, **kwargs)

    def score_ordering(self, *args, **kwargs):
        """See :meth:`ruleofthumb.core.RoT.score_ordering`."""
        return self._model.score_ordering(*args, **kwargs)


def fit(y_outputs, x_inputs, *, modality="auto", **kwargs):
    """Fit an :class:`Explainer` to black-box outputs, auto-detecting the modality.

    ``modality="auto"`` dispatches on ``x_inputs.ndim``: 2 → tabular,
    3 → text ``(N, tokens, embedding)``, 4 → image ``(N, C, H, W)``. Pass
    ``modality="tabular" | "text" | "image"`` explicitly to override. The
    remaining keyword arguments are forwarded to the matching factory
    (:func:`fit_tabular`, :func:`fit_text` or :func:`fit_image`).
    """
    if modality == "auto":
        ndim = np.asarray(x_inputs).ndim
        modality = {2: "tabular", 3: "text", 4: "image"}.get(ndim)
        if modality is None:
            raise ValueError(f"cannot infer a modality from input ndim={ndim}; pass modality= explicitly")
    factories = {"tabular": fit_tabular, "text": fit_text, "image": fit_image}
    if modality not in factories:
        raise ValueError(f"unknown modality: {modality!r}")
    return factories[modality](y_outputs, x_inputs, **kwargs)


def fit_tabular(
    y_outputs,
    x_inputs,
    *,
    epochs=500,
    batch_size=5000,
    learning_rate=0.05,
    dropout_rate=0.5,
    pretrain_epochs=5,
    weight_decay=0.01,
    seed=None,
    n_classes=2,
    device=None,
):
    """Fit a tabular :class:`Explainer` on ``(N, d)`` feature inputs."""
    model = RoT(n_classes, (x_inputs.shape[1],), dropout_rate=dropout_rate, device=device)
    model.fit(
        _as_float_inputs(x_inputs),
        _as_labels(y_outputs),
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        pretrain_epochs=pretrain_epochs,
        weight_decay=weight_decay,
        seed=seed,
    )
    return Explainer(model, "tabular")


def fit_text(
    y_outputs,
    x_inputs,
    *,
    lengths=None,
    attention_mask=None,
    l1_penalty=0.01,
    epochs=500,
    batch_size=5000,
    learning_rate=0.05,
    dropout_rate=0.5,
    pretrain_epochs=5,
    weight_decay=0.01,
    seed=None,
    n_classes=2,
    device=None,
):
    """Fit a text :class:`Explainer` on ``(N, tokens, embedding)`` inputs.

    Padding is explicit: pass ``attention_mask`` (a ``(N, T)`` boolean array)
    or ``lengths`` (per-sample token counts); HuggingFace tokenizer
    ``attention_mask`` tensors compose directly. Without either, every token
    is treated as real data.
    """
    model = RoTText(
        n_classes,
        (x_inputs.shape[1], x_inputs.shape[2]),
        dropout_rate=dropout_rate,
        l1_penalty=l1_penalty,
        device=device,
    )
    mask = _as_token_mask(lengths if attention_mask is None else attention_mask, x_inputs.shape[1])
    model.fit(
        _as_float_inputs(x_inputs),
        _as_labels(y_outputs),
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        mask=mask,
        pretrain_epochs=pretrain_epochs,
        weight_decay=weight_decay,
        seed=seed,
    )
    return Explainer(model, "text")


def fit_image(
    y_outputs,
    x_inputs,
    *,
    mask=None,
    epochs=500,
    batch_size=5000,
    learning_rate=0.05,
    dropout_rate=0.5,
    pretrain_epochs=5,
    weight_decay=0.01,
    seed=None,
    n_classes=2,
    device=None,
):
    """Fit an image :class:`Explainer` on ``(N, C, H, W)`` inputs.

    Pass a boolean validity ``mask`` of shape ``(N, H, W)`` alongside padded
    batches (see :func:`ruleofthumb.image.pad_images`); masked-out pixels
    receive exactly zero importance.
    """
    model = RoTImage(n_classes, (x_inputs.shape[1],), dropout_rate=dropout_rate, device=device)
    if mask is not None:
        mask = torch.as_tensor(np.asarray(mask)).to(torch.bool)
    model.fit(
        _as_float_inputs(x_inputs),
        _as_labels(y_outputs),
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        mask=mask,
        pretrain_epochs=pretrain_epochs,
        weight_decay=weight_decay,
        seed=seed,
    )
    return Explainer(model, "image")
