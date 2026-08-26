"""High-level Rule-of-Thumb explainer facade.

One entry point for all three modalities:

- :func:`fit` — auto-detects the modality from the input shape;
- :func:`fit_tabular` / :func:`fit_text` / :func:`fit_image` — explicit,
  each accepting only its own keyword arguments.

All factories return a fitted :class:`Explainer` whose ``get_explanation``
returns signed, SHAP-comparable importances: class-1 contributions for
binary tasks, full per-class output for ``n_classes > 2``.
"""

import functools

import numpy as np
import torch

from ruleofthumb.core import RoT
from ruleofthumb.embed import embed_texts
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


def _is_string_batch(x):
    """True when ``x`` is a non-empty list/tuple of raw strings."""
    return isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(s, str) for s in x)


def _as_float_inputs(x_inputs):
    return torch.from_numpy(np.asarray(x_inputs)).to(torch.float)


def _as_labels(y_outputs):
    return torch.as_tensor(np.asarray(y_outputs)).flatten()


class Explainer:
    """Fitted Rule-of-Thumb explainer.

    Create instances through :func:`fit`, :func:`fit_tabular`,
    :func:`fit_text` or :func:`fit_image` rather than constructing directly.

    A text explainer fitted from raw strings remembers how to embed them:
    every public method then accepts the same strings in place of
    ``(N, tokens, embedding)`` arrays, re-embedding (and re-deriving padding
    masks) on each call.

    The underlying model is available as :attr:`model` (a :class:`~ruleofthumb.core.RoT`
    subclass) for full manual control; the reveal-pipeline methods
    (:meth:`get_order`, :meth:`ordered_predict`, :meth:`score_ordering`, ...)
    delegate to it verbatim.
    """

    def __init__(self, model, modality, string_embedder=None):
        self._model = model
        self._modality = modality
        self._string_embedder = string_embedder

    @property
    def modality(self):
        """Which pipeline this explainer runs: ``"tabular"``, ``"text"`` or ``"image"``."""
        return self._modality

    @property
    def model(self):
        """The fitted underlying RoT model."""
        return self._model

    def _resolve_strings(self, x):
        """Embed a string batch; returns ``(points tensor, mask tensor)`` or ``None``."""
        if not _is_string_batch(x):
            return None
        if self._string_embedder is None:
            raise ValueError(
                "this explainer was fitted on embedding arrays and cannot consume raw strings; "
                "pass arrays or refit from strings"
            )
        embedded = self._string_embedder(list(x))
        points = torch.from_numpy(embedded.embeddings).to(self._model.device)
        mask = torch.from_numpy(embedded.attention_mask)
        return points, mask

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
        accepts at most one of ``mask`` / ``attention_mask`` / ``lengths``
        (or none at all when ``x_numpy`` is a list of raw strings — padding
        is derived automatically), image accepts ``mask`` only.
        """
        resolved = self._resolve_strings(x_numpy) if self._modality == "text" else None
        if resolved is not None:
            if mask is not None or attention_mask is not None or lengths is not None:
                raise ValueError("raw strings derive their padding automatically; pass no padding arguments")
            x, text_mask = resolved
        else:
            x = torch.from_numpy(np.asarray(x_numpy)).to(self._model.device)
            text_mask = None
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
            if resolved is None:
                padding = lengths if attention_mask is None else attention_mask
                if padding is None:
                    padding = mask
                text_mask = _as_token_mask(padding, x.shape[1])
            imp = self._model.importance(x, mask=text_mask)
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

    _MASK_METHODS = frozenset({"get_order", "score", "predict"})

    def _delegate(self, name, args, kwargs):
        """Call the raw-model method, embedding a leading string batch for text."""
        if self._modality == "text" and args and _is_string_batch(args[0]):
            if "mask" in kwargs and kwargs["mask"] is not None:
                raise ValueError("raw strings derive their padding automatically; pass mask=None")
            points, mask = self._resolve_strings(args[0])
            args = (points,) + args[1:]
            if name in self._MASK_METHODS:
                kwargs["mask"] = mask
        return getattr(self._model, name)(*args, **kwargs)

    def get_order(self, *args, **kwargs):
        """See :meth:`ruleofthumb.core.RoT.get_order`. Accepts raw strings for text explainers."""
        return self._delegate("get_order", args, kwargs)

    def ordered_predict(self, *args, **kwargs):
        """See :meth:`ruleofthumb.core.RoT.ordered_predict`. Accepts raw strings for text explainers."""
        return self._delegate("ordered_predict", args, kwargs)

    def score(self, *args, **kwargs):
        """See :meth:`ruleofthumb.core.RoT.score`. Accepts raw strings for text explainers."""
        return self._delegate("score", args, kwargs)

    def predict(self, *args, **kwargs):
        """See :meth:`ruleofthumb.core.RoT.predict`. Accepts raw strings for text explainers."""
        return self._delegate("predict", args, kwargs)

    def score_ordering(self, *args, **kwargs):
        """See :meth:`ruleofthumb.core.RoT.score_ordering`. Accepts raw strings for text explainers."""
        return self._delegate("score_ordering", args, kwargs)


def fit(y_outputs, x_inputs, *, modality="auto", **kwargs):
    """Fit an :class:`Explainer` to black-box outputs, auto-detecting the modality.

    ``modality="auto"`` dispatches on the input: a list of raw strings →
    text; otherwise on ``x_inputs.ndim``: 2 → tabular, 3 → text
    ``(N, tokens, embedding)``, 4 → image ``(N, C, H, W)``. Pass
    ``modality="tabular" | "text" | "image"`` explicitly to override. The
    remaining keyword arguments are forwarded to the matching factory
    (:func:`fit_tabular`, :func:`fit_text` or :func:`fit_image`).
    """
    if modality == "auto":
        if _is_string_batch(x_inputs):
            modality = "text"
        else:
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
    tokenizer=None,
    model=None,
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
    """Fit a text :class:`Explainer` on ``(N, tokens, embedding)`` inputs or raw strings.

    Pass a list of raw strings and they are embedded automatically with the
    bundled default HuggingFace model (:data:`ruleofthumb.DEFAULT_TEXT_MODEL`);
    supply ``tokenizer`` / ``model`` (a pre-loaded ``AutoModel``) to override
    it. Padding masks are derived automatically — ``lengths`` /
    ``attention_mask`` must not be given alongside strings.

    For array inputs, padding is explicit: pass ``attention_mask`` (an
    ``(N, T)`` boolean array) or ``lengths`` (per-sample token counts);
    HuggingFace tokenizer ``attention_mask`` tensors compose directly.
    Without either, every token is treated as real data.

    Explainers fitted from strings accept the same strings back in every
    public method; each call re-embeds the texts.
    """
    string_embedder = None
    if _is_string_batch(x_inputs):
        if lengths is not None or attention_mask is not None:
            raise ValueError("raw strings derive their padding automatically; pass neither lengths nor attention_mask")
        string_embedder = functools.partial(embed_texts, tokenizer=tokenizer, model=model, device=device)
        embedded = string_embedder(list(x_inputs))
        x_inputs = embedded.embeddings
        mask = torch.from_numpy(embedded.attention_mask)
    else:
        mask = _as_token_mask(lengths if attention_mask is None else attention_mask, x_inputs.shape[1])
    rot = RoTText(
        n_classes,
        (x_inputs.shape[1], x_inputs.shape[2]),
        dropout_rate=dropout_rate,
        l1_penalty=l1_penalty,
        device=device,
    )
    rot.fit(
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
    return Explainer(rot, "text", string_embedder=string_embedder)


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
