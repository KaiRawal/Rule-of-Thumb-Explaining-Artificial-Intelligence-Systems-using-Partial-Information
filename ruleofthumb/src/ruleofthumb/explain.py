"""High-level Rule-of-Thumb explainer wrappers.

Port of the ``RuleOfThumb`` wrappers from the original experiment code:

- ``RuleOfThumb``: tabular wrapper (``AdversarialAttack/rule_of_thumb.py``).
- ``TextRuleOfThumb``: text/LLM-embedding wrapper
  (``MovieReviewSentiments/Code/rule_of_thumb.py``).

Both wrap models whose class count is configurable via ``n_classes``.
"""

import numpy as np
import torch

from ruleofthumb.core import RoT
from ruleofthumb.text import RoTText, lengths_to_mask


def _as_token_mask(mask_or_lengths, n_tokens):
    """Normalise an attention mask given as a ``(N, T)`` array or lengths."""
    if mask_or_lengths is None:
        return None
    m = torch.as_tensor(np.asarray(mask_or_lengths))
    if m.ndim == 1:
        return lengths_to_mask(m, n_tokens)
    return m.to(torch.bool)


class RuleOfThumb:
    """Tabular RoT explainer."""

    def __init__(
        self,
        y_outputs,
        x_inputs,
        epochs=500,
        batch_size=5000,
        learning_rate=0.05,
        dropout_rate=0.5,
        pretrain_epochs=5,
        weight_decay=0.01,
        seed=None,
        n_classes=2,
    ) -> None:
        y_preds = y_outputs.flatten()
        self._explainer_model = RoT(n_classes, (x_inputs.shape[1],), dropout_rate=dropout_rate)
        xx = torch.from_numpy(x_inputs)
        yy = torch.from_numpy(y_preds)
        self._explainer_model.fit(
            xx.to(torch.float),
            yy,
            epochs=epochs,
            batch_size=batch_size,
            lr=learning_rate,
            pretrain_epochs=pretrain_epochs,
            weight_decay=weight_decay,
            seed=seed,
        )

    def get_explanation(self, x_numpy) -> np.ndarray:
        """Return per-feature importances of shape ``[N, d]``."""
        x = torch.from_numpy(x_numpy)
        imp = self._explainer_model.importance(x).detach().numpy()
        imp = np.abs(imp).sum(1)
        imp = imp.reshape(imp.shape[0], -1)
        return imp


class TextRuleOfThumb:
    """Text / LLM-embedding RoT explainer (token x embedding inputs).

    Padding is explicit since v0.2: pass ``attention_mask`` (a ``(N, T)``
    boolean array) or ``lengths`` (per-sample token counts). Composes directly
    with HuggingFace tokenizer ``attention_mask`` outputs. Without either,
    every token is treated as real data.
    """

    def __init__(
        self,
        y_outputs,
        x_inputs,
        epochs=500,
        batch_size=5000,
        learning_rate=0.05,
        dropout_rate=0.5,
        attention_mask=None,
        lengths=None,
        pretrain_epochs=5,
        weight_decay=0.01,
        l1_penalty=0.01,
        seed=None,
        n_classes=2,
    ) -> None:
        y_preds = y_outputs.flatten()
        self._explainer_model = RoTText(
            n_classes, (x_inputs.shape[1], x_inputs.shape[2]), dropout_rate=dropout_rate, l1_penalty=l1_penalty
        )
        mask = _as_token_mask(lengths if attention_mask is None else attention_mask, x_inputs.shape[1])
        xx = torch.from_numpy(x_inputs)
        yy = torch.from_numpy(y_preds)
        self._explainer_model.fit(
            xx.to(torch.float),
            yy,
            epochs=epochs,
            batch_size=batch_size,
            lr=learning_rate,
            mask=mask,
            pretrain_epochs=pretrain_epochs,
            weight_decay=weight_decay,
            seed=seed,
        )

    def get_explanation(self, x_numpy, attention_mask=None, lengths=None) -> np.ndarray:
        """Return per-token importances for class 1, shape ``[N, tokens]``.

        Padded tokens receive exactly zero importance when a mask/lengths is
        supplied.
        """
        x = torch.from_numpy(x_numpy)
        mask = _as_token_mask(lengths if attention_mask is None else attention_mask, x.shape[1])
        imp = self._explainer_model.importance(x, mask=mask).detach().numpy()
        imp = imp[:, 1, :, :].sum(axis=2)
        imp = imp.reshape(imp.shape[0], -1)
        return imp

    def _get_exp_abs_sum(self, x_numpy, attention_mask=None, lengths=None) -> np.ndarray:
        """Return abs-summed per-token importances, shape ``[N, tokens]``."""
        x = torch.from_numpy(x_numpy)
        mask = _as_token_mask(lengths if attention_mask is None else attention_mask, x.shape[1])
        imp = self._explainer_model.importance(x, mask=mask).detach().numpy()
        imp = np.abs(imp).sum(1)
        imp = imp.reshape(imp.shape[0], -1)
        return imp

    def _get_exp_sum(self, x_numpy, attention_mask=None, lengths=None) -> np.ndarray:
        """Return summed per-token importances, shape ``[N, tokens]``."""
        x = torch.from_numpy(x_numpy)
        mask = _as_token_mask(lengths if attention_mask is None else attention_mask, x.shape[1])
        imp = self._explainer_model.importance(x, mask=mask).detach().numpy()
        imp = imp.sum(1)
        imp = imp.reshape(imp.shape[0], -1)
        return imp

    def _get_exp_0m1(self, x_numpy, attention_mask=None, lengths=None) -> np.ndarray:
        """Return class0-minus-class1 importances, shape ``[N, tokens]``."""
        x = torch.from_numpy(x_numpy)
        mask = _as_token_mask(lengths if attention_mask is None else attention_mask, x.shape[1])
        imp = self._explainer_model.importance(x, mask=mask).detach().numpy()
        imp = imp[:, 0, :] - imp[:, 1, :]
        imp = imp.reshape(imp.shape[0], -1)
        return imp

    def _get_exp_1m0(self, x_numpy, attention_mask=None, lengths=None) -> np.ndarray:
        """Return class1-minus-class0 importances, shape ``[N, tokens]``."""
        x = torch.from_numpy(x_numpy)
        mask = _as_token_mask(lengths if attention_mask is None else attention_mask, x.shape[1])
        imp = self._explainer_model.importance(x, mask=mask).detach().numpy()
        imp = imp[:, 1, :] - imp[:, 0, :]
        imp = imp.reshape(imp.shape[0], -1)
        return imp
