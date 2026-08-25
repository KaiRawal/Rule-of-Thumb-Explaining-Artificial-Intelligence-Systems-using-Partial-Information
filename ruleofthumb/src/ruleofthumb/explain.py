"""High-level Rule-of-Thumb explainer wrappers.

Faithful port of the ``RuleOfThumb`` wrappers from the original experiment
code:

- ``RuleOfThumb``: tabular wrapper (``AdversarialAttack/rule_of_thumb.py``).
- ``TextRuleOfThumb``: text/LLM-embedding wrapper
  (``MovieReviewSentiments/Code/rule_of_thumb.py``).

Both hard-code two output classes; see ``ToDo.md``.
"""

import numpy as np
import torch

from ruleofthumb.core import RoT
from ruleofthumb.text import RoT_text


class RuleOfThumb:
    """Tabular RoT explainer."""

    def __init__(self, y_outputs, x_inputs, epochs=500, batch_size=5000, learning_rate=0.05, dropout_rate=0.5) -> None:
        y_preds = y_outputs.flatten()
        self._explainer_model = RoT(2, (x_inputs.shape[1],), dropout_rate=dropout_rate)
        xx = torch.from_numpy(x_inputs)
        yy = torch.from_numpy(y_preds)
        self._explainer_model.fit(xx.to(torch.float), yy, epochs=epochs, batch_size=batch_size, lr=learning_rate)

    def get_explanation(self, x_numpy) -> np.array:
        """Return per-feature importances of shape ``[N, d]``."""
        x = torch.from_numpy(x_numpy)
        imp = self._explainer_model.importance(x).detach().numpy()
        imp = np.abs(imp).sum(1)
        imp = imp.reshape(imp.shape[0], -1)
        return imp


class TextRuleOfThumb:
    """Text / LLM-embedding RoT explainer (token x embedding inputs)."""

    def __init__(self, y_outputs, x_inputs, epochs=500, batch_size=5000, learning_rate=0.05, dropout_rate=0.5) -> None:
        y_preds = y_outputs.flatten()
        self._explainer_model = RoT_text(2, (x_inputs.shape[1], x_inputs.shape[2]), dropout_rate=dropout_rate)
        xx = torch.from_numpy(x_inputs)
        yy = torch.from_numpy(y_preds)
        self._explainer_model.fit(xx.to(torch.float), yy, epochs=epochs, batch_size=batch_size, lr=learning_rate)

    def get_explanation(self, x_numpy) -> np.array:
        """Return per-token importances for class 1, shape ``[N, tokens]``."""
        x = torch.from_numpy(x_numpy)
        imp = self._explainer_model.importance(x).detach().numpy()
        imp = imp[:, 1, :, :].sum(axis=2)
        imp = imp.reshape(imp.shape[0], -1)
        return imp

    def _get_exp_abs_sum(self, x_numpy) -> np.array:
        """Return abs-summed per-token importances, shape ``[N, tokens]``."""
        x = torch.from_numpy(x_numpy)
        imp = self._explainer_model.importance(x).detach().numpy()
        imp = np.abs(imp).sum(1)
        imp = imp.reshape(imp.shape[0], -1)
        return imp

    def _get_exp_sum(self, x_numpy) -> np.array:
        """Return summed per-token importances, shape ``[N, tokens]``."""
        x = torch.from_numpy(x_numpy)
        imp = self._explainer_model.importance(x).detach().numpy()
        imp = imp.sum(1)
        imp = imp.reshape(imp.shape[0], -1)
        return imp

    def _get_exp_0m1(self, x_numpy) -> np.array:
        """Return class0-minus-class1 importances, shape ``[N, tokens]``."""
        x = torch.from_numpy(x_numpy)
        imp = self._explainer_model.importance(x).detach().numpy()
        imp = imp[:, 0, :] - imp[:, 1, :]
        imp = imp.reshape(imp.shape[0], -1)
        return imp

    def _get_exp_1m0(self, x_numpy) -> np.array:
        """Return class1-minus-class0 importances, shape ``[N, tokens]``."""
        x = torch.from_numpy(x_numpy)
        imp = self._explainer_model.importance(x).detach().numpy()
        imp = imp[:, 1, :] - imp[:, 0, :]
        imp = imp.reshape(imp.shape[0], -1)
        return imp
