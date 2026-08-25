# ruleofthumb

Rule of Thumb (RoT): explaining AI systems using partial information.

RoT trains a simple, transparent surrogate ("rule of thumb") on partial
observations of a black-box model's behaviour. The surrogate attributes the
model's output to input features, producing feature importances that can be
revealed incrementally (most-important-first) while preserving the black box's
predictions.

This package consolidates the three research variants of RoT into one
installable library:

- **Tabular** (`ruleofthumb.explain.RuleOfThumb`): vector inputs, one
  importance weight per feature.
- **Text / LLM embeddings** (`ruleofthumb.text.RoTText`): token-by-embedding
  inputs with padding masks and length-normalised scores.
- **Images** (`ruleofthumb.image.RoTImage`, `RoTImageMixed`): importance shared
  across spatial locations of CNN feature maps.

## Install

```bash
pip install ruleofthumb              # core (numpy + torch)
pip install "ruleofthumb[viz]"       # plotting utilities
```

For local development:

```bash
git clone <this repo>
cd Rule-of-Thumb-Explaining-Artificial-Intelligence-Systems-using-Partial-Information/ruleofthumb
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
pytest
```

## Quick start

See `examples/01_tabular_quickstart.ipynb`.

```python
from ruleofthumb import RuleOfThumb

rot = RuleOfThumb(y_outputs=black_box_probs, x_inputs=X_train)
importances = rot.get_explanation(X_test)  # [N, d]
```

## Limitations (preserved from the original research code)

- **Rectangular batches only.** Text inputs are `(N, T, E)` tensors; ragged
  sequences must be padded to a common length by the caller. Image batches
  require equal `H × W` across samples.
- **Text padding must be all `-1` embeddings.** Tokens whose embedding vector
  is entirely `-1` are masked; any other fill value is treated as real data.
- The RoT image weights are spatially shared, so a fitted model can score
  images of any size one sample at a time.

See `ToDo.md` for the full list and `tests/test_limitations.py` for pinned
behaviour.

## Status

v0.1.0 is a faithful structural port of the original experiment code; known
hard-coded limitations are tracked in `ToDo.md`.
