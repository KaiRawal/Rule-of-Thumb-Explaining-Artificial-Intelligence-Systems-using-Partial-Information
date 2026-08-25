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
- **Images** (`ruleofthumb.image.RoTImage`): importance shared across spatial
  locations of CNN feature maps.

## Install

Requires Python >= 3.9. A single install ships everything (core + plotting +
LLM/vision helpers):

```bash
pip install ruleofthumb
```

Or from this repository:

```bash
git clone <this repo>
cd Rule-of-Thumb-Explaining-Artificial-Intelligence-Systems-using-Partial-Information/ruleofthumb
pip install .
```

For local development:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Usage

### Tabular data

Wrap any black-box model's outputs on your training inputs; the wrapper fits
the surrogate and returns per-feature importances.

```python
import numpy as np
from ruleofthumb import RuleOfThumb

X_train = np.random.rand(1000, 4).astype(np.float32)
black_box_probs = (X_train[:, 0] > 0.5).astype(np.int64)  # e.g. model.predict(X_train)

rot = RuleOfThumb(y_outputs=black_box_probs, x_inputs=X_train)
importances = rot.get_explanation(X_train)  # signed, shape [N, d]; positive = evidence toward class 1
```

### Text / token embeddings

Inputs are `(N, tokens, embedding)` float arrays. Padding is explicit: pass a
validity mask (`True` = real token) or per-sample lengths. The wrapper accepts
HuggingFace `attention_mask` tensors directly.

```python
import numpy as np
from ruleofthumb import TextRuleOfThumb
from ruleofthumb.text import pad_sequences

# Ragged inputs? Pad them — any fill value works, the mask carries the truth:
sequences = [np.random.rand(t, 384).astype(np.float32) for t in (20, 14, 17, 9)]
x, lengths = pad_sequences(sequences)                # x: (4, 20, 384)
labels = np.array([1, 0, 1, 0], dtype=np.int64)      # e.g. LLM predictions per text

rot = TextRuleOfThumb(y_outputs=labels, x_inputs=x.numpy(), lengths=lengths)
token_importances = rot.get_explanation(x.numpy(), lengths=lengths)
# signed, shape [N, max_tokens]: positive = evidence toward class 1; padded tokens score exactly 0
# (for n_classes > 2 the output is per-class instead: [N, n_classes, max_tokens])
```

Already have a rectangular batch and your own mask? Pass it directly as
`attention_mask=...` to the constructor and `get_explanation`.

### Images

Inputs are `(N, channels, height, width)` tensors; importance is shared across
spatial locations. Mixed-size batches are supported two ways:

```python
import numpy as np
import torch
from ruleofthumb.image import RoTImage, pad_images

images = [np.random.rand(3, h, w).astype(np.float32) for h, w in [(32, 32), (28, 40)]]
labels = torch.randint(0, 2, (2,))

# Option A: pad into one batch and pass the validity mask
x, mask = pad_images(images)                          # x: (2, 3, 32, 40); mask: (2, 32, 40)
model = RoTImage(classes=2, sample_shape=(3,))
model.fit(x, labels, epochs=50, batch_size=2, lr=0.01, mask=mask)
imp = model.importance(x, mask=mask)                  # padded pixels get zero importance

# Option B: loop over unpadded samples one at a time (weights are size-agnostic,
# so no padding or mask is needed per sample)
for img in images:
    single_imp = model.importance(torch.from_numpy(img[None]))
```

## Quick start notebooks

Hello-world examples on dummy data — no downloads or GPUs needed:

- `examples/01_tabular_quickstart.ipynb`
- `examples/02_text_quickstart.ipynb`
- `examples/03_image_quickstart.ipynb`

## Migrating from v0.1 sentinel padding

v0.2 removed the implicit `-1` sentinel: **no fill value has special meaning
any more**. Padding is now always explicit via a validity mask (`True` = real
token/pixel). Code changes required:

```python
# v0.1 (implicit): pad with -1, the model inferred padding from the data
x[:, n_tokens:] = -1.0
rot = TextRuleOfThumb(y, x)
exp = rot.get_explanation(x)

# v0.2 (explicit): keep any pad value you like, but pass the mask yourself
x[:, n_tokens:] = 0.0                                # any value works now
lengths = torch.tensor([n_tokens] * len(x))          # or a (N, T) boolean mask
rot = TextRuleOfThumb(y, x, lengths=lengths)         # or attention_mask=...
exp = rot.get_explanation(x, lengths=lengths)

# Migrating an existing -1-padded array? Rebuild its mask in one line:
from ruleofthumb.text import sentinel_mask
mask = sentinel_mask(x_old)                          # True where tokens are real
```

Without a mask every position is treated as real data — padded positions are
no longer masked implicitly.

The incremental-reveal pipeline (`get_order` / `ordered_predict` /
`score_ordering`) is mask-aware: padded feature positions are ranked last and
reported as `-1`, and by default reveal curves stop after each sample's real
features are exhausted. By default one reveal step covers a whole **token**
(text) or **pixel** (image) — its embedding dims / channels are revealed
together. Pass `granularity="element"` to `get_order`, `ordered_predict` and
`score_ordering` for the finer per-element curves (the value must match how
the order was produced). Use `include_padded=True` on `ordered_predict` /
`score_ordering` to retain the full rectangular curve including constant
trailing steps. `score_ordering` defaults to per-step accuracy (valid for any
number of classes); pass `return_confusion=True` for per-step K×K confusion
counts (rows = true label, columns = predicted class), or `metric=` for a
custom callable over the binary counts `(tp, fp, fn, tn)`.

## Limitations

- **Rectangular batches.** Inputs are stored as rectangular tensors; use
  `pad_sequences` / `pad_images` plus masks for ragged or mixed-size data.
- Reveal-curve granularity must match between `get_order` and
  `ordered_predict` / `score_ordering` (no auto-detection of the order's
  granularity).
- Custom reveal-curve metrics (`metric=`) are defined over binary counts
  (`tp, fp, fn, tn`); use them only where that view is meaningful. The default
  accuracy metric and `return_confusion=True` work for any number of classes.

See `ToDo.md` for the full list and `tests/test_masks.py` for pinned
behaviour.

## Status

v0.2.1 is a cleaned structural port of the original experiment code with
explicit, generalised padding support; known limitations and planned
follow-ups are tracked in `ToDo.md`.
