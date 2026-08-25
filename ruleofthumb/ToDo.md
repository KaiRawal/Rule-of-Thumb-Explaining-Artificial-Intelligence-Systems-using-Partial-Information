# ruleofthumb — ToDo

Planned improvements to the pip-installable `ruleofthumb` package.

## API / correctness

1. Add an `n_classes` parameter (currently `classes=2` hard-coded in the
   explainer wrappers and models).
2. Generalise binary reductions to multiclass: the text wrapper's default
   explanation uses class-1 only, and `score_ordering` computes binary
   confusion counts. Expose reduction options (`abs_sum`, `sum`,
   `class_diff(i, j)`).
3. Move `mins` / `maxs` from class attributes of `RoT` to instance attributes
   set in `__init__`.
4. Expose hard-coded training hyperparameters as arguments: pretrain phase
   (5 epochs), SWA burn-in (`epochs // 10 + 1`), weight decay (`0.01`) and
   the text model's `l1_penalty` default (`0.01`).
5. Unify the tabular/text explainer wrappers behind one facade with an
   explicit reduction parameter; add a matching image wrapper.

## New functionality

6. `ruleofthumb.embed`: embedding-extraction utilities for tokenising and
   embedding text inputs (source: legacy `gen_token_embeddings.py`).
7. **Native text ingestion**: accept raw strings directly in the text
   explainer entry points. Embed via a sensible bundled default HuggingFace
   model, with an override parameter for caller-supplied tokeniser/model;
   derive attention masks / lengths automatically so callers never hand-build
   `(N, tokens, embedding)` arrays. Builds on item 6.
8. **Native image ingestion**: accept image file paths (PNG / JPEG / etc.)
   directly in the image explainer entry points. Decode with Pillow /
   torchvision and apply standard transforms (resize / centre-crop to a
   common size, normalisation), deriving validity masks automatically to fit
   the existing pad-and-mask API.
9. `ruleofthumb.plot`: plotting utilities, including porting existing SHAP
   visualisations to RoT so explanations can be inspected the same way as
   SHAP ones — text visualisations (token highlighting, word clouds), image
   visualisations (saliency-style importance overlays) and rich inline
   rendering for Jupyter notebooks; mirror SHAP's plotting API shapes where
   useful (source: legacy `viz.py`, `word_clouds.py`, `run.py`).
10. Bring back the ability to use non-linear additive functions within the
    RoT framework: per-feature non-linear shape functions combined with the
    linear RoT importance, with configurable sub-model widths and tests
    (source: legacy `rot_class.py`; see also the drafted, unapplied
    OpenXAI integration patch in the repository history).

## Non-goals

- Bitstring/partial-information sampling scripts are not needed by design:
  RoT operates per token/pixel.

---

## Changelog

- **v0.2.1** — reveal curves default to whole units (token per step for text,
  pixel per step for images); `granularity="element"` restores per-element
  curves.
- **v0.2.0** — explicit mask-based padding replaces the implicit `-1`
  sentinel (breaking); ragged text via `pad_sequences`, mixed-size images via
  `pad_images` or per-sample looping.
- **v0.1.x** — initial package port of the original experiment code,
  including dead-code cleanup.
