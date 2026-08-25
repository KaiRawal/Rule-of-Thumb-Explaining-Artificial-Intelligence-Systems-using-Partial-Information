# ruleofthumb — ToDo

Planned improvements to the pip-installable `ruleofthumb` package, ordered
into execution tiers: quick wins first, then foundational API work that later
features build on, then new functionality.

## API / correctness

1. Move `mins` / `maxs` from class attributes of `RoT` to instance attributes
   set in `__init__`. Unfitted models currently share the class-level ±inf
   defaults across instances.
2. Use stable tie-breaking in `get_order`: `np.argsort` currently uses
   unstable quicksort, so equal-importance units rank nondeterministically.
3. Expose hard-coded training hyperparameters as arguments: pretrain phase
   (5 epochs), SWA burn-in (`epochs // 10 + 1`), weight decay (`0.01`) and
   the text model's `l1_penalty` default (`0.01`).
4. Add reproducible seeding: a `seed=` parameter threaded through
   `training_loop` / `fit`, covering batch shuffling (`torch.randperm`),
   dropout masks and weight initialisation. Fits are currently
   irreproducible and reveal-curve results can vary between runs.
5. Add an `n_classes` parameter (currently `classes=2` hard-coded in the
   explainer wrappers and models).
6. Generalise binary reductions to multiclass (builds on item 5): the text
   wrapper's default explanation uses class-1 only, and `score_ordering`
   computes binary confusion counts. Expose reduction options (`abs_sum`,
   `sum`, `class_diff(i, j)`).
7. Add a `device=` parameter (CPU default, CUDA optional): tensors are
   currently created without an explicit device, so fitting cannot use a GPU.
8. Unify the tabular/text explainer wrappers behind one facade with an
   explicit reduction parameter; add the missing image wrapper. This is the
   prerequisite for the native-ingestion and plotting items below — build
   them on the facade, not the legacy-shaped wrappers.

## New functionality

9. `ruleofthumb.embed`: embedding-extraction utilities for tokenising and
    embedding text inputs (source: legacy `gen_token_embeddings.py`).
10. **Native text ingestion** (builds on items 8–9): accept raw strings
    directly in the text explainer entry points. Embed via a sensible bundled
    default HuggingFace model, with an override parameter for caller-supplied
    tokeniser/model; derive attention masks / lengths automatically so callers
    never hand-build `(N, tokens, embedding)` arrays.
11. **Native image ingestion** (builds on item 8): accept image file paths
    (PNG / JPEG / etc.) directly in the image explainer entry points. Decode
    with Pillow / torchvision and apply standard transforms (resize /
    centre-crop to a common size, normalisation), deriving validity masks
    automatically to fit the existing pad-and-mask API.
12. Save/load fitted explainers: persistence helpers (state_dict round-trip)
    so fitted models survive process boundaries without refitting.
13. **Automatic training** (builds on items 3–4): Keras-tuner-style
    hyperparameter autotuning — a validation-split-driven search (random or
    grid) over the exposed training hyperparameters (`learning_rate`,
    `batch_size`, `epochs`, `dropout_rate`, weight decay, SWA burn-in) that
    fits candidate configurations, scores them on held-out data via the
    reveal-fidelity metric, and returns the best fitted explainer. Requires
    item 3 (hyperparameters exposed) and pairs with item 4 (seeding) so
    candidate comparisons are fair.
14. `ruleofthumb.plot`: plotting utilities (builds on item 8), including
    porting existing SHAP visualisations to RoT so explanations can be
    inspected the same way as SHAP ones — text visualisations (token
    highlighting, word clouds), image visualisations (saliency-style
    importance overlays) and rich inline rendering for Jupyter notebooks;
    mirror SHAP's plotting API shapes where useful (source: legacy `viz.py`,
    `word_clouds.py`, `run.py`).
15. Bring back the ability to use non-linear additive functions within the
    RoT framework: per-feature non-linear shape functions combined with the
    linear RoT importance, with configurable sub-model widths and tests
    (source: legacy `rot_class.py`; see also the drafted, unapplied
    OpenXAI integration patch in the repository history).

## Release / maintenance

16. CI workflow: GitHub Actions running `pytest` + `ruff check .` on push/PR,
    enforcing the definition-of-done mechanically.
17. PyPI release checklist: build sdist + wheel, twine upload, tag releases
    consistently with the three-place version bump (`pyproject.toml`,
    `src/ruleofthumb/__init__.py`, `tests/test_explain.py` assertion).

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
