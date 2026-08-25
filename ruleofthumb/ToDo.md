# ruleofthumb — ToDo

Planned improvements to the pip-installable `ruleofthumb` package, ordered
into execution tiers: quick wins first, then foundational API work that later
features build on, then new functionality.

## API / correctness

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

- **v0.2.10** — unified explainer facade (breaking): the `RuleOfThumb` /
  `TextRuleOfThumb` wrapper classes are replaced by the public `Explainer`
  class plus `fit` / `fit_tabular` / `fit_text` / `fit_image` factories
  (`fit` auto-detects the modality from input ndim). Adds the previously
  missing image wrapper (signed per-pixel explanations, channels summed,
  `(N, H, W)` masks) and public delegating reveal-pipeline methods
  (`get_order`, `ordered_predict`, `score_ordering`, `score`, `predict`);
  raw models are unchanged.
- **v0.2.9** — `device=` parameter on all three RoT models and both explainer
  wrappers (`None` auto-detects cuda > mps > cpu; default cpu otherwise).
  Fit and inference move inputs to the model's device; raw-model methods
  return tensors on the model's device while `get_order` ranks host-side,
  `score_ordering` returns CPU tensors, and wrapper `get_explanation` still
  returns numpy arrays.
- **v0.2.8** — multiclass generalisation: `TextRuleOfThumb.get_explanation`
  follows the tabular semantics (signed class-1 `[N, tokens]` for binary,
  full per-class `[N, n_classes, tokens]` for K > 2); `score_ordering`
  defaults to per-step accuracy for any number of classes and accepts
  `return_confusion=True` for per-step K×K confusion counts (rows = true
  label); custom binary-count `metric=` callables are retained.
- **v0.2.11** — integration-test tier (`tests/integration/`) running against
  real data and models: breast-cancer + LogisticRegression (tabular binary),
  digits + RandomForest (tabular multiclass), fixed film reviews +
  distilbert-SST-2 (text binary), and digit images through committed TinyCNN
  black boxes (image binary and 10-class). All black-box models/datasets are
  built once by `tests/integration/generate_artifacts.py` and committed under
  `tests/integration/artifacts/` (with a provenance `manifest.json`); the test
  suite trains nothing and downloads nothing except HF-cached SST-2 weights.
  New unit tests cover the remaining cells: multiclass text (reveal pipeline,
  per-class padding, mask equivalence, seeding) and binary images via the
  facade with a conv black box. `scikit-learn` moved to the `[dev]` extra —
  it is only needed to (re)generate artifacts, never at runtime.
- **v0.2.7** — tabular `RuleOfThumb.get_explanation` returns signed,
  SHAP-comparable importances: class-1 contributions for binary tasks
  (additive with the class-1 bias), full per-class output for K > 2; unused
  private helpers `_get_exp_abs_sum` / `_get_exp_sum` / `_get_exp_0m1` /
  `_get_exp_1m0` removed.
- **v0.2.6** — the tabular/text explainer wrappers accept `n_classes=` (default
  2, previously hard-coded) and pass it through to the underlying models.
- **v0.2.5** — reproducible seeding: `training_loop` and all three `fit`
  methods accept `seed=` (covering batch shuffling and dropout draws; fits
  seed once up front so the whole fit is one deterministic stream), and the
  tabular/text wrappers thread `seed=` through.
- **v0.2.4** — training hyperparameters are exposed as arguments: `fit` takes
  `pretrain_epochs=` (was hard-coded 5) and `weight_decay=` (was hard-coded
  0.01) on all three variants; `training_loop` takes `swa_burn_in=` (legacy
  default `epochs // 10 + 1`); the tabular/text wrappers thread
  `pretrain_epochs=`, `weight_decay=` and (text) `l1_penalty=` through.
- **v0.2.3** — `get_order` uses a stable descending sort, so units with equal
  importance rank deterministically (earlier index first) instead of
  nondeterministically under quicksort.
- **v0.2.2** — `mins` / `maxs` are now instance attributes set in `RoT.__init__`
  (±inf defaults) instead of class attributes, so unfitted models no longer
  share state across instances.
- **v0.2.1** — reveal curves default to whole units (token per step for text,
  pixel per step for images); `granularity="element"` restores per-element
  curves.
- **v0.2.0** — explicit mask-based padding replaces the implicit `-1`
  sentinel (breaking); ragged text via `pad_sequences`, mixed-size images via
  `pad_images` or per-sample looping.
- **v0.1.x** — initial package port of the original experiment code,
  including dead-code cleanup.
