# ruleofthumb — ToDo

Planned improvements to the pip-installable `ruleofthumb` package, ordered
into execution tiers: quick wins first, then foundational API work that later
features build on, then new functionality.

## API / correctness

## New functionality

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
16. **Optional per-location importance weights.** The text and image variants
    share their `a` / `b` parameters across all positions — shape
    `(K, E)` for text, `(K, C)` for images (`2·K·C` parameters, independent of
    spatial size) — making each surrogate a linear model on a pooled
    representation: the token-mean embedding for text, per-channel spatial
    sums for images. This is faithful to the legacy design and yields
    position-invariant saliency, but caps fidelity at the separability of the
    pooled vector: raw single-channel images pool to total ink mass alone, so
    multiclass fidelity floors out near the majority baseline no matter how
    long the fit runs (measured on digits; see
    `tests/integration/test_image_integration.py`, which pins both the failure
    and the partial recovery from coordinate channels). The same applies to
    text: pooling to the token-mean makes word order invisible, so tasks
    driven by syntax, negation scope or token counts — rather than which words
    appear — hit an analogous ceiling even with high-dimensional embeddings.
    Consider an opt-in unshared mode — image weights `(K, C, H, W)` / text
    weights `(K, T, E)` or per-position biases — turning the surrogate into a
    full linear multiclass model over raw inputs (`2·K·H·W` parameters),
    behind a constructor flag so the legacy behaviour stays the default.

## Release / maintenance

17. CI workflow: GitHub Actions running `pytest` + `ruff check .` on push/PR,
    enforcing the definition-of-done mechanically.
18. PyPI release checklist: build sdist + wheel, twine upload, tag releases
    consistently with the three-place version bump (`pyproject.toml`,
    `src/ruleofthumb/__init__.py`, `tests/test_explain.py` assertion).

## Non-goals

- Bitstring/partial-information sampling scripts are not needed by design:
  RoT operates per token/pixel.

---

## Changelog

- **v0.2.14** — native text ingestion: `fit_text` (and `fit`'s auto-detection)
  accept raw strings directly, embedding them with the bundled
  `answerdotai/ModernBERT-base` default and deriving attention masks
  automatically; callers supply `tokenizer=` / `model=` to override the
  embedder. Explainers fitted from strings accept the same strings back in
  every public method (`get_explanation`, `get_order`, `ordered_predict`,
  `score_ordering`, `score`, `predict`) — each call re-embeds the texts;
  string inputs compose with no explicit padding arguments.
- **v0.2.13** — new `ruleofthumb.embed` module: `embed_texts` tokenises and
  embeds raw strings with a HuggingFace transformer (default
  `answerdotai/ModernBERT-base`, overridable via `tokenizer=` / `model=`),
  returning a frozen `TextEmbeddings` dataclass with rectangular zero-padded
  `(N, tokens, dim)` float32 embeddings, a boolean attention mask ready for
  `fit_text`, and decoded per-sample token strings aligned with the embedding
  rows; batching, `max_length=` truncation and `device=` (auto-detect
  cuda > mps > cpu) supported. Port of the legacy `gen_token_embeddings.py`
  workflow as a library function.
- **v0.2.12** — integration tier expanded and hardened; every case now asserts
  the RoT surrogate's own **predicted-class accuracy** against its black box
  plus explicit feature-importance anchors: breast-cancer explanations track
  the LogisticRegression coefficient profile, COMPAS explanations rank
  `priors_count` top for GBM/SVC/MLP black boxes, wine models agree on shared
  dominant features, text top tokens carry sentiment words (with a
  "brilliant" > "awful" pair check), pet saliency maps reproduce committed
  reference heatmaps and point in the dog direction for GPT-"dog" images, and
  the 10-class image confusion matrix is asserted to collapse near the
  majority baseline (documenting the spatially-shared-weight capacity limit).
  New cases mirror real DS workflows via pandas: the legacy GPT-4o-mini
  cat-vs-dog experiment miniaturized (raw JPEGs + labels committed from
  `ExplanationExampleRemote/DATA` read-only; MobileNetV3-Small features and
  all embeddings recomputed afresh every run — never cached), COMPAS
  two-year recidivism fetched once by the generator with the canonical
  ProPublica filters, and sklearn wine. Typical black boxes added:
  GradientBoosting/SVC(RBF)/MLP per dataset. `pandas` joined the `[dev]`
  extra.
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
