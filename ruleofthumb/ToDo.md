# ToDo — known hard-coded limitations (v1 port, not yet fixed)

The v1 package is a faithful port of the original experiment code. The items
below were identified during the migration audit and are intentionally **not**
fixed yet so behaviour matches the published experiments.

## Deliberate deviations from the original source (torch >= 2.x compatibility)

These are not semantic changes, but note them for reproducibility:

- **SWA registration bypass** (`core.RoT._set_swa_model`): the original
  `self.swa_model = AveragedModel(self)` silently registered the SWA copy as a
  submodule of the model (doubling its parameter count). Old torch tolerated
  this; torch >= 2.x raises in `update_parameters`. We now store the SWA model
  via `object.__setattr__` so it is not registered as a submodule.
- **`project()` uses `torch.clamp`**: equivalent to the original nested
  `torch.min(torch.max(...))`, but also accepts the scalar class defaults
  (`mins=-inf, maxs=inf`), which modern torch rejects as `max()` operands.

## v1 cleanup deviations (2026-08)

Simplifications applied on top of the faithful port; behaviour of the kept
APIs is unchanged:

- **`ruleofthumb.models` removed.** The module (`Linear_regression`,
  `per_point_NAM/RBF/poly`, `RoT_additive`, `rand_order`) was dead code: no
  experiment ever instantiated it and nothing in the package consumed it (see
  the additive-models entry below).
- **PEP8 class renames, no aliases:** `RoT_image` → `RoTImage`,
  `RoT_text` → `RoTText`. Breaking change.
- **Dead parameters removed:** the never-used `scheduler` argument of
  `RoT.training_loop`; `RoT_text.continue_fit` / `use_sgd`.
- **Dead code removed:** a stray debug `print()` in `RoT_image.fit_project`,
  commented-out blocks and an unused `response_mean` computation in
  `RoT_text.loss`, and an unused `SWALR` import in `core.py`.

## Mixed-length inputs — FIXED in v0.2.0 (2026-08)

The items below were preserved as documented limitations in v0.1 and are now
**fixed** by the mask-first redesign (breaking change; see README
"Migrating from v0.1 sentinel padding"). Pinned by `tests/test_masks.py`:

1. ~~Rectangular batches only, for both text and images.~~ Ragged text is
   handled by `ruleofthumb.text.pad_sequences` + validity masks; mixed-size
   images by `ruleofthumb.image.pad_images` + masks, or per-sample looping.
2. ~~Text padding sentinel is all `-1` embeddings.~~ Sentinel inference was
   removed entirely: masks are explicit (`mask` / `attention_mask` / `lengths`
   parameters) and no fill value has special meaning.
   `ruleofthumb.text.sentinel_mask` migrates legacy `-1`-padded arrays.
3. ~~Length normalisation is heuristic.~~ Text scores now normalise by true
   token counts taken from the mask/lengths.
4. Image weights remain spatially shared and size-agnostic: one fitted model
   scores any `(C, H, W)` sample individually (no padding needed), and padded
   batches are supported via validity masks.

Note: `get_order` / `ordered_predict` / `score_ordering` are mask-aware;
reveal steps operate at feature-element granularity (tokens x embedding dims
for text, channels x pixels for images), matching v0.1 semantics.

## Other known hard-coded limitations (not yet fixed)

## Class-count / binary assumptions

4. **`classes=2` hard-coded** in every `RuleOfThumb` wrapper
   (`AdversarialAttack/rule_of_thumb.py`, `MovieReviewSentiments/Code/rule_of_thumb.py`).
   Add an `n_classes` parameter.
5. **Binary reductions.** The text wrapper's default explanation uses
   `imp[:, 1, :, :]` (class-1 only); `score_ordering` computes binary
   confusion counts (`pred == 0/1`). Generalise to multiclass and expose
   reduction options (`abs_sum`, `sum`, `class_diff(i, j)`).

## Training-loop magic numbers

6. Pretrain phase fixed at **5 epochs** inside `RoT.fit`.
7. SWA burn-in hard-coded as `epochs // 10 + 1`.
8. Weight decay values scattered: `0.01` (tabular/text), `0.002` (text variant
   in `MovieReviewSentiments/Code/rot_class.py`).
9. `l1_penalty` default `0.01` in `RoT_text`.
10. Sub-model widths: NAM `width=128`, RBF `width=32`, polynomial `width=6`.

## State / correctness

11. **Class-level mutable state**: `mins=-np.inf, maxs=np.inf` are class
    attributes of `RoT`; move to instance attributes set in `__init__`.
12. Verbose epoch-printing ladder in `MovieReviewSentiments/Code/rot_class.py`
    training loop (up to `%128`) — replace with a logging level.
13. `fit(write_file=...)` / `continue_fit(append_file=...)` debug file writes in
    `ExplanationExampleRemote/rot_class.py` — remove or route through `logging`.

## API / packaging follow-ups

14. Unify the three `RuleOfThumb` wrappers behind one facade with explicit
    reduction parameter (currently duplicated in `explain.py` as
    `RuleOfThumb` + `TextRuleOfThumb`).
15. Port embedding-extraction utilities (`gen_token_embeddings.py`,
    `03_gen_embeddings.py`) into `ruleofthumb.embed`.
16. Port plotting utilities (`viz.py`, `word_clouds.py`, saliency overlays from
    `ExplanationExampleRemote/run.py`) into `ruleofthumb.plot`.
17. Bitstring/partial-information sampling scripts are *not* needed per design
    decision (RoT operates per token); do not port them.
18. **Bring back the ability to use non-linear additive functions within the
    RoT framework.** This was the original thesis behind code removed in the
    v1 cleanup: `RoT_additive` combined per-feature non-linear shape functions
    (`per_point_NAM`, `per_point_RBF`, `per_point_poly`) with the linear RoT
    importance, with helpers `Linear_regression` and `rand_order`. It was
    never exercised by any published experiment (the only usage is a drafted,
    unapplied patch in `OpenXAIBenchmark/CODE/patch.diff`). If the skipped
    experiments are revived, port these into `ruleofthumb.models` with tests,
    starting from any legacy copy, e.g. `AdversarialAttack/rot_class.py`
    (lines ~200–268).
