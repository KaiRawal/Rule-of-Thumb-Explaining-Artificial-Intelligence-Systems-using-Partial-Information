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

## Mixed-length inputs — preserved by design (do NOT "fix" silently)

Decision (maintainer): ruleofthumb keeps the original rectangular-batch-only
behaviour. The items below are **documented limitations**, pinned by
`tests/test_limitations.py`:

1. **Rectangular batches only, for both text and images.** Ragged token
   sequences or mixed-size image batches are not supported; callers must pad
   to a common shape themselves.
2. **Text padding sentinel is all `-1` embeddings.**
   `RoT_text.importance` masks tokens whose embeddings sum to
   `-embedding_dim`: `mask = (points.sum(dim=2) != -points.shape[2])`. Any
   other fill value is treated as real data (see the characterisation test).
   No `attention_mask` / `pad_value` parameter exists.
3. **Length normalisation is heuristic.** Text score/loss normalisation counts
   non-zero importance rows (`modified_length = T - zero_rows`) rather than
   using a true sequence length.
4. **Image weights are spatially shared and size-agnostic**, so one fitted
   `RoT_image` can score any `(C, H, W)` sample individually — but batches
   must share `H, W`.

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
    `03_gen_embeddings.py`) into `ruleofthumb.embed` ([llm] extra).
16. Port plotting utilities (`viz.py`, `word_clouds.py`, saliency overlays from
    `ExplanationExampleRemote/run.py`) into `ruleofthumb.plot` ([viz] extra).
17. Bitstring/partial-information sampling scripts are *not* needed per design
    decision (RoT operates per token); do not port them.
