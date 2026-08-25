# AGENTS.md — ruleofthumb package

Instructions for AI coding agents working on the `ruleofthumb` pip package.
Everything in this file is binding: if your planned change conflicts with it,
stop and ask the maintainer.

## Project overview

This repository contains the research codebase for the paper *"Rule of Thumb:
Explaining AI Systems using Partial Information"* plus `ruleofthumb/`, a
pip-installable Python package that consolidates three research variants of
the RoT explainer into one library:

- **Tabular** (`ruleofthumb.explain.RuleOfThumb`) — vector inputs, one
  importance weight per feature.
- **Text / LLM embeddings** (`ruleofthumb.text.RoTText`,
  `ruleofthumb.explain.TextRuleOfThumb`) — token-by-embedding inputs with
  explicit padding masks and length-normalised scores.
- **Images** (`ruleofthumb.image.RoTImage`) — importance shared across spatial
  locations of feature maps.

The package explains a black-box model by fitting a transparent surrogate on
partial observations of its behaviour, producing per-feature importances that
can be revealed incrementally (most-important-first) while preserving the
black box's predictions.

## The legacy research code: inspiration only, NEVER touch it

Everything **outside** `ruleofthumb/` is frozen historical experiment code:
`AdversarialAttack/`, `MovieReviewSentiments/`, `ExplanationExampleLocal/`,
`ExplanationExampleRemote/`, `AIAuditing/`, `ScientificDiscovery/`,
`JudicialCaseOutcomePrediction/`, `SyntheticResumeFiltering/`,
`OpenXAIBenchmark/`, `LitReview/`, `Paper/`, the site directories, and others.

- It exists as **reference and provenance only**. Reading, grepping, or citing
  it is fine — several ToDo porting items point at legacy sources.
- Never edit, move, delete, reformat, rename, or "fix" anything there, no
  matter how broken or outdated it looks. Duplicating messy code verbatim was
  a deliberate phase of this project's history; the package has since moved
  past it, but the archive must remain untouched.
- Success criterion for every session: `git status` shows changes confined to
  `ruleofthumb/`.

## Environment rules

- All development happens in **`ruleofthumb/.venv`** (pre-created: editable
  install of the package + `[dev]` dependencies).
- **Never install into or modify the repo-root `.venv`** — that environment
  belongs to the legacy research code and must stay exactly as it is.
- Run everything from the `ruleofthumb/` directory:
  - `.venv/bin/python -m pytest`
  - `.venv/bin/python -m ruff check .`
  - `.venv/bin/python -m build --sdist`
  - `.venv/bin/jupyter nbconvert --to notebook --execute --inplace examples/<nb>.ipynb`
- This machine runs **lightweight verification only**: the test suite (~1s),
  lint, sdist builds, and executing the small example notebooks. Do not run
  heavy training or experiments here.

## Package architecture

```
ruleofthumb/
├── pyproject.toml          # hatchling; single-install deps; [dev] extra only
├── requirements.txt        # pinned env matching pyproject (convenience)
├── ToDo.md                 # forward-looking package TODOs + changelog footer
├── README.md               # user docs incl. v0.1→v0.2 migration guide
├── src/ruleofthumb/
│   ├── core.py             # RoT base class
│   ├── text.py             # RoTText, pad_sequences, sentinel_mask, lengths_to_mask
│   ├── image.py            # RoTImage, pad_images
│   ├── explain.py          # Explainer facade + fit / fit_tabular / fit_text / fit_image factories
│   └── __init__.py         # exports + __version__
├── tests/                  # pytest suite (test_masks.py pins mask/reveal behaviour)
└── examples/               # hello-world quickstart notebooks (dummy data)
```

Key mechanics:

- `RoT.training_loop` runs an SGD loop with SWA averaging (SWA model stored via
  `object.__setattr__` to avoid submodule registration — required for
  torch >= 2.x).
- The incremental-reveal pipeline (`get_order` → `ordered_predict` →
  `score_ordering`) simulates revealing inputs most-important-first and scores
  prediction fidelity along the curve. Reveal units are tokens (text) /
  pixels (images) / features (tabular) by default.

## Design decisions (binding)

These were deliberate choices made during the v0.1 → v0.2.x evolution. Do not
regress them:

1. **Masks are first-class and explicit** (v0.2.0, breaking change). Padding
   is *never* inferred from data values; `-1` has no special meaning anywhere.
   Convention: masks are boolean validity tensors (`True` = real token /
   pixel). Utilities return `(padded_batch, lengths_or_mask)`.
   `ruleofthumb.text.sentinel_mask` exists solely to migrate legacy `-1`
   padded arrays.
2. **Unit-granularity reveal curves by default** (v0.2.1): one reveal step per
   token (text) or pixel (image), aggregating embedding dims / channels via
   abs-sum. `granularity="element"` restores per-feature-element curves; the
   granularity used by `ordered_predict` / `score_ordering` must match how
   the order was produced (no auto-detection). Tabular is unaffected by the
   setting.
3. **PEP8 class names without aliases**: `RoTImage`, `RoTText`. Breaking API
   changes are acceptable pre-1.0, but each one needs a README migration note.
4. **Single install**: all runtime dependencies live in base `dependencies`
   (plotting, transformers, vision included); only `[dev]` exists as an
   extra. License is SPDX `license = "MIT"`.
5. **Dead code is deleted**, not kept "for port fidelity". Removals are
   recorded in the ToDo changelog footer instead.
6. **Known deliberate limitations** stay until their ToDo item lands:
   `classes=2` hard-coded, binary-only reduction metrics, class-level
   `mins`/`maxs`, hard-coded training hyperparameters (5 pretrain epochs,
   SWA burn-in `epochs // 10 + 1`, weight decay `0.01`, `l1_penalty 0.01`).

## ToDo.md conventions

`ToDo.md` lists **forward-looking improvements to the pip package only** —
never fixes for the legacy research code. Structure:

- *API / correctness* — refactors and generalisations of existing APIs.
- *New functionality* — new modules and capabilities (e.g. native ingestion
  of raw strings / image file paths, plotting, additive shape functions).
  Terse legacy source pointers like `(source: legacy rot_class.py)` are
  allowed and encouraged.
- *Non-goals* — explicit design decisions against features (do not re-litigate
  them silently).
- *Changelog footer* — completed work summarised per version; once an item
  ships it moves from the todo list to here.

When behaviour changes, update README (usage + migration notes) and bump the
version.

## Definition of done

Before finishing any change:

- [ ] `pytest` green, `ruff check .` clean (run inside `ruleofthumb/.venv`).
- [ ] `python -m build --sdist` succeeds if packaging changed.
- [ ] Touched example notebooks still execute end-to-end without errors.
- [ ] Version bumped in all three places when behaviour changed:
      `pyproject.toml`, `src/ruleofthumb/__init__.py`, and the assertion in
      `tests/test_explain.py`.
- [ ] Untracked caches cleaned up: `__pycache__/`, `.pytest_cache/`,
      `.ruff_cache/` (all gitignored, but keep the tree tidy).
- [ ] `git status` shows modifications confined to `ruleofthumb/`; the legacy
      research code and root `.venv` are byte-for-byte untouched.
