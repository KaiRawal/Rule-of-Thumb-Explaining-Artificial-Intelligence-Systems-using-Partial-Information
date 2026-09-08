# Publishing `ruleofthumb` v0.0.1 to PyPI (+ docs to Pages)

> ⚠️ Experimental 0.0.x pre-alpha — entirely vibe-coded from hand-written
> research code. May break; backwards-incompatible changes expected.

This runbook takes the branch from "ready" to "public". Nothing here
publishes automatically — every upload/tag/push is a manual command you
run. Do not commit from automation; review `git status` first (changes
must stay inside `ruleofthumb/` until the planned split).

## 0. Prerequisites

- PyPI account + API token (`Account Settings → API tokens`). Store as
  `$PYPI_TOKEN`; for the dry run, a TestPyPI token as `$TEST_PYPI_TOKEN`.
- `uv` installed (this branch uses the `uv_build` backend; `uv build` is
  the only supported build command).
- Clean tree on branch `pip-package`:
  `git status --short` shows only `ruleofthumb/` paths.
- Version pinned in all three places: `pyproject.toml`,
  `src/ruleofthumb/__init__.py`, `tests/test_explain.py` (all `0.0.1`).

## 1. Final verification (from `ruleofthumb/`, venv only)

```bash
cd ruleofthumb
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
uv build
tar tzf dist/ruleofthumb-0.0.1.tar.gz | head -n 20  # LICENSE + README present
pip install dist/ruleofthumb-0.0.1-py3-none-any.whl --force-reinstall
python -c "import ruleofthumb; print(ruleofthumb.__version__)"
```

Expected: pytest green, ruff clean, `dist/` holds exactly one sdist +
one wheel, import prints `0.0.1`. Delete and rebuild `dist/` if stale
artifacts linger (`rm -rf dist/ && uv build`).

Docs build (one command, venv only — stages the `examples/` notebooks as
generated copies under `docs/notebooks/`, re-executes them, validates links):

```bash
.venv/bin/pip install -e ".[docs]" && mkdir -p docs/notebooks && cp examples/0*.ipynb docs/notebooks/ && .venv/bin/mkdocs build --strict
```

## 2. Dry run: TestPyPI

```bash
cd ruleofthumb
uv publish --publish-url https://test.pypi.org/legacy/ --token "$TEST_PYPI_TOKEN"
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ruleofthumb==0.0.1
python -c "import ruleofthumb; print(ruleofthumb.__version__)"
```

Check the TestPyPI project page renders: title, README (with the
experimental banner), MIT licence, Homepage/Repository/Issues/
Documentation links.

## 3. Publish: PyPI

```bash
cd ruleofthumb
uv publish --token "$PYPI_TOKEN"
```

Then confirm on `https://pypi.org/p/ruleofthumb/`:

- Version `0.0.1`, licence MIT, Python `>=3.9`.
- README banner visible; `project.urls` all resolve (Documentation is a
  placeholder Pages URL until the split — see §5).

## 4. Tag and push (manual)

```bash
git status --short   # confirm: only ruleofthumb/ files
git add ruleofthumb
git commit -m "feat: first public pre-alpha v0.0.1 (uv_build, docs, PyPI metadata)"
git tag ruleofthumb-v0.0.1
git push origin pip-package --tags
```

Open the PR / release notes from the tag, pasting the `ToDo.md`
`v0.0.1` changelog entry.

## 5. Docs to GitHub Pages (manual, one time)

No CI workflows in this branch (deferred with ToDo item 17) — deploy by
hand from `ruleofthumb/`. The workflow template at
`docs/pages-workflow-template.yml` is parked, not wired up.

```bash
cd ruleofthumb
# 1. One-command site build (notebooks staged + executed, links validated):
.venv/bin/pip install -e ".[docs]" && mkdir -p docs/notebooks && cp examples/0*.ipynb docs/notebooks/ && .venv/bin/mkdocs build --strict
# 2. Sanity-check the output locally:
.venv/bin/mkdocs serve   # browse http://127.0.0.1:8000 — Home, Guides,
                         # executed Notebooks, Migration, Capacity, API, Test report
# 3. Publish the contents of site/ to Pages. Easiest path:
.venv/bin/mkdocs gh-deploy   # pushes site/ to the gh-pages branch
```

Then enable Pages if needed (`Settings → Pages → Deploy from branch →
gh-pages`), and confirm the site URL matches `[project.urls]
Documentation` in `pyproject.toml` (currently a placeholder Pages URL —
if it differs, that means a follow-up release to correct the metadata).

Notes:

- Expected build noise (safe to ignore): `[IPKernelApp] WARNING | Kernel
  is running over TCP...` lines from notebook execution, and the red
  "Material for MkDocs" banner about MkDocs 2.0 (upstream notice, not a
  build warning). `mkdocs build --strict` still fails on any real warning
  — a green build ends with `Documentation built`.
- `docs/notebooks/` and `site/` are gitignored generated copies — never
  commit them. The canonical notebooks live in `examples/`.
- Re-run the one-command build before each release so the executed
  notebooks and the included `tests/TEST_SUITE.md` report stay fresh.
- After the planned split, the same commands run from the standalone repo
  root; `mkdocs.yml:site_url` and `[project.urls] Documentation` carry
  `TODO(split)` markers to update then.

## 6. After the release

- Yank/re-publish only if the wheel is broken (`yank`, never reuse
  `0.0.1` — bump to `0.0.2`).
- Next version: bump all three version sites, add a `ToDo.md` changelog
  entry newest-first, rebuild docs.
- Planned split: `git subtree split -P ruleofthumb -b rot-split`; the new
  repo root already contains `LICENSE`, `.gitignore`, `pyproject.toml`,
  `README.md`, `docs/`, `mkdocs.yml`, `PUBLISH.md`.

## 7. If something fails

| Symptom | Fix |
|---|---|
| `uv build` errors on backend | Check `[build-system]` = `uv_build>=0.12.10,<0.13` / `uv_build`; run with `RUST_LOG=uv=debug uv build`. |
| Sdist missing `LICENSE`/`README` | Check `license-files = ["LICENSE*"]` and `readme = "README.md"`; `LICENSE` must sit next to `pyproject.toml`. |
| `uv publish` 403 | Token scope (project vs account), or name taken — confirm `pypi.org/p/ruleofthumb/`. |
| `mkdocs build --strict` warnings as errors | Fix the flagged link/docstring, don't drop `--strict`. |
| Heavy install (`torch`, `transformers`, `shap`) surprises users | Known single-install decision (AGENTS.md); point to README install notes. |
