# Website development

Source: `site-src/`. Output (hosted): `website/`.

Live site: `https://kairawal.github.io/Rule-of-Thumb-Explaining-Artificial-Intelligence-Systems-using-Partial-Information/website/`
(GitHub Pages serves `website/` on `main`; root `.nojekyll` preserves `assets/`).

## Edit–build–publish

Edit only under `site-src/`:

- content: `site-src/partials/*.html` (order in `site-src/build.py:PARTIAL_ORDER`)
- table of contents: `site-src/toc.json` (`{{TOC}}` placeholder)
- static files: `site-src/assets/`, `site-src/figures/`, `site-src/favicon.ico`, `site-src/favicon.svg`

Build with the project Python (3.10.0):

```bash
~/software/PythonVenvs/rot_env/bin/python site-src/build.py --preview  # local preview in /tmp/opencode/rot-preview, does not touch website/
~/software/PythonVenvs/rot_env/bin/python site-src/build.py            # rebuild website/index.html + sync static files
~/software/PythonVenvs/rot_env/bin/python site-src/build.py --check    # verify website/ matches site-src/reference/
```

After an intentional content change:

```bash
~/software/PythonVenvs/rot_env/bin/python site-src/build.py --update-reference  # re-snapshot reference, then commit
```

Never edit `website/index.html` directly; `build.py` overwrites it.
Commit `site-src/`, `site-src/reference/index.html`, and `website/`, then push `main` to publish.

## Legacy directories (removed)

- `website_astro/`: frozen Astro build output archived in `572a79f`, superseded by the Python build.
- `website-src/`: dead Astro source (no commits after `4bc0614`); its `AGENTS.md` (`astro dev`) no longer applies.
- `site_new/`: empty untracked scaffold.
- `website-*.zip`, `site-src-*.zip`: ignored 4open.science download artifacts.
