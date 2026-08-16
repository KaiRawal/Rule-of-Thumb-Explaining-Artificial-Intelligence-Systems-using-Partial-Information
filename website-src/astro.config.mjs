// @ts-check
import { defineConfig } from 'astro/config';

// https://astro.build/config
export default defineConfig({
  // Served by GitHub Pages from the /website subdirectory of this repo.
  base: '/Rule-of-Thumb-Explaining-Artificial-Intelligence-Systems-using-Partial-Information/website/',
  build: {
    // Rename `_astro` -> `assets` so Jekyll (GitHub Pages, repo-root) keeps
    // the CSS/JS instead of dropping underscore-prefixed directories.
    assets: 'assets',
  },
});
