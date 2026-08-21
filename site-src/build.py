#!/usr/bin/env python3
"""Assemble the static website from modular partials (no Astro).

Usage:
  python3 build.py                 assemble website/index.html (byte-identical to reference) + sync assets/figures/favicons
  python3 build.py --check         verify website/index.html is byte-identical to site-src/reference/index.html
  python3 build.py --update-reference
                                   re-snapshot reference after an intentional content edit
  python3 build.py --preview       write a locally-styled preview to /tmp/opencode/rot-preview (never touches website/)
"""

import argparse
import html
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(ROOT, "..", "website"))
REF = os.path.join(ROOT, "reference", "index.html")
PREVIEW = "/tmp/opencode/rot-preview"

# Order matters: this reproduces the exact byte layout of the reference build.
PARTIAL_ORDER = [
    "_head.html",
    "abstract.html",
    "s1.html",
    "s2.html",
    "s2-1.html",
    "s3.html",
    "s3-1.html",
    "s3-2.html",
    "s3-3.html",
    "s3-4.html",
    "s4.html",
    "s5.html",
    "acknowledgments.html",
    "references.html",
    "supplementary.html",
    "appendix-a.html",
    "appendix-b.html",
    "appendix-c.html",
    "appendix-d.html",
    "appendix-e.html",
    "_footer.html",
]

# Absolute base path baked into the byte-exact output (matches the original Astro build).
BASE = "/Rule-of-Thumb-Explaining-Artificial-Intelligence-Systems-using-Partial-Information/website"
CSS = "index.7CYC1PGc.css"

STATIC_SUBDIRS = ("assets", "figures")
FAVICONS = ("favicon.ico", "favicon.svg")


def render_toc(items):
    lis = []
    for it in items:
        cls = f' class="lvl-{it["lvl"]}"' if it.get("lvl") else ""
        lis.append(f'<li{cls}><a href="#{it["id"]}">{html.escape(it["label"])}</a></li>')
    return '<ul class="toc">' + "".join(lis) + "</ul>"


def assemble():
    toc = json.load(open(os.path.join(ROOT, "toc.json"), encoding="utf-8"))
    toc_html = render_toc(toc)
    parts = []
    for name in PARTIAL_ORDER:
        p = open(os.path.join(ROOT, "partials", name), encoding="utf-8").read()
        parts.append(p.replace("{{TOC}}", toc_html))
    return "".join(parts)


def sync_static(dst_root):
    for sub in STATIC_SUBDIRS:
        src = os.path.join(ROOT, sub)
        dst = os.path.join(dst_root, sub)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    for name in FAVICONS:
        shutil.copy2(os.path.join(ROOT, name), os.path.join(dst_root, name))


def build():
    os.makedirs(OUT, exist_ok=True)
    content = assemble()
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(content)
    sync_static(OUT)
    return content


def preview():
    dst = PREVIEW
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)

    # Rewrite the absolute stylesheet href to a relative one for local preview.
    content = assemble().replace(f"{BASE}/assets/{CSS}", f"assets/{CSS}")
    with open(os.path.join(dst, "index.html"), "w", encoding="utf-8") as f:
        f.write(content)

    # Rewrite the absolute KaTeX font urls inside the CSS to relative ones.
    css_src = os.path.join(ROOT, "assets", CSS)
    css = open(css_src, encoding="utf-8").read().replace(BASE + "/assets/", "assets/")
    os.makedirs(os.path.join(dst, "assets"), exist_ok=True)
    with open(os.path.join(dst, "assets", CSS), "w", encoding="utf-8") as f:
        f.write(css)

    for fname in os.listdir(os.path.join(ROOT, "assets")):
        if fname == CSS:
            continue
        shutil.copy2(os.path.join(ROOT, "assets", fname), os.path.join(dst, "assets", fname))
    shutil.copytree(os.path.join(ROOT, "figures"), os.path.join(dst, "figures"))
    for name in FAVICONS:
        shutil.copy2(os.path.join(ROOT, name), os.path.join(dst, name))

    print(f"Preview written to {dst}")
    print(f"Preview with:  python3 -m http.server 8000 -d {dst}")


def main():
    ap = argparse.ArgumentParser(description="Assemble the static website from modular partials.")
    ap.add_argument("--check", action="store_true", help="verify website/index.html matches the reference build")
    ap.add_argument("--update-reference", action="store_true", help="re-snapshot website/index.html as the reference")
    ap.add_argument("--preview", action="store_true", help="write a locally-styled preview (does not touch website/)")
    args = ap.parse_args()

    if args.preview:
        preview()
        return

    content = build()

    if args.update_reference:
        os.makedirs(os.path.dirname(REF), exist_ok=True)
        shutil.copy2(os.path.join(OUT, "index.html"), REF)
        print("reference updated")

    if args.check:
        if not os.path.exists(REF):
            sys.exit("no reference build found; run --update-reference first")
        ref = open(REF, encoding="utf-8").read()
        if content != ref:
            sys.exit("ERROR: website/index.html differs from the reference build")
        print("OK: website/index.html is byte-identical to the reference build")

    print("website/ updated")


if __name__ == "__main__":
    main()