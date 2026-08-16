#!/usr/bin/env python3
"""Convert paper figure PDFs into web-friendly SVG/PNG assets in public/figures.

Cropped figures (Fig 1 pet images, Fig 2 resumes) are rendered as high-DPI PNGs
with the exact trim geometry from paper.tex; all others are converted to SVG.
"""
import os
import shutil
import subprocess
import sys

SRC = "/home/vivek/kai-paper"
OUT = "/home/vivek/kai-paper/website/public/figures"
DPI = 200


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


import re

def page_size(path):
    out = subprocess.run(
        ["pdfinfo", path], check=True, capture_output=True, text=True
    ).stdout
    for line in out.splitlines():
        if line.startswith("Page size:"):
            m = re.search(r"([\d.]+)\s*x\s*([\d.]+)", line)
            if m:
                return float(m.group(1)), float(m.group(2))
    raise RuntimeError(f"no page size for {path}")


def crop_png(src, dst, trim):
    """trim = (left, bottom, right, top) in points -> PNG at DPI."""
    pw, ph = page_size(src)
    l, b, r, t = trim
    x = int(round(l * DPI / 72.0))
    y = int(round(t * DPI / 72.0))
    w = int(round((pw - l - r) * DPI / 72.0))
    h = int(round((ph - t - b) * DPI / 72.0))
    run(
        [
            "pdftoppm", "-png", "-r", str(DPI),
            "-x", str(x), "-y", str(y), "-W", str(w), "-H", str(h),
            "-f", "1", "-l", "1", src, os.path.splitext(dst)[0],
        ]
    )
    # pdftoppm appends -1 to filename
    generated = os.path.splitext(dst)[0] + "-1.png"
    shutil.move(generated, dst)


def svg(src, dst, page=1):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    run(["pdftocairo", "-svg", "-f", str(page), "-l", str(page), src, dst])


def main():
    specs = [
        # (dest name, source, mode, trim/page)
        ("fig1/saliency_2_c.png", "Figures/pet_example/saliency_2_c.pdf", "crop", (0, 15, 0, 0)),
        ("fig1/saliency_6_d.png", "Figures/pet_example/saliency_6_d.pdf", "crop", (0, 15, 0, 0)),
        ("fig1/saliency_1450_c.png", "Figures/pet_example/saliency_1450_c.pdf", "crop", (0, 15, 0, 0)),
        ("fig1/saliency_224_c.png", "Figures/pet_example/saliency_224_c.pdf", "crop", (0, 15, 0, 0)),
        ("fig1/saliency_117_d.png", "Figures/pet_example/saliency_117_d.pdf", "crop", (0, 15, 0, 0)),
        ("fig1/saliency_1773_d.png", "Figures/pet_example/saliency_1773_d.pdf", "crop", (0, 15, 0, 0)),
        ("fig2/resume60.png", "figures/resume60.pdf", "crop", (28.3465, 354.3307, 28.3465, 28.3465)),
        ("fig2/resume215.png", "figures/resume215.pdf", "crop", (28.3465, 340.1575, 28.3465, 14.1732)),
        ("fig3/rot_force.svg", "Figures/pima_example/RoT_force_plot-19.pdf", "svg", None),
        ("fig3/shap_force.svg", "Figures/pima_example/SHAP_force_plot-19.pdf", "svg", None),
        ("fig4/rot_abridged.svg", "Figures/indian_cases/RoT351281e5_abridged_377_0_NoPow.pdf", "svg", None),
        ("fig4/shap_abridged.svg", "Figures/indian_cases/SHAP500_abridged_377_0_NoPow.pdf", "svg", None),
        ("fig5/wordcloud.svg", "figures/all_wordcloud_SUM.pdf", "svg", None),
        ("fig6/markup_bar.svg", "Figures/markup/sl-markup_bar.pdf", "svg", None),
        ("fig6/rf_bar.svg", "Figures/markup/sl-rf_bar.pdf", "svg", None),
        ("fig6/rot_bar.svg", "Figures/markup/rot_bar.pdf", "svg", None),
        ("fig7/rot_swarm.svg", "Figures/pima_fairpca/Age-rotswarm.pdf", "svg", None),
        ("fig7/shap_swarm.svg", "Figures/pima_fairpca/Age-shapswarm.pdf", "svg", None),
        ("fig7/lime_swarm.svg", "Figures/pima_fairpca/Age-limeswarm.pdf", "svg", None),
        ("fig8/shap_rot_triangle.svg", "Figures/advattack/triangle_shap_rot.pdf", "svg", None),
        ("fig8/lime_rot_triangle.svg", "Figures/advattack/triangle_lime_rot.pdf", "svg", None),
        ("fig9/unsorted_runtimes.svg", "Figures/speedups/unsorted_runtimes.pdf", "svg", None),
        ("fig9/sorted_runtimes_cumulative.svg", "Figures/speedups/sorted_runtimes_cumulative.pdf", "svg", None),
        ("sfig1/compas_bars.svg", "Figures/openxai/OpenXAI_COMPAS_bars.pdf", "svg", None),
        ("sfig1/compas_scatter.svg", "Figures/openxai/OpenXAI_COMPAS_scatter.pdf", "svg", None),
        ("sfig2/rot_full.svg", "Figures/indian_cases/RoT351281e5_FULL_377_0_NoPow.pdf", "svg", None),
        ("sfig3/shap_full.svg", "Figures/indian_cases/SHAP500_FULL_377_0_NoPow.pdf", "svg", None),
        ("sfig4/ig_full.svg", "Figures/indian_cases/IG_stride_hack_FULL_377_0_NoPow.pdf", "svg", None),
        ("sfig5/lime500_full.svg", "Figures/indian_cases/LIME500_FULL_377_0_NoPow.pdf", "svg", None),
        ("sfig6/lime5000_full.svg", "Figures/indian_cases/LIME5000_FULL_377_0_NoPow.pdf", "svg", None),
        ("sfig7/infrastructure.svg", "Figures/resumes/infrastructure.pdf", "svg", None),
        ("sfig7/network.svg", "Figures/resumes/network.pdf", "svg", None),
        ("sfig7/teaching.svg", "Figures/resumes/teaching.pdf", "svg", None),
        ("sfig7/financial.svg", "Figures/resumes/financial.pdf", "svg", None),
        ("sfig7/construction.svg", "Figures/resumes/construction.pdf", "svg", None),
        ("sfig7/professional.svg", "Figures/resumes/professional.pdf", "svg", None),
        ("sfig8/markup_bar.svg", "Figures/markup/markup_bar.pdf", "svg", None),
        ("sfig8/rf_bar.svg", "Figures/markup/rf_bar.pdf", "svg", None),
        ("sfig8/lr_bar.svg", "Figures/markup/lr_bar.pdf", "svg", None),
        ("sfig8/l1_bar.svg", "Figures/markup/l1-bar.pdf", "svg", None),
        ("sfig8/l2_bar.svg", "Figures/markup/l2-bar.pdf", "svg", None),
        ("sfig8/rot_bar.svg", "Figures/markup/rot_bar.pdf", "svg", None),
        ("sfig9/l_markup_bar.svg", "Figures/markup/l_markup_bar.pdf", "svg", None),
        ("sfig9/l_rf_bar.svg", "Figures/markup/l_rf_bar.pdf", "svg", None),
        ("sfig9/l_lr_bar.svg", "Figures/markup/l_lr_bar.pdf", "svg", None),
        ("sfig9/l_l1_bar.svg", "Figures/markup/l_l1-bar.pdf", "svg", None),
        ("sfig9/l_l2_bar.svg", "Figures/markup/l_l2-bar.pdf", "svg", None),
        ("sfig9/rot_bar.svg", "Figures/markup/rot_bar.pdf", "svg", None),
        ("sfig10/markup_violin.svg", "Figures/markup/markup_violin.pdf", "svg", None),
        ("sfig10/rf_violin.svg", "Figures/markup/rf_violin.pdf", "svg", None),
        ("sfig10/lr_violin.svg", "Figures/markup/lr_violin.pdf", "svg", None),
        ("sfig10/l1_violin.svg", "Figures/markup/l1-violin.pdf", "svg", None),
        ("sfig10/l2_violin.svg", "Figures/markup/l2-violin.pdf", "svg", None),
        ("sfig10/rot_violin.svg", "Figures/markup/rot_violin.pdf", "svg", None),
        ("sfig11/l_markup_violin.svg", "Figures/markup/l_markup_violin.pdf", "svg", None),
        ("sfig11/l_rf_violin.svg", "Figures/markup/l_rf_violin.pdf", "svg", None),
        ("sfig11/l_lr_violin.svg", "Figures/markup/l_lr_violin.pdf", "svg", None),
        ("sfig11/l_l1_violin.svg", "Figures/markup/l_l1-violin.pdf", "svg", None),
        ("sfig11/l_l2_violin.svg", "Figures/markup/l_l2-violin.pdf", "svg", None),
        ("sfig11/rot_violin.svg", "Figures/markup/rot_violin.pdf", "svg", None),
    ]
    for dst, src, mode, trim in specs:
        s = os.path.join(SRC, src)
        d = os.path.join(OUT, dst)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        try:
            if mode == "crop":
                crop_png(s, d, trim)
                print(f"OK {dst} (png)")
            else:
                svg(s, d)
                print(f"OK {dst} (svg)")
        except Exception as e:
            print(f"FAIL {dst}: {e}")


if __name__ == "__main__":
    sys.exit(main())