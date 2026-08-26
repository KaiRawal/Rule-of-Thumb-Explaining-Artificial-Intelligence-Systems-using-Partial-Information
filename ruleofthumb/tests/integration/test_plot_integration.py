"""Integration tests: visualisations on real committed artifacts.

Every plot family renders from genuine explanations (breast-cancer LR
profile, SST-2 token importances, pet saliency reference heatmaps) and
exports to PNG bytes.
"""

import io

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ruleofthumb import embed_texts, fit_tabular, fit_text, load_images, plot

SEED = 0


def _png_bytes(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def test_tabular_plots_render_and_export(tabular_binary):
    x = tabular_binary["x"]
    explainer = fit_tabular(tabular_binary["y"], x, epochs=200, batch_size=500, learning_rate=0.05, seed=SEED)
    names = [f"feature_{i}" for i in range(x.shape[1])]

    for name in ("waterfall", "force", "decision"):
        png = _png_bytes(getattr(plot, name)(explainer, x[:1], feature_names=names))
        assert len(png) > 1000
    for name in ("bar", "beeswarm"):
        png = _png_bytes(getattr(plot, name)(explainer, x[:50], feature_names=names))
        assert len(png) > 1000


def test_text_html_highlights_real_tokens(text_sst2):
    texts = text_sst2["texts"]
    embedded = embed_texts(texts)
    mask = embedded.attention_mask
    explainer = fit_text(text_sst2["y"], embedded.embeddings, attention_mask=mask, epochs=200, batch_size=500, learning_rate=0.05, seed=SEED)

    imp = explainer.get_explanation(embedded.embeddings, attention_mask=mask)[0]
    tokens = embedded.tokens[0]
    html = getattr(plot.text_html(imp, tokens), "data", "")

    top_tokens = [tokens[i] for i in np.argsort(-np.abs(imp))[:5] if tokens[i]]
    assert all(token in html for token in top_tokens)


def test_saliency_renders_reference_heatmaps_over_pet_images(pets):
    """The committed reference heatmaps overlay cleanly onto the raw JPEGs."""
    labels = pets["labels"]
    paths = [f"{pets['images_dir']}/{name}" for name in labels["filename"]]
    images = load_images(paths, size=(224, 224))

    for i in range(min(3, len(paths))):
        fig = plot.saliency(pets["reference"][i].astype(np.float32), image=images.images[i].transpose(1, 2, 0))
        assert len(_png_bytes(fig)) > 1000


def test_word_clouds_render_review_tokens(text_sst2):
    texts = text_sst2["texts"]
    embedded = embed_texts(texts)
    mask = embedded.attention_mask
    explainer = fit_text(text_sst2["y"], embedded.embeddings, attention_mask=mask, epochs=200, batch_size=500, learning_rate=0.05, seed=SEED)

    rows, token_lists = [], []
    for i in range(len(texts)):
        imp = explainer.get_explanation(embedded.embeddings, attention_mask=mask)[i]
        rows.append(imp)
        token_lists.append([t for t, real in zip(embedded.tokens[i], mask[i]) if real])

    fig = plot.word_clouds(rows, token_lists, seed=SEED)
    assert len(_png_bytes(fig)) > 1000
