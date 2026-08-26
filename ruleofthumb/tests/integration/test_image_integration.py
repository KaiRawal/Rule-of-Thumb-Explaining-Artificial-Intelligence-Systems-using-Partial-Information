"""Integration tests: image modality, binary and 10-class digit tasks.

Black boxes are TinyCNNs (architecture in ``cnn.py``) trained once by
``generate_artifacts.py`` and committed as state_dicts:

- 10-class digit classifier on the same 500 images as the tabular artifacts;
- binary dense-vs-sparse classifier whose labels split at the median ink mass.

Only the RoT explainer is fitted live, through the
:func:`ruleofthumb.fit_image` facade. Note the image RoT shares importance
across spatial locations, so a sample's score is monotone in total ink mass:
the dense-vs-sparse task is exactly the kind of signal it can express, while
the 10-class task mostly exercises structure (shapes, confusion counts,
ranking), with fidelity asserted against the live majority baseline.
"""

import collections
import os

import numpy as np
import torch
from _helpers import rot_accuracy

from ruleofthumb import fit_image

SEED = 0


def _fit_image(x, y, n_classes):
    return fit_image(y, x, epochs=300, batch_size=5000, learning_rate=0.05, seed=SEED, n_classes=n_classes)


def _pet_paths_and_labels(pets):
    labels = pets["labels"]
    paths = [os.path.join(pets["images_dir"], name) for name in labels["filename"]]
    y = labels["gpt_label"].eq("dog").to_numpy().astype(np.int64)
    return paths, y


def test_native_image_ingestion_end_to_end(pets):
    """Raw JPEG paths + black-box predictions in; per-pixel importances out.

    No ``(N, C, H, W)`` array is ever built by the caller: fit_image decodes
    the files (resize + centre-crop to 64x64), and get_explanation re-loads
    the same paths. The black box splits at the median green-channel mass —
    a signal the spatially-shared surrogate can express (unlike the semantic
    GPT cat/dog labels, which pool to uninformative colour mass; see item 16).
    """
    from ruleofthumb import load_images

    paths, _ = _pet_paths_and_labels(pets)
    loaded = load_images(paths, size=(64, 64))
    green_mass = loaded.images[:, 1].sum(axis=(1, 2))
    y = (green_mass > np.median(green_mass)).astype(np.int64)

    exp = fit_image(y, paths, size=(64, 64), epochs=300, batch_size=5000, learning_rate=0.05, seed=SEED)

    assert exp.modality == "image"
    imp = exp.get_explanation(paths)
    assert imp.shape == (len(paths), 64, 64)
    assert np.isfinite(imp).all()
    assert (np.abs(imp) > 0).any()

    # RoT predicted-class accuracy against the black box's labels
    assert rot_accuracy(exp, loaded.images, y) >= 0.9


def test_native_path_equals_array_path(pets):
    """Fitting from file paths matches the pre-loaded array path exactly."""
    from ruleofthumb import load_images

    paths, y = _pet_paths_and_labels(pets)
    loaded = load_images(paths, size=(64, 64))

    native = fit_image(y, paths, size=(64, 64), epochs=300, batch_size=5000, learning_rate=0.05, seed=SEED)
    arrays = fit_image(y, loaded.images, epochs=300, batch_size=5000, learning_rate=0.05, seed=SEED)

    imp_native = native.get_explanation(paths)
    imp_arrays = arrays.get_explanation(loaded.images)
    assert imp_native.shape == imp_arrays.shape
    assert np.allclose(imp_native, imp_arrays)


def test_black_box_labels_match_committed_cnns(image_multiclass):
    assert image_multiclass["consistent_multi"]
    assert image_multiclass["consistent_binary"]


def test_binary_explanation_shape_and_fidelity(image_multiclass):
    x, y = image_multiclass["x"], image_multiclass["y_binary"]
    exp = _fit_image(x, y, n_classes=2)
    assert exp.model.classes == 2

    imp = exp.get_explanation(x)
    assert imp.shape == (len(x),) + x.shape[2:]
    assert np.isfinite(imp).all()

    predictions = exp.predict(torch.from_numpy(x)).cpu().numpy()
    accuracy = float((predictions == y).mean())
    assert accuracy >= 0.9  # the surrogate can express this signal almost perfectly

    # signed importances are additive with the class-1 bias
    scores = exp.model.score(torch.from_numpy(x)).detach().cpu().numpy()
    bias1 = exp.model.g[1].detach().cpu().numpy()
    assert np.allclose(imp.sum((1, 2)) + bias1, scores[:, 1], atol=1e-3)


def test_binary_reveal_curve_recovers_full_accuracy(image_multiclass):
    x, y = image_multiclass["x"], image_multiclass["y_binary"]
    exp = _fit_image(x, y, n_classes=2)
    xt = torch.from_numpy(x)
    order = exp.get_order(xt)
    assert order.shape == (len(x),) + x.shape[2:]
    curve = exp.score_ordering(xt, torch.from_numpy(y.astype(np.int64)), order)
    full_accuracy = float((exp.predict(xt).cpu().numpy() == y).mean())
    assert abs(float(curve[-1]) - full_accuracy) < 1e-6
    assert curve[-1] >= curve[0] + 0.3


def test_binary_seed_reproducibility(image_multiclass):
    x, y = image_multiclass["x"], image_multiclass["y_binary"]
    imp_a = _fit_image(x, y, n_classes=2).get_explanation(x)
    imp_b = _fit_image(x, y, n_classes=2).get_explanation(x)
    assert np.allclose(imp_a, imp_b)


def test_multiclass_explanation_shape_and_structure(image_multiclass):
    x, y = image_multiclass["x"], image_multiclass["y"]
    exp = _fit_image(x, y, n_classes=10)
    assert exp.modality == "image"
    assert exp.model.classes == 10

    imp = exp.get_explanation(x)
    assert imp.shape == (len(x), 10) + x.shape[2:]
    for k in range(10):
        assert (imp[:, k] != 0).any()

    # fidelity floor: at least as good as always predicting the majority class,
    # measured live so it adapts to any machine/library combination
    predictions = exp.predict(torch.from_numpy(x)).cpu().numpy()
    accuracy = float((predictions == y).mean())
    majority = max(collections.Counter(y.tolist()).values()) / len(y)
    assert accuracy >= majority


def test_ink_pixels_outrank_empty_borders(image_multiclass):
    x = image_multiclass["x"]
    exp_multi = _fit_image(x, image_multiclass["y"], n_classes=10)
    imp_multi = np.abs(exp_multi.get_explanation(x))

    flat_x = x.reshape(len(x), -1)
    ink = flat_x.sum(0) > 0  # pixels that carry signal in any sample
    border = ~ink
    assert ink.any() and border.any()  # digits leave an empty margin
    mean_ink = float(imp_multi.reshape(len(x), 10, -1)[:, :, ink].mean())
    mean_border = float(imp_multi.reshape(len(x), 10, -1)[:, :, border].mean())
    assert mean_ink > mean_border


def test_multiclass_confusion_counts_and_reveal_curve(image_multiclass):
    x, y = image_multiclass["x"], image_multiclass["y"]
    exp = _fit_image(x, y, n_classes=10)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y.astype(np.int64))
    order = exp.get_order(xt)
    assert order.shape == (len(x),) + x.shape[2:]

    confusion = exp.score_ordering(xt, yt, order, return_confusion=True)
    pred = exp.ordered_predict(xt, order).cpu()
    valid = pred != -1
    steps = pred.shape[1]
    assert tuple(confusion.shape) == (steps, 10, 10)
    for j in range(steps):
        assert int(confusion[j].sum()) == int(valid[:, j].sum())

    # explicit confusion-matrix check on the final step: because the image RoT
    # shares weights spatially (score is affine in total ink mass), 10-class
    # fidelity is capped near the majority baseline — the matrix must look bad
    final = confusion[steps - 1]
    accuracy = sum(final[c][c].item() for c in range(10)) / len(x)
    majority = max(collections.Counter(y.tolist()).values()) / len(y)
    assert majority <= accuracy <= majority + 0.05

    column_mass = final.sum(0)
    top2_coverage = float(np.sort(column_mass)[::-1][:2].sum()) / len(x)
    assert top2_coverage >= 0.9  # predictions collapse onto a couple of classes

    curve = exp.score_ordering(xt, yt, order)
    full_accuracy = float((exp.predict(xt).cpu().numpy() == y).mean())
    assert abs(float(curve[-1]) - full_accuracy) < 1e-6


def test_coordinate_channels_restore_multiclass_capacity(image_multiclass):
    """Encoding position as intensity-gated channels lifts the surrogate off the floor.

    The C=1 case pools to total ink alone (majority-baseline ceiling). With
    coordinate channels the pool becomes {mass, sum(row*ink), sum(col*ink)} —
    mass plus center-of-mass — so the same spatially-shared surrogate gains
    real class signal (logistic-regression ceiling on these three numbers is
    ~0.41; full per-pixel linear capacity would be ~0.98).
    """
    x, y = image_multiclass["x_coords"], image_multiclass["y_coords"]
    assert image_multiclass["consistent_coords"]
    exp = _fit_image(x, y, n_classes=10)

    accuracy = rot_accuracy(exp, x, y)
    majority = max(collections.Counter(y.tolist()).values()) / len(y)
    assert accuracy >= max(majority + 0.15, 0.3)  # calibrated floor (measured ~0.35)

    # explicit contrast with the C=1 case on the same digits
    baseline_exp = _fit_image(image_multiclass["x"], image_multiclass["y"], n_classes=10)
    assert accuracy >= rot_accuracy(baseline_exp, image_multiclass["x"], image_multiclass["y"]) + 0.15

    imp = exp.get_explanation(x)
    assert imp.shape == (len(x), 10) + x.shape[2:]

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y.astype(np.int64))
    order = exp.get_order(xt)
    confusion = exp.score_ordering(xt, yt, order, return_confusion=True)
    valid = exp.ordered_predict(xt, order).cpu() != -1
    for j in range(confusion.shape[0]):
        assert int(confusion[j].sum()) == int(valid[:, j].sum())

    curve = exp.score_ordering(xt, yt, order)
    assert abs(float(curve[-1]) - accuracy) < 1e-6

    imp_again = _fit_image(x, y, n_classes=10).get_explanation(x)
    assert np.allclose(imp, imp_again)
