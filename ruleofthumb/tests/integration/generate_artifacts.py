"""One-off generator for the committed integration-test artifacts.

Run from ``ruleofthumb/`` (requires the ``[dev]`` extra for scikit-learn):

    .venv/bin/python tests/integration/generate_artifacts.py

Trains every black-box model exactly once with fixed seeds and writes
everything to ``tests/integration/artifacts/``. The test suite never trains a
model and never downloads a dataset; it only loads these files. The script is
deterministic and rerunnable: identical outputs given identical library
versions (see ``manifest.json``).

Artifacts produced:
- ``tabular_binary.npz`` + ``breast_cancer_lr.joblib`` — scaled breast-cancer
  features, LogisticRegression predictions as black-box labels.
- ``digits_tabular.npz`` + ``digits_rf.joblib`` — 500 flattened digit images,
  RandomForest predictions as black-box labels.
- ``digits_image.npz`` — binary (0 vs 1) and 10-class subsets of digit images
  shaped ``(N, 1, 8, 8)``; the 10-class labels come from a trained TinyCNN,
  the binary labels from a dense-vs-sparse TinyCNN on the same images.
- ``cnn_multiclass.pt`` / ``cnn_binary.pt`` — state_dicts of the TinyCNNs
  used as image black boxes (architecture in ``tests/integration/cnn.py``).
- ``reviews.txt`` — one fixed film-review snippet per line.
- ``manifest.json`` — library versions + per-file sha256/shape provenance.
"""

import hashlib
import json
import os
import sys

import joblib
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn import TinyCNN

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def record(manifest, name, arrays=None):
    path = os.path.join(ARTIFACTS, name)
    entry = {"sha256": sha256(path)}
    if arrays is not None:
        entry["shapes"] = {k: list(v.shape) for k, v in arrays.items()}
    manifest["files"][name] = entry


def make_tabular_binary(manifest):
    data = load_breast_cancer()
    scaler = StandardScaler()
    x = scaler.fit_transform(data.data).astype(np.float32)
    lr = LogisticRegression(max_iter=5000, random_state=0, solver="liblinear")
    lr.fit(x, data.target)
    y = lr.predict(x).astype(np.int64)

    np.savez_compressed(os.path.join(ARTIFACTS, "tabular_binary.npz"), x=x, y=y)
    joblib.dump({"scaler": scaler, "logistic_regression": lr}, os.path.join(ARTIFACTS, "breast_cancer_lr.joblib"))
    record(manifest, "tabular_binary.npz", {"x": x, "y": y})
    record(manifest, "breast_cancer_lr.joblib")
    print(f"tabular binary: x{x.shape}, logistic-regression train accuracy {lr.score(x, data.target):.3f}")


def _digit_subset(per_class):
    digits = load_digits()
    x = digits.data / 16.0  # scale to [0, 1]
    rng = np.random.RandomState(0)
    keep = []
    for c in range(10):
        idx = np.flatnonzero(digits.target == c)
        keep.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return x[keep], digits.target[keep]


def make_digits_tabular(manifest, x, y_true):
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(x, y_true)
    y = rf.predict(x).astype(np.int64)

    np.savez_compressed(os.path.join(ARTIFACTS, "digits_tabular.npz"), x=x.astype(np.float32), y=y)
    joblib.dump({"random_forest": rf}, os.path.join(ARTIFACTS, "digits_rf.joblib"))
    record(manifest, "digits_tabular.npz", {"x": x, "y": y})
    record(manifest, "digits_rf.joblib")
    print(f"tabular multiclass: x{x.shape}, forest oob-style train accuracy {rf.score(x, y_true):.3f}")


def _train_tiny_cnn(x, y, n_classes, epochs=40, batch=64):
    torch.manual_seed(0)
    cnn = TinyCNN(n_classes=n_classes)
    optimiser = torch.optim.AdamW(cnn.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.int64)
    for _ in range(epochs):
        cnn.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch):
            sel = perm[i : i + batch]
            optimiser.zero_grad()
            loss_fn(cnn(xt[sel]), yt[sel]).backward()
            optimiser.step()
    cnn.eval()
    return cnn


def make_digits_image(manifest, per_class_binary=60):
    # binary subset (0 vs 1), kept for robustness/future items even though the
    # current suite exercises binary images at unit level
    digits = load_digits()
    images = (digits.data / 16.0).reshape(-1, 1, 8, 8).astype(np.float32)
    rng = np.random.RandomState(1)
    bin_idx = []
    for c in (0, 1):
        idx = np.flatnonzero(digits.target == c)
        bin_idx.append(rng.choice(idx, size=min(per_class_binary, len(idx)), replace=False))
    bin_idx = np.concatenate(bin_idx)
    rng.shuffle(bin_idx)

    # multiclass subset: same selection rule as the tabular artifacts so both
    # views describe the same 500 digits
    xm, ym_true = _digit_subset(50)
    xm = xm.astype(np.float32).reshape(-1, 1, 8, 8)
    xt = torch.from_numpy(xm)

    cnn_multi = _train_tiny_cnn(xm, ym_true, n_classes=10)
    with torch.no_grad():
        y_multi = cnn_multi(xt).argmax(1).numpy().astype(np.int64)
    multi_accuracy = float((y_multi == ym_true).mean())

    # binary dense-vs-sparse black box on the same images: labels split at the
    # median ink mass, the exact signal a pixel-shared surrogate can express
    ink = xm.reshape(len(xm), -1).sum(1)
    dense_sparse = (ink >= np.median(ink)).astype(np.int64)
    cnn_binary = _train_tiny_cnn(xm, dense_sparse, n_classes=2)
    with torch.no_grad():
        y_binary = cnn_binary(xt).argmax(1).numpy().astype(np.int64)
    binary_accuracy = float((y_binary == dense_sparse).mean())

    torch.save(cnn_multi.state_dict(), os.path.join(ARTIFACTS, "cnn_multiclass.pt"))
    torch.save(cnn_binary.state_dict(), os.path.join(ARTIFACTS, "cnn_binary.pt"))
    np.savez_compressed(
        os.path.join(ARTIFACTS, "digits_image.npz"),
        x_bin=images[bin_idx],
        y_bin=digits.target[bin_idx].astype(np.int64),
        x_multi=xm,
        y_multi=y_multi,
        y_binary=y_binary,
    )
    record(manifest, "cnn_multiclass.pt")
    record(manifest, "cnn_binary.pt")
    record(
        manifest,
        "digits_image.npz",
        {"x_bin": images[bin_idx], "x_multi": xm, "y_multi": y_multi, "y_binary": y_binary},
    )
    print(f"image multiclass: x{xm.shape}, tiny-cnn accuracy {multi_accuracy:.3f}")
    print(f"image binary (dense vs sparse): tiny-cnn accuracy {binary_accuracy:.3f}")


REVIEWS = [
    "A brilliant film with wonderful performances and a sharp, moving script.",
    "Absolutely loved it. The direction is confident and the ending is devastating in the best way.",
    "One of the best films of the year: funny, humane and beautifully shot.",
    "A charming, warm-hearted movie that earns every laugh and every tear.",
    "Superb acting and a tight, gripping story from start to finish.",
    "Delightful from the first frame to the last. I could watch it again immediately.",
    "The cast is phenomenal and the soundtrack perfectly matches the mood.",
    "A masterpiece of quiet storytelling with stunning cinematography.",
    "Clever, exciting and full of heart. Highly recommended.",
    "An instant classic. The screenplay crackles and nothing is wasted.",
    "I was bored senseless. The plot drags and the characters are paper-thin.",
    "A dull, predictable mess that wastes its talented cast.",
    "Terrible pacing, wooden dialogue and an ending that insults the audience.",
    "One of the worst films I have seen this year. Painfully unfunny.",
    "The story is incoherent and the special effects look cheap.",
    "Utterly forgettable. I checked my watch more often than I followed the plot.",
    "A clumsy, overlong slog with no likeable characters to root for.",
    "Awful writing and worse editing. Avoid at all costs.",
    "The film tries hard but collapses into a confusing, joyless muddle.",
    "Shallow, noisy and exhausting. A huge disappointment.",
    "What a gorgeous, tender piece of cinema. Every scene sings.",
    "Mediocre at best. It has a couple of good moments buried in tedium.",
    "Fresh, original and surprisingly emotional. A real gem.",
    "A riveting thriller with excellent performances across the board.",
    "Lifeless and derivative. You have seen everything here done better elsewhere.",
    "Charming and bittersweet, with a lead performance worth remembering.",
    "Crude humor and a lazy script make this a chore to sit through.",
    "Vibrant, inventive filmmaking that rewards a second viewing.",
    "The worst kind of sequel: cynical, loud and completely unnecessary.",
    "Graceful, haunting and wonderfully acted. Don't miss it.",
]


def write_reviews(manifest):
    path = os.path.join(ARTIFACTS, "reviews.txt")
    with open(path, "w") as f:
        f.write("\n".join(REVIEWS) + "\n")
    record(manifest, "reviews.txt")
    print(f"reviews: {len(REVIEWS)} fixed snippets")


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    manifest = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sklearn": __import__("sklearn").__version__,
        "files": {},
    }
    make_tabular_binary(manifest)
    x_tab, y_tab = _digit_subset(50)
    make_digits_tabular(manifest, x_tab, y_tab)
    make_digits_image(manifest)
    write_reviews(manifest)

    path = os.path.join(ARTIFACTS, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"manifest written to {path}")


if __name__ == "__main__":
    main()
