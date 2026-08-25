"""Shared fixtures for the integration-test tier.

Every fixture loads committed artifacts from ``tests/integration/artifacts/``
(regenerate with ``tests/integration/generate_artifacts.py``). Nothing is
trained or downloaded at test time except the HuggingFace SST-2 weights,
which load from the local HF cache when present.
"""

import os

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
import joblib
from cnn import TinyCNN

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

SST2_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def _require(name):
    path = os.path.join(ARTIFACTS, name)
    if not os.path.exists(path):
        raise RuntimeError(f"missing artifact {path}; regenerate with tests/integration/generate_artifacts.py")
    return path


@pytest.fixture(scope="session")
def tabular_binary():
    """Breast-cancer features + LogisticRegression black box."""
    data = np.load(_require("tabular_binary.npz"))
    lr = joblib.load(_require("breast_cancer_lr.joblib"))["logistic_regression"]
    return {"x": data["x"], "y": data["y"], "coef": lr.coef_[0]}


@pytest.fixture(scope="session")
def tabular_multiclass():
    """Flattened digit images + RandomForest black box (10 classes)."""
    data = np.load(_require("digits_tabular.npz"))
    rf = joblib.load(_require("digits_rf.joblib"))["random_forest"]
    return {"x": data["x"], "y": data["y"], "forest": rf}


@pytest.fixture(scope="session")
def image_multiclass():
    """Digit images ``(N, 1, 8, 8)`` + TinyCNN black boxes (2 and 10 classes)."""
    data = np.load(_require("digits_image.npz"))
    state = torch.load(_require("cnn_multiclass.pt"), map_location="cpu", weights_only=True)
    cnn_multi = TinyCNN(n_classes=10)
    cnn_multi.load_state_dict(state)
    cnn_multi.eval()
    state = torch.load(_require("cnn_binary.pt"), map_location="cpu", weights_only=True)
    cnn_binary = TinyCNN(n_classes=2)
    cnn_binary.load_state_dict(state)
    cnn_binary.eval()
    x = data["x_multi"]
    with torch.no_grad():
        consistent_multi = bool((cnn_multi(torch.from_numpy(x)).argmax(1).numpy() == data["y_multi"]).all())
        consistent_binary = bool((cnn_binary(torch.from_numpy(x)).argmax(1).numpy() == data["y_binary"]).all())
    return {
        "x": x,
        "y": data["y_multi"],
        "y_binary": data["y_binary"],
        "cnn_multi": cnn_multi,
        "cnn_binary": cnn_binary,
        "consistent_multi": consistent_multi,
        "consistent_binary": consistent_binary,
    }


@pytest.fixture(scope="session")
def text_sst2():
    """Fixed film reviews encoded with the cached SST-2 transformer."""
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(SST2_NAME)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(SST2_NAME)
    model.eval()

    with open(_require("reviews.txt")) as f:
        texts = [line.strip() for line in f if line.strip()]
    encoded = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=48)
    attention_mask = encoded["attention_mask"].to(torch.bool)
    with torch.no_grad():
        out = model(**encoded, output_hidden_states=True)
    logits = out.logits
    probabilities = torch.softmax(logits, dim=-1)
    embeddings = out.hidden_states[-1].numpy().astype(np.float32)
    return {
        "tokenizer": tokenizer,
        "encoded": encoded,
        "texts": texts,
        "embeddings": embeddings,
        "attention_mask": attention_mask,
        "y": logits.argmax(-1).numpy().astype(np.int64),
        "confidence": probabilities.max(-1).values,
        "n_classes": logits.shape[-1],
    }
