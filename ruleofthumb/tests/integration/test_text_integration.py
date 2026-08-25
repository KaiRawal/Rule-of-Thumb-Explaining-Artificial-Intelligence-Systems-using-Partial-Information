"""Integration tests: text modality against a real transformer.

The black box is the cached ``distilbert-base-uncased-finetuned-sst-2-english``
model; its token embeddings and predicted labels are computed live at session
start. Only the RoT explainer is fitted, through the
:func:`ruleofthumb.fit_text` facade.
"""

import numpy as np
import torch

from ruleofthumb import fit_text

SEED = 0

POSITIVE_WORDS = {
    "brilliant", "wonderful", "best", "superb", "delightful", "phenomenal",
    "masterpiece", "gem", "gorgeous", "recommended", "loved", "excellent",
    "charming", "tender", "haunting", "vibrant", "inventive", "riveting", "graceful",
}
NEGATIVE_WORDS = {
    "bored", "boring", "dull", "terrible", "worst", "awful", "waste", "forgettable",
    "disappointment", "avoid", "chore", "mess", "wooden", "lifeless", "mediocre",
    "unfunny", "insults", "cheap", "clumsy", "lazy",
}


def _fit_text(embeddings, attention_mask, y):
    return fit_text(y, embeddings, attention_mask=attention_mask, epochs=200, batch_size=500, learning_rate=0.05, seed=SEED)


def test_black_box_predictions_are_confident(text_sst2):
    assert text_sst2["n_classes"] == 2
    # the fixed reviews must be clearly polarised so sign checks are meaningful
    assert float(text_sst2["confidence"].min()) >= 0.9


def test_explanation_shape_and_padding_zeros(text_sst2):
    embeddings = text_sst2["embeddings"]
    mask = text_sst2["attention_mask"]
    exp = _fit_text(embeddings, mask.numpy(), text_sst2["y"])

    imp = exp.get_explanation(embeddings, attention_mask=mask.numpy())
    n, tokens = mask.shape
    assert imp.shape == (n, tokens)
    assert np.isfinite(imp).all()
    # padded tokens receive exactly zero importance
    assert (imp[~mask.numpy()] == 0).all()
    assert (np.abs(imp) > 0).any()


def test_sentiment_words_carry_signed_importance(text_sst2):
    tokenizer = text_sst2["tokenizer"]
    encoded = text_sst2["encoded"]
    exp = _fit_text(text_sst2["embeddings"], text_sst2["attention_mask"].numpy(), text_sst2["y"])
    imp = exp.get_explanation(text_sst2["embeddings"], attention_mask=text_sst2["attention_mask"].numpy())

    positive_mass, negative_mass = 0.0, 0.0
    for i in range(len(text_sst2["texts"])):
        tokens = encoded.tokens(i)
        word_groups = {}
        for position, word_id in enumerate(encoded.word_ids(i)):
            if word_id is None:  # [CLS] / [SEP] / padding
                continue
            word_groups.setdefault(word_id, []).append(position)
        for positions in word_groups.values():
            word = tokenizer.convert_tokens_to_string([tokens[p] for p in positions]).strip().lower()
            mass = float(imp[i, positions].sum())
            if word in POSITIVE_WORDS:
                positive_mass += mass
            elif word in NEGATIVE_WORDS:
                negative_mass += mass

    # class-1 contributions: positive words push toward "positive", negative away
    assert positive_mass > 0.0
    assert negative_mass < 0.0


def test_reveal_curve_recovers_full_accuracy(text_sst2):
    embeddings, y = text_sst2["embeddings"], text_sst2["y"]
    mask = text_sst2["attention_mask"].numpy()
    exp = _fit_text(embeddings, mask, y)

    xt = torch.from_numpy(embeddings)
    order = exp.get_order(xt, mask=torch.from_numpy(mask))
    # padded positions ranked last and reported as -1
    lengths = text_sst2["attention_mask"].sum(1)
    for i, length in enumerate(lengths):
        assert (order[i, :length] != -1).all() and (order[i, length:] == -1).all()

    curve = exp.score_ordering(xt, torch.from_numpy(y.astype(np.int64)), order)
    full_accuracy = float((exp.predict(xt, mask=torch.from_numpy(mask)).cpu().numpy() == y).mean())
    assert abs(float(curve[-1]) - full_accuracy) < 1e-6
    assert full_accuracy >= 0.85  # surrogate fidelity vs the transformer's labels


def test_seed_reproducibility(text_sst2):
    embeddings, y = text_sst2["embeddings"], text_sst2["y"]
    mask = text_sst2["attention_mask"].numpy()
    imp_a = _fit_text(embeddings, mask, y).get_explanation(embeddings, attention_mask=mask)
    imp_b = _fit_text(embeddings, mask, y).get_explanation(embeddings, attention_mask=mask)
    assert np.allclose(imp_a, imp_b)
