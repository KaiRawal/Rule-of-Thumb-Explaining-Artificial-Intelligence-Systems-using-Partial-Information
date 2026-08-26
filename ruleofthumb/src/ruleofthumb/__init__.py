"""Rule of Thumb: explaining AI systems using partial information."""

from ruleofthumb.core import RoT
from ruleofthumb.embed import DEFAULT_TEXT_MODEL, TextEmbeddings, embed_texts
from ruleofthumb.explain import Explainer, fit, fit_image, fit_tabular, fit_text
from ruleofthumb.image import pad_images
from ruleofthumb.text import pad_sequences, sentinel_mask

__version__ = "0.2.14"

__all__ = [
    "DEFAULT_TEXT_MODEL",
    "Explainer",
    "RoT",
    "TextEmbeddings",
    "__version__",
    "embed_texts",
    "fit",
    "fit_image",
    "fit_tabular",
    "fit_text",
    "pad_images",
    "pad_sequences",
    "sentinel_mask",
]
