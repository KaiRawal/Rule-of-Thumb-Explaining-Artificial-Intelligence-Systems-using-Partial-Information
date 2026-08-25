"""Rule of Thumb: explaining AI systems using partial information."""

from ruleofthumb.core import RoT
from ruleofthumb.explain import Explainer, fit, fit_image, fit_tabular, fit_text
from ruleofthumb.image import pad_images
from ruleofthumb.text import pad_sequences, sentinel_mask

__version__ = "0.2.12"

__all__ = [
    "Explainer",
    "RoT",
    "__version__",
    "fit",
    "fit_image",
    "fit_tabular",
    "fit_text",
    "pad_images",
    "pad_sequences",
    "sentinel_mask",
]
