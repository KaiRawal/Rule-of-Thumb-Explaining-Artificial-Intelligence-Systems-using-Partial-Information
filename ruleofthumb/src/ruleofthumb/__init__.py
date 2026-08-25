"""Rule of Thumb: explaining AI systems using partial information."""

from ruleofthumb.core import RoT
from ruleofthumb.explain import RuleOfThumb, TextRuleOfThumb
from ruleofthumb.image import pad_images
from ruleofthumb.text import pad_sequences, sentinel_mask

__version__ = "0.2.7"

__all__ = [
    "RoT",
    "RuleOfThumb",
    "TextRuleOfThumb",
    "__version__",
    "pad_images",
    "pad_sequences",
    "sentinel_mask",
]
