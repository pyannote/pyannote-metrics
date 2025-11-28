from enum import Enum
from typing import Callable
import warnings

from .english import EnglishTextNormalizer
from .chinese import ChineseNormalizer


class Normalizer(str, Enum):
    NONE = "none"
    FILE = "file"
    ENGLISH = "english"


class NoNormalizer:
    def __call__(self, text: str) -> str:
        return text


BaseNormalizer = Callable[[str], str]


def get_normalizer(file: dict, normalizer: Normalizer) -> BaseNormalizer:
    """Return text normalizer based on `normalizer` and possibly `file`'s language

    Parameters
    ----------
    file : dict
        Audio file metadata. May contain `language` field.
    normalizer : Normalizer
        Normalizer to use.

    Returns
    -------
    normalizer : BaseNormalizer
        Text normalizer
    """

    # no normalization
    if normalizer == Normalizer.NONE:
        return NoNormalizer()

    # per-file normalizer
    if normalizer == Normalizer.FILE:
        language = file.get("language", None)
        if language is None:
            return NoNormalizer()

        elif language == "english":
            warnings.warn(f"Using english text normalizer for file {file['uri']}")
            return EnglishTextNormalizer()
        elif language == "chinese":
            warnings.warn(f"Using chinese text normalizer for file {file['uri']}")
            return ChineseNormalizer()
        else:
            warnings.warn(
                f"No text normalizer available for language '{language}' in file {file['uri']}"
            )
            return NoNormalizer()

    if normalizer == Normalizer.ENGLISH:
        return EnglishTextNormalizer()

    raise ValueError(f"Unsupported normalizer '{normalizer}'")
