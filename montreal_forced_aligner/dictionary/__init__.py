"""
Pronunciation dictionaries
==========================

"""

from montreal_forced_aligner.dictionary.mixins import DictionaryMixin, SanitizeFunction
from montreal_forced_aligner.dictionary.multispeaker import (
    MultispeakerDictionary,
    MultispeakerDictionaryMixin,
)
from montreal_forced_aligner.dictionary.pronunciation import (
    DictionaryData,
    PronunciationDictionary,
    PronunciationDictionaryMixin,
)

__all__ = [
    "pronunciation",
    "multispeaker",
    "mixins",
    "DictionaryData",
    "DictionaryMixin",
    "SanitizeFunction",
    "MultispeakerDictionary",
    "MultispeakerDictionaryMixin",
    "PronunciationDictionary",
    "PronunciationDictionaryMixin",
]
