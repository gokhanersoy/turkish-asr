"""
Turkish Text Normalizer for ASR Evaluation.
Handles Turkish-specific casing (I -> ı, İ -> i), punctuation removal, 
and standardized text cleaning for WER/CER calculations.
"""

import re
import string


class TurkishTextNormalizer:
    def __init__(self, remove_punctuation: bool = True, lowercase: bool = True):
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase

    @staticmethod
    def turkish_lowercase(text: str) -> str:
        """
        Custom lowercase function for Turkish characters.
        Standard lower() turns 'I' into 'i', which is wrong for Turkish ('I' -> 'ı').
        """
        rep = {
            'I': 'ı',
            'İ': 'i',
        }
        for k, v in rep.items():
            text = text.replace(k, v)
        return text.lower()

    def normalize(self, text: str) -> str:
        if not text:
            return ""

        # 1. Turkish-specific lowercasing
        if self.lowercase:
            text = self.turkish_lowercase(text)

        # 2. Punctuation removal
        if self.remove_punctuation:
            # Remove punctuation keeping turkish characters (ç, ğ, ı, ö, ş, ü) intact
            text = re.sub(r'[^\w\s]', '', text)

        # 3. Collapse multiple whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text


# Singleton instance for quick import
normalizer = TurkishTextNormalizer()


def normalize_turkish_text(text: str) -> str:
    return normalizer.normalize(text)
