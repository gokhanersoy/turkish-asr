"""
Turkish Text Normalizer for ASR Evaluation.
Handles Turkish-specific casing (I -> ı, İ -> i), punctuation removal, 
and standardized text cleaning for WER/CER calculations.
"""

import re
import string


def turkish_num2words(n: int) -> str:
    """Converts a non-negative integer to its Turkish word representation."""
    if n == 0:
        return "sıfır"
    units = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
    tens = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
    
    if n < 10:
        return units[n]
    if n < 100:
        t, u = divmod(n, 10)
        return (tens[t] + " " + units[u]).strip()
    if n < 1000:
        h, r = divmod(n, 100)
        prefix = "yüz" if h == 1 else f"{units[h]} yüz"
        rem = turkish_num2words(r) if r > 0 else ""
        return (prefix + " " + rem).strip()
    if n < 1000000:
        k, r = divmod(n, 1000)
        prefix = "bin" if k == 1 else f"{turkish_num2words(k)} bin"
        rem = turkish_num2words(r) if r > 0 else ""
        return (prefix + " " + rem).strip()
    return str(n)


class TurkishTextNormalizer:
    def __init__(self, remove_punctuation: bool = True, lowercase: bool = True, convert_numbers: bool = True):
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.convert_numbers = convert_numbers

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

        # 2. Number to words conversion (e.g., '17' -> 'on yedi')
        if self.convert_numbers:
            text = re.sub(r'\b\d+\b', lambda m: turkish_num2words(int(m.group(0))), text)

        # 3. Punctuation removal
        if self.remove_punctuation:
            # Remove punctuation keeping turkish characters (ç, ğ, ı, ö, ş, ü) intact
            text = re.sub(r'[^\w\s]', '', text)

        # 4. Collapse multiple whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text


# Singleton instance for quick import
normalizer = TurkishTextNormalizer()


def normalize_turkish_text(text: str) -> str:
    return normalizer.normalize(text)
