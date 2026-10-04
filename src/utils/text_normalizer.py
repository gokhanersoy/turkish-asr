"""
Turkish Text Normalization Utility for ASR Evaluation and Training.
Handles Turkish-specific casing rules (I/ı, İ/i), number-to-text expansion, 
and punctuation standardization.
"""

import re
import unicodedata

# Turkish Casing Mapping
TURKISH_LOWER_MAP = {
    ord("I"): "ı",
    ord("İ"): "i",
}

TURKISH_UPPER_MAP = {
    ord("ı"): "I",
    ord("i"): "İ",
}


def turkish_lowercase(text: str) -> str:
    """Convert text to lowercase preserving Turkish letter mappings (I -> ı, İ -> i)."""
    if not text:
        return ""
    return text.translate(TURKISH_LOWER_MAP).lower()


def turkish_uppercase(text: str) -> str:
    """Convert text to uppercase preserving Turkish letter mappings (ı -> I, i -> İ)."""
    if not text:
        return ""
    return text.translate(TURKISH_UPPER_MAP).upper()


_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]


def _number_to_turkish_text_small(n: int) -> str:
    """Convert number 0-999 to Turkish words."""
    if n == 0:
        return ""
    parts = []
    hundreds = n // 100
    remainder = n % 100
    if hundreds > 1:
        parts.append(_ONES[hundreds] + " yüz")
    elif hundreds == 1:
        parts.append("yüz")
    
    tens = remainder // 10
    ones = remainder % 10
    if tens > 0:
        parts.append(_TENS[tens])
    if ones > 0:
        parts.append(_ONES[ones])
    return " ".join(parts)


def number_to_turkish_text(num_str: str) -> str:
    """Convert digits in a string to spoken Turkish words."""
    try:
        n = int(num_str)
    except ValueError:
        return num_str
    
    if n == 0:
        return "sıfır"
    
    if n < 0:
        return "eksi " + number_to_turkish_text(str(abs(n)))
    
    units = ["", "bin", "milyon", "milyar", "trilyon"]
    parts = []
    unit_idx = 0
    
    while n > 0:
        chunk = n % 1000
        if chunk > 0:
            chunk_str = _number_to_turkish_text_small(chunk)
            if unit_idx == 1 and chunk == 1:
                # In Turkish, 1000 is 'bin', not 'bir bin'
                parts.append("bin")
            else:
                unit_name = units[unit_idx]
                if unit_name:
                    parts.append(f"{chunk_str} {unit_name}")
                else:
                    parts.append(chunk_str)
        n = n // 1000
        unit_idx += 1
        
    parts.reverse()
    return " ".join(parts).strip()


class TurkishTextNormalizer:
    """
    Standardized Text Normalizer for Turkish ASR pipeline.
    Replaces digits with spoken Turkish, normalizes unicode, handles casing and removes punctuation.
    """

    def __init__(self, expand_numbers: bool = True, remove_punct: bool = True):
        self.expand_numbers = expand_numbers
        self.remove_punct = remove_punct
        self.digits_regex = re.compile(r"\d+")

    def __call__(self, text: str) -> str:
        if not text:
            return ""

        # Unicode NFKC Normalization
        text = unicodedata.normalize("NFKC", text)

        # Turkish Lowercasing
        text = turkish_lowercase(text)

        # Expand Numbers to Words
        if self.expand_numbers:
            text = self.digits_regex.sub(lambda m: number_to_turkish_text(m.group(0)), text)

        # Remove Punctuation except Turkish characters and spaces
        if self.remove_punct:
            # Retain Turkish alphabet, numbers, and spaces
            text = re.sub(r"[^\w\s\u00c0-\u024f]", " ", text)
            # Remove underscores
            text = text.replace("_", " ")

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
