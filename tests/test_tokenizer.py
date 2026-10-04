"""
Unit tests for Turkish ASR Normalizer and Morphological Tokenizer.
"""

import os
import unittest
from src.utils.text_normalization import normalize_turkish_text
from src.morpho_tokenizers.turkish_morpho_tokenizer import TurkishMorphoTokenizer

class TestTurkishASR(unittest.TestCase):
    def test_turkish_text_normalization(self):
        # Verify Turkish specific upper/lowercasing (I -> ı, İ -> i)
        text = "İSTANBUL VE İZMİR ŞEHİRLERİ"
        normalized = normalize_turkish_text(text)
        self.assertEqual(normalized, "istanbul ve izmir şehirleri")

    def test_tokenizer_encoding_decoding(self):
        tokenizer = TurkishMorphoTokenizer()
        train_data = [
            "gelmeyebileceklerini biliyoruz",
            "evlerimizden çıktık",
            "arkadaşlarımızla buluştuk"
        ]
        tokenizer.train_morfessor(train_data)
        
        sample = "arkadaşlarımızla evlerimizden çıktık"
        tokens = tokenizer._tokenize(sample)
        reconstructed = tokenizer.convert_tokens_to_string(tokens)
        self.assertEqual(reconstructed, sample)

if __name__ == "__main__":
    unittest.main()
