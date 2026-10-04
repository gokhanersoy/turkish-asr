"""
Morphology-Aware Tokenizer for Turkish ASR.
Integrates Morfessor morphological segmentation with Subword BPE/WordPiece tokenization
to handle agglutinative word structures gracefully.
"""

from typing import List, Union
from src.utils.text_normalizer import TurkishTextNormalizer


class MorphoTokenizer:
    """
    Turkish Morpho-Tactic Tokenizer.
    Splits agglutinative words into root + suffix morphemes before passing to subword tokenizers.
    """

    def __init__(self, morfessor_model_path: str = None, normalizer: TurkishTextNormalizer = None):
        self.normalizer = normalizer or TurkishTextNormalizer()
        self.morfessor_model = None

        if morfessor_model_path:
            try:
                import morfessor
                io = morfessor.MorfessorIO()
                self.morfessor_model = io.read_binary_model(file_name=morfessor_model_path)
            except ImportError:
                print("Warning: morfessor library not found. Falling back to heuristic morphological segmenter.")

    def segment_word(self, word: str) -> List[str]:
        """
        Segment a single word into morphemes using Morfessor or heuristic rules.
        """
        word = self.normalizer(word)
        if not word:
            return []

        if self.morfessor_model:
            morphemes, _ = self.morfessor_model.viterbi_segment(word)
            return morphemes

        # Heuristic Morpheme Segmentation Fallback for Common Turkish Suffixes
        common_suffixes = [
            "lerimizden", "larımızdan", "lerinden", "larından",
            "lerimiz", "larımız", "lerin", "ların", "lerde", "larda",
            "mizden", "mızdan", "nizden", "nızdan",
            "ler", "lar", "den", "dan", "ten", "tan",
            "de", "da", "te", "ta", "e", "a", "i", "ı", "u", "ü",
            "ci", "cı", "cu", "cü", "li", "lı", "lu", "lü", "siz", "sız", "suz", "süz"
        ]

        for suffix in common_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                root = word[:-len(suffix)]
                return [root, f"+{suffix}"]

        return [word]

    def segment_text(self, text: str) -> str:
        """
        Segment a full Turkish transcript into morpheme-delimited text.
        """
        norm_text = self.normalizer(text)
        words = norm_text.split()
        segmented_words = []

        for word in words:
            morphemes = self.segment_word(word)
            segmented_words.append(" ".join(morphemes))

        return " ".join(segmented_words)

    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(text, str):
            return self.segment_text(text)
        elif isinstance(text, list):
            return [self.segment_text(t) for t in text]
        return text
