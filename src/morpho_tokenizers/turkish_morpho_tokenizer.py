"""
Turkish Morphological Tokenizer (TurkishMorphoTokenizer).
Combines Morfessor agglutinative root-suffix segmentation with HuggingFace PreTrainedTokenizer API.
Seamlessly integrates with PyTorch ASR models (Conformer, Wav2Vec2, Whisper, LLMs).
"""

import os
import json
import logging
from typing import List, Dict, Union, Optional, Tuple
import morfessor

from transformers import PreTrainedTokenizer
from src.utils.text_normalization import normalize_turkish_text

logger = logging.getLogger(__name__)

class TurkishMorphoTokenizer(PreTrainedTokenizer):
    """
    HuggingFace-compatible Morphological Tokenizer for Turkish.
    Splits agglutinative words based on Morfessor root + suffix boundaries.
    """
    
    vocab_files_names = {
        "vocab_file": "vocab.json",
        "morfessor_file": "morfessor_model.bin"
    }

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        morfessor_file: Optional[str] = None,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        mask_token="<mask>",
        **kwargs
    ):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token

        # Load Vocab
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.encoder = json.load(f)
        else:
            self.encoder = {
                pad_token: 0,
                unk_token: 1,
                bos_token: 2,
                eos_token: 3,
                mask_token: 4,
            }
            
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Load Morfessor Model
        self.morfessor_model = None
        if morfessor_file and os.path.exists(morfessor_file):
            io_handler = morfessor.MorfessorIO()
            self.morfessor_model = io_handler.read_binary_model_file(morfessor_file)

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            **kwargs
        )

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.encoder)

    def _segment_word(self, word: str) -> List[str]:
        """
        Segments a single Turkish word into root + suffix morphemes.
        """
        if not word:
            return []
        if self.morfessor_model:
            morphemes, _ = self.morfessor_model.viterbi_segment(word)
            return morphemes
        return [word]

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes Turkish text into morpheme tokens.
        """
        text = normalize_turkish_text(text)
        words = text.split()
        tokens = []
        
        for w in words:
            morphemes = self._segment_word(w)
            for idx, m in enumerate(morphemes):
                # Mark continuation morphemes with ## prefix if needed
                token = f"##{m}" if idx > 0 else m
                tokens.append(token)
                
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.encoder.get(self.unk_token, 1))

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Reconstructs original Turkish text from morpheme tokens.
        """
        words = []
        curr_word = ""
        
        for token in tokens:
            if token.startswith("##"):
                curr_word += token[2:]
            else:
                if curr_word:
                    words.append(curr_word)
                curr_word = token
                
        if curr_word:
            words.append(curr_word)
            
        return " ".join(words)

    def train_morfessor(self, sentences: List[str], alpha: float = 1.0, count_modifier=None):
        """
        Trains Morfessor model on Turkish text corpus and builds vocabulary.
        """
        logger.info(f"Training Morfessor model on {len(sentences)} sentences...")
        io_handler = morfessor.MorfessorIO()
        model = morfessor.BaselineModel()
        
        # Word frequency counting
        word_counts = {}
        for s in sentences:
            s_clean = normalize_turkish_text(s)
            for w in s_clean.split():
                word_counts[w] = word_counts.get(w, 0) + 1

        data = [(count, word) for word, count in word_counts.items()]
        model.load_data(data)
        model.train_batch()
        self.morfessor_model = model
        
        # Build vocabulary from segmentation results
        vocab_set = set(self.encoder.keys())
        for word in word_counts.keys():
            morphemes = self._segment_word(word)
            for idx, m in enumerate(morphemes):
                token = f"##{m}" if idx > 0 else m
                vocab_set.add(token)

        # Assign IDs to vocabulary
        self.encoder = {token: idx for idx, token in enumerate(sorted(list(vocab_set)))}
        self.decoder = {idx: token for token, idx in self.encoder.items()}
        logger.info(f"Training complete. Built vocabulary size: {len(self.encoder)}")

    def save_pretrained(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Saves tokenizer vocabulary and Morfessor model files.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        vocab_path = os.path.join(save_directory, "vocab.json")
        morfessor_path = os.path.join(save_directory, "morfessor_model.bin")

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)

        if self.morfessor_model:
            io_handler = morfessor.MorfessorIO()
            io_handler.write_binary_model_file(morfessor_path, self.morfessor_model)

        logger.info(f"Saved TurkishMorphoTokenizer to '{save_directory}'.")
        return (vocab_path, morfessor_path)
