"""
Unit tests for MorphoSpeech-LLM architecture and TurkishMorphoTokenizer.
Offline mock version for quick unit testing.
"""

import unittest
import torch
import torch.nn as nn
from typing import Optional

from src.tokenizers.turkish_morpho_tokenizer import TurkishMorphoTokenizer
from src.models.morphospeech_llm import MorphoSpeechConfig, MorphoSpeechLLM


class DummyEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(1024, hidden_dim)

    def forward(self, x):
        return self.proj(x)


class TestMorphoSpeechLLM(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TurkishMorphoTokenizer()
        # Train tokenizer on sample Turkish sentences
        sentences = [
            "türkiye cumhuriyeti büyük bir ülkedir",
            "konuşma tanıma sistemleri başarıyla çalışıyor",
            "gelmeyebileceklerini biliyorduk"
        ]
        self.tokenizer.train_morfessor(sentences)

        self.config = MorphoSpeechConfig(
            acoustic_model_name="dummy_model",
            vocab_size=self.tokenizer.vocab_size,
            encoder_dim=128,
            decoder_dim=128,
            freeze_encoder=False
        )
        self.model = MorphoSpeechLLM(self.config, tokenizer=self.tokenizer)
        # Replace acoustic encoder with DummyEncoder for offline test
        self.model.encoder = DummyEncoder(hidden_dim=128)

    def test_tokenizer_segmentation(self):
        text = "gelmeyebileceklerini"
        tokens = self.tokenizer._tokenize(text)
        self.assertTrue(len(tokens) > 0)
        reconstructed = self.tokenizer.convert_tokens_to_string(tokens)
        self.assertEqual(reconstructed.replace(" ", ""), text)

    def test_model_forward_pass(self):
        batch_size = 2
        seq_len = 50
        # Dummy audio features (batch_size, seq_len, in_dim=1024)
        dummy_audio = torch.randn(batch_size, seq_len, 1024)
        dummy_labels = torch.randint(0, self.config.vocab_size, (batch_size, 10))

        outputs = self.model(input_features=dummy_audio, labels=dummy_labels)
        self.assertIn("loss", outputs)
        self.assertIn("ctc_loss", outputs)
        self.assertIn("lm_loss", outputs)
        self.assertTrue(outputs["loss"].item() >= 0)

    def test_generation(self):
        dummy_audio = torch.randn(1, 30, 1024)
        generated_ids = self.model.generate_transcription(dummy_audio, max_length=5)
        self.assertEqual(generated_ids.shape[0], 1)
        self.assertTrue(generated_ids.shape[1] <= 5)


if __name__ == "__main__":
    unittest.main()
