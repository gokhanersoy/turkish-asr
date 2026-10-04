"""
Dataset Loader and Data Collator for Turkish ASR (Common Voice & FLEURS).
Handles audio resampling to 16kHz, text normalization, morpho-tokenization, and batch collation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import torchaudio
from datasets import load_dataset, DatasetDict, Audio
from transformers import AutoFeatureExtractor, AutoTokenizer, AutoProcessor

from src.utils.text_normalizer import TurkishTextNormalizer
from src.tokenizer.morpho_tokenizer import MorphoTokenizer


class TurkishASRDatasetLoader:
    """
    Unified dataset loader for Turkish ASR benchmarks (Common Voice & FLEURS).
    """

    def __init__(
        self,
        dataset_name: str = "mozilla-foundation/common_voice_13_0",
        language_code: str = "tr",
        sampling_rate: int = 16000,
        morpho_tokenizer: MorphoTokenizer = None,
    ):
        self.dataset_name = dataset_name
        self.language_code = language_code
        self.sampling_rate = sampling_rate
        self.normalizer = TurkishTextNormalizer()
        self.morpho_tokenizer = morpho_tokenizer or MorphoTokenizer(normalizer=self.normalizer)

    def load_common_voice(self, split: str = "test", max_samples: int = None) -> Any:
        """Load Mozilla Common Voice Turkish dataset split."""
        dataset = load_dataset(
            self.dataset_name,
            self.language_code,
            split=split,
            trust_remote_code=True,
        )
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
        return dataset

    def load_fleurs(self, split: str = "test", max_samples: int = None) -> Any:
        """Load Google FLEURS Turkish dataset split."""
        dataset = load_dataset(
            "google/fleurs",
            "tr_in",
            split=split,
            trust_remote_code=True,
        )
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
        return dataset

    def prepare_sample(self, batch: Dict[str, Any], processor: AutoProcessor) -> Dict[str, Any]:
        """Preprocess audio array and text transcript for model consumption."""
        audio = batch["audio"]
        text = batch.get("sentence") or batch.get("transcription") or batch.get("text", "")

        # 1. Turkish Normalization
        norm_text = self.normalizer(text)

        # 2. Morpho-tokenization
        morpho_text = self.morpho_tokenizer(norm_text)

        # 3. Audio log-mel feature extraction
        input_features = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # 4. Target text tokenization
        labels = processor.tokenizer(morpho_text).input_ids

        batch["input_features"] = input_features
        batch["labels"] = labels
        batch["normalized_text"] = norm_text
        batch["morpho_text"] = morpho_text
        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that dynamically pads input audio features and target text token labels.
    """

    processor: Any
    decoder_start_token_id: int = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If decoder_start_token_id is present, cut the first token if it matches
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
