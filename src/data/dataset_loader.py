"""
Dataset Loader and Data Collator for Turkish ASR (Google FLEURS & Mozilla Common Voice).
Handles audio resampling to 16kHz, text normalization, morpho-tokenization, and batch collation.
Uses Google FLEURS Turkish ('google/fleurs', 'tr_tr') as the primary open benchmark dataset.
"""

from dataclasses import dataclass
import os
import logging
from typing import Any, Dict, List, Union
import torch
import torchaudio

try:
    from datasets import load_dataset, Audio, DatasetDict
except ImportError:
    load_dataset = None

from src.utils.text_normalizer import TurkishTextNormalizer
from src.asr_tokenizer.morpho_tokenizer import MorphoTokenizer

logger = logging.getLogger(__name__)


class TurkishASRDatasetLoader:
    """
    Unified dataset loader for Turkish ASR benchmarks.
    Defaults to Google FLEURS Turkish ('google/fleurs', 'tr_tr') as primary open dataset.
    """

    def __init__(
        self,
        dataset_name: str = "google/fleurs",
        language_code: str = "tr_tr",
        sampling_rate: int = 16000,
        morpho_tokenizer: MorphoTokenizer = None,
        hf_token: str = None,
    ):
        self.dataset_name = dataset_name
        self.language_code = language_code
        self.sampling_rate = sampling_rate
        self.normalizer = TurkishTextNormalizer()
        self.morpho_tokenizer = morpho_tokenizer or MorphoTokenizer(normalizer=self.normalizer)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def load_dataset_split(self, split: str = "test", max_samples: int = None, streaming: bool = False) -> Any:
        """
        Load dataset split for Turkish ASR (defaults to Google FLEURS 'tr_tr').
        """
        if load_dataset is None:
            raise ImportError("HuggingFace `datasets` package is required.")

        kwargs = {"split": split}
        if streaming:
            kwargs["streaming"] = True
        if self.hf_token:
            kwargs["token"] = self.hf_token

        try:
            logger.info(f"Loading {self.dataset_name} ({self.language_code}, split={split})...")
            dataset = load_dataset(self.dataset_name, self.language_code, **kwargs)
            dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
            if not streaming and max_samples and max_samples < len(dataset):
                dataset = dataset.select(range(max_samples))
            return dataset
        except Exception as e:
            logger.warning(f"Could not load {self.dataset_name} ({e}). Attempting fallback to Google FLEURS ('google/fleurs', 'tr_tr')...")
            return self.load_fleurs(split=split, max_samples=max_samples, streaming=streaming)

    def load_common_voice(self, split: str = "test", max_samples: int = None, streaming: bool = False) -> Any:
        """Compatibility wrapper calling load_dataset_split."""
        return self.load_dataset_split(split=split, max_samples=max_samples, streaming=streaming)

    def load_fleurs(self, split: str = "test", max_samples: int = None, streaming: bool = False) -> Any:
        """Load Google FLEURS Turkish dataset split."""
        if load_dataset is None:
            raise ImportError("HuggingFace `datasets` package is required.")

        kwargs = {"split": split}
        if streaming:
            kwargs["streaming"] = True
        if self.hf_token:
            kwargs["token"] = self.hf_token

        for config in ["tr_tr", "tr_in"]:
            try:
                logger.info(f"Loading FLEURS Turkish dataset ('google/fleurs', config='{config}', split='{split}')...")
                dataset = load_dataset("google/fleurs", config, **kwargs)
                dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
                if not streaming and max_samples and max_samples < len(dataset):
                    dataset = dataset.select(range(max_samples))
                return dataset
            except Exception as err:
                logger.warning(f"Failed to load FLEURS config '{config}': {err}")

        raise RuntimeError("Unable to load Google FLEURS Turkish dataset.")

    def prepare_sample(self, batch: Dict[str, Any], processor: Any) -> Dict[str, Any]:
        """Preprocess audio array and text transcript for model consumption."""
        audio = batch["audio"]
        text = batch.get("transcription") or batch.get("raw_transcription") or batch.get("sentence") or batch.get("text") or ""

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
        if self.decoder_start_token_id is not None and (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
