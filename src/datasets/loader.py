"""
HuggingFace Audio Dataset Loader for Turkish ASR Benchmarks.
Supports Mozilla Common Voice, FLEURS, and custom audio datasets.
Includes streaming support for ultra-fast evaluations without full dataset download.
"""

import os
from typing import Optional, Dict, Any, List
import logging

try:
    from datasets import load_dataset, Audio
except ImportError:
    load_dataset = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TurkishDatasetLoader:
    def __init__(self, sample_rate: int = 16000, hf_token: Optional[str] = None):
        self.sample_rate = sample_rate
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def load_common_voice(
        self,
        version: str = "mozilla-foundation/common_voice_13_0",
        split: str = "test",
        max_samples: Optional[int] = None,
        streaming: bool = True
    ):
        """
        Loads Turkish subset of Mozilla Common Voice dataset.
        Falls back to FLEURS if gated repository error occurs.
        """
        if load_dataset is None:
            raise ImportError(
                "HuggingFace `datasets` package is required to load audio benchmarks.")

        logger.info(
            f"Loading Common Voice Turkish dataset ({version}, split={split}, streaming={streaming})...")

        kwargs = {"split": split, "streaming": streaming, "trust_remote_code": True}
        if self.hf_token:
            kwargs["token"] = self.hf_token

        try:
            dataset = load_dataset(version, "tr", **kwargs)
            dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sample_rate))

            if streaming:
                samples = list(dataset.take(max_samples)) if max_samples else list(dataset)
                logger.info(f"Successfully streamed {len(samples)} audio samples from Common Voice TR.")
                return samples
            else:
                if max_samples and max_samples < len(dataset):
                    dataset = dataset.select(range(max_samples))
                logger.info(f"Successfully loaded {len(dataset)} audio samples from Common Voice TR.")
                return dataset
        except Exception as e:
            logger.warning(f"Failed to load Common Voice dataset ({e}). Falling back to Google FLEURS ('google/fleurs', 'tr_tr')...")
            return self.load_fleurs(split=split, max_samples=max_samples, streaming=streaming)

    def load_fleurs(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
        streaming: bool = True
    ):
        """
        Loads Turkish subset of Google/Meta FLEURS dataset.
        Language config for Turkish in google/fleurs is 'tr_tr'.
        """
        if load_dataset is None:
            raise ImportError(
                "HuggingFace `datasets` package is required to load audio benchmarks.")

        kwargs = {"split": split, "streaming": streaming, "trust_remote_code": True}
        if self.hf_token:
            kwargs["token"] = self.hf_token

        for config in ["tr_tr", "tr_in"]:
            try:
                logger.info(f"Loading FLEURS Turkish dataset (config='{config}', split={split}, streaming={streaming})...")
                dataset = load_dataset("google/fleurs", config, **kwargs)
                dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sample_rate))

                if streaming:
                    samples = list(dataset.take(max_samples)) if max_samples else list(dataset)
                    logger.info(f"Successfully streamed {len(samples)} audio samples from FLEURS TR.")
                    return samples
                else:
                    if max_samples and max_samples < len(dataset):
                        dataset = dataset.select(range(max_samples))
                    logger.info(f"Successfully loaded {len(dataset)} audio samples from FLEURS TR.")
                    return dataset
            except Exception as err:
                logger.warning(f"Failed to load FLEURS config '{config}': {err}")

        raise RuntimeError("Could not load Google FLEURS Turkish dataset.")
