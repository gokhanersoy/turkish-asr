"""
HuggingFace Audio Dataset Loader for Turkish ASR Benchmarks.
Supports Mozilla Common Voice, FLEURS, and custom audio datasets.
Includes streaming support for ultra-fast evaluations without full dataset download.
"""

from typing import Optional, Dict, Any, List
import logging

try:
    from datasets import load_dataset, Audio
except ImportError:
    load_dataset = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TurkishDatasetLoader:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def load_common_voice(
        self,
        version: str = "mozilla-foundation/common_voice_13_0",
        split: str = "test",
        max_samples: Optional[int] = None,
        streaming: bool = True
    ):
        """
        Loads Turkish subset of Mozilla Common Voice dataset.
        """
        if load_dataset is None:
            raise ImportError(
                "HuggingFace `datasets` package is required to load audio benchmarks.")

        logger.info(
            f"Loading Common Voice Turkish dataset ({version}, split={split}, streaming={streaming})...")
        dataset = load_dataset(version, "tr", split=split, streaming=streaming)
        dataset = dataset.cast_column(
            "audio", Audio(sampling_rate=self.sample_rate))

        if streaming:
            if max_samples:
                samples = list(dataset.take(max_samples))
            else:
                samples = list(dataset)
            logger.info(
                f"Successfully streamed {len(samples)} audio samples from Common Voice TR.")
            return samples
        else:
            if max_samples and max_samples < len(dataset):
                dataset = dataset.select(range(max_samples))
            logger.info(
                f"Successfully loaded {len(dataset)} audio samples from Common Voice TR.")
            return dataset

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

        logger.info(
            f"Loading FLEURS Turkish dataset (config='tr_tr', split={split}, streaming={streaming})...")
        dataset = load_dataset("google/fleurs", "tr_tr",
                               split=split, streaming=streaming)
        dataset = dataset.cast_column(
            "audio", Audio(sampling_rate=self.sample_rate))

        if streaming:
            if max_samples:
                samples = list(dataset.take(max_samples))
            else:
                samples = list(dataset)
            logger.info(
                f"Successfully streamed {len(samples)} audio samples from FLEURS TR.")
            return samples
        else:
            if max_samples and max_samples < len(dataset):
                dataset = dataset.select(range(max_samples))
            logger.info(
                f"Successfully loaded {len(dataset)} audio samples from FLEURS TR.")
            return dataset
