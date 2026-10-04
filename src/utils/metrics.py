"""
Evaluation Metrics (WER & CER) tailored for Turkish ASR.
Utilizes TurkishTextNormalizer prior to metric computation to ensure accurate, reproducible benchmarking.
"""

from typing import Dict, List
from src.utils.text_normalizer import TurkishTextNormalizer

_normalizer = TurkishTextNormalizer()


def compute_wer_cer(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER) after Turkish text normalization.

    Args:
        predictions: List of model hypothesis transcripts.
        references: List of ground-truth transcripts.

    Returns:
        Dict containing 'wer' and 'cer' as percentages (0.0 to 100.0).
    """
    try:
        import jiwer
    except ImportError:
        raise ImportError("jiwer package is required for metric calculation. Please run `pip install jiwer`.")

    norm_preds = [_normalizer(p) for p in predictions]
    norm_refs = [_normalizer(r) for r in references]

    # Filter out empty references to prevent zero division
    valid_pairs = [(p, r) for p, r in zip(norm_preds, norm_refs) if r.strip()]
    if not valid_pairs:
        return {"wer": 0.0, "cer": 0.0}

    filtered_preds, filtered_refs = zip(*valid_pairs)

    wer = jiwer.wer(list(filtered_refs), list(filtered_preds)) * 100.0
    cer = jiwer.cer(list(filtered_refs), list(filtered_preds)) * 100.0

    return {
        "wer": round(wer, 2),
        "cer": round(cer, 2),
    }
