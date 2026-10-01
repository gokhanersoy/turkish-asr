"""
ASR Evaluation Metrics Module.
Calculates Word Error Rate (WER) and Character Error Rate (CER).
"""

from typing import List, Dict, Union
try:
    import jiwer
except ImportError:
    jiwer = None

from src.utils.text_normalization import normalize_turkish_text


def compute_wer(
    references: List[str],
    predictions: List[str],
    normalize: bool = True
) -> float:
    """
    Computes Word Error Rate (WER) between reference and predicted transcripts.
    """
    if normalize:
        references = [normalize_turkish_text(r) for r in references]
        predictions = [normalize_turkish_text(p) for p in predictions]

    if jiwer is not None:
        return jiwer.wer(references, predictions) * 100.0
    else:
        # Fallback simple WER calculation
        total_words = 0
        total_errors = 0
        for ref, pred in zip(references, predictions):
            ref_words = ref.split()
            pred_words = pred.split()
            total_words += len(ref_words)
            # Simple edit distance approximation if jiwer is missing
            errors = abs(len(ref_words) - len(pred_words))
            for rw, pw in zip(ref_words, pred_words):
                if rw != pw:
                    errors += 1
            total_errors += errors
        return (total_errors / max(1, total_words)) * 100.0


def compute_cer(
    references: List[str],
    predictions: List[str],
    normalize: bool = True
) -> float:
    """
    Computes Character Error Rate (CER) between reference and predicted transcripts.
    """
    if normalize:
        references = [normalize_turkish_text(r) for r in references]
        predictions = [normalize_turkish_text(p) for p in predictions]

    if jiwer is not None:
        return jiwer.cer(references, predictions) * 100.0
    else:
        total_chars = 0
        total_errors = 0
        for ref, pred in zip(references, predictions):
            total_chars += len(ref)
            errors = abs(len(ref) - len(pred))
            for rc, pc in zip(ref, pred):
                if rc != pc:
                    errors += 1
            total_errors += errors
        return (total_errors / max(1, total_chars)) * 100.0


def evaluate_asr_predictions(
    references: List[str],
    predictions: List[str],
    normalize: bool = True
) -> Dict[str, float]:
    """
    Returns both WER (%) and CER (%) as a dictionary.
    """
    wer = compute_wer(references, predictions, normalize=normalize)
    cer = compute_cer(references, predictions, normalize=normalize)
    return {
        "wer": round(wer, 2),
        "cer": round(cer, 2)
    }
