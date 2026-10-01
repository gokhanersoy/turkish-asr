#!/usr/bin/env python3
"""
Main CLI Script to Benchmark Turkish ASR Models on Standard Datasets.
Usage:
    python scripts/run_benchmark.py --model openai/whisper-small --dataset fleurs --max-samples 50
"""

from src.models.evaluator import TurkishASREvaluator
from src.datasets.loader import TurkishDatasetLoader
import sys
import os
import argparse
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Turkish ASR Benchmark Runner")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-small",
        help="HuggingFace model ID or local path (e.g., openai/whisper-small, facebook/wav2vec2-xls-r-300m)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["common_voice", "fleurs"],
        default="fleurs",
        help="Dataset to evaluate on (common_voice or fleurs)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (test, validation)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Maximum number of samples to evaluate (for quick test)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation JSON report"
    )

    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("         TÜRKÇE ASR BENCHMARK RUNNER               ")
    logger.info("==================================================")
    logger.info(f"Model: {args.model}")
    logger.info(
        f"Dataset: {args.dataset} (split: {args.split}, max_samples: {args.max_samples})")

    loader = TurkishDatasetLoader()

    if args.dataset == "common_voice":
        dataset = loader.load_common_voice(
            split=args.split, max_samples=args.max_samples)
        text_col = "sentence"
    elif args.dataset == "fleurs":
        dataset = loader.load_fleurs(
            split=args.split, max_samples=args.max_samples)
        text_col = "transcription"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    evaluator = TurkishASREvaluator(model_name_or_path=args.model)
    results = evaluator.evaluate_dataset(dataset, text_column=text_col)

    logger.info("\n================ EVALUATION SUMMARY ================")
    logger.info(f"Model Name        : {results['model_name']}")
    logger.info(f"Sample Count      : {results['sample_count']}")
    logger.info(f"Word Error Rate   : {results['wer_percentage']}% (WER)")
    logger.info(f"Char Error Rate   : {results['cer_percentage']}% (CER)")
    logger.info(f"Avg Latency/Sample: {results['avg_latency_seconds']} s")
    logger.info("====================================================")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    clean_model_name = args.model.replace("/", "_")
    output_file = os.path.join(
        args.output_dir, f"{args.dataset}_{clean_model_name}_results.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
