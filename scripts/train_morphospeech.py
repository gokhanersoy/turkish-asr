#!/usr/bin/env python3
"""
CLI Fine-Tuning Script for MorphoSpeech-LLM / SOTA Turkish ASR Models.
Usage:
    python scripts/train_morphospeech.py --model openai/whisper-small --dataset fleurs --epochs 3 --batch-size 8
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path FIRST before importing src modules
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import logging

from src.asr_datasets.loader import TurkishDatasetLoader
from src.training.trainer import MorphoSpeechTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Turkish ASR SOTA Fine-Tuning")
    parser.add_argument(
        "--model", 
        type=str, 
        default="openai/whisper-large-v3", 
        help="Model ID (openai/whisper-large-v3, openai/whisper-small, facebook/wav2vec2-xls-r-300m)"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["common_voice", "fleurs"], 
        default="fleurs", 
        help="Dataset split to train on"
    )
    parser.add_argument(
        "--train-samples", 
        type=int, 
        default=200, 
        help="Number of training samples"
    )
    parser.add_argument(
        "--val-samples", 
        type=int, 
        default=30, 
        help="Number of validation samples"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=4, 
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=1e-4, 
        help="Learning rate"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="checkpoints/morphospeech_sota", 
        help="Directory to save checkpoints"
    )

    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("  TÜRKÇE ASR SOTA EĞİTİMİ (MORPHOSPEECH FINE-TUNING) ")
    logger.info("==================================================")
    logger.info(f"Model        : {args.model}")
    logger.info(f"Veriseti     : {args.dataset}")
    logger.info(f"Eğitim Sayısı: {args.train_samples} ses")
    logger.info(f"Doğrulama    : {args.val_samples} ses")
    logger.info(f"Epoch Sayısı : {args.epochs}")

    loader = TurkishDatasetLoader()
    
    if args.dataset == "common_voice":
        train_ds = loader.load_common_voice(split="train", max_samples=args.train_samples, streaming=True)
        val_ds = loader.load_common_voice(split="test", max_samples=args.val_samples, streaming=True)
        text_col = "sentence"
    elif args.dataset == "fleurs":
        train_ds = loader.load_fleurs(split="train", max_samples=args.train_samples, streaming=True)
        val_ds = loader.load_fleurs(split="test", max_samples=args.val_samples, streaming=True)
        text_col = "transcription"

    trainer = MorphoSpeechTrainer(
        model_name=args.model,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )

    best_wer = trainer.train(train_ds, val_ds, epochs=args.epochs, text_column=text_col)
    
    logger.info("\n================ SOTA EĞİTİMİ TAMAMLANTI ================")
    logger.info(f"En İyi Validation WER: {best_wer:.2f}%")
    logger.info(f"En İyi Checkpoint     : {args.output_dir}/best_sota_model")
    logger.info("=========================================================")

if __name__ == "__main__":
    main()
