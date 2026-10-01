#!/usr/bin/env python3
"""
CLI Script to Train TurkishMorphoTokenizer on Benchmark Text Datasets (FLEURS / Common Voice).
Usage:
    python scripts/train_tokenizer.py --dataset fleurs --sample-count 500 --output-dir models/turkish_morpho_tokenizer
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.loader import TurkishDatasetLoader
from src.tokenizers.turkish_morpho_tokenizer import TurkishMorphoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train Turkish Morphological Tokenizer")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["common_voice", "fleurs"], 
        default="fleurs", 
        help="Dataset split to extract training text from"
    )
    parser.add_argument(
        "--sample-count", 
        type=int, 
        default=200, 
        help="Number of sentences to use for tokenizer training"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models/turkish_morpho_tokenizer", 
        help="Target directory to save trained tokenizer and vocabulary"
    )

    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("    TÜRKÇE MORFOLOJİK TOKENIZER EĞİTİMİ (MORFESSOR)   ")
    logger.info("==================================================")
    
    loader = TurkishDatasetLoader()
    logger.info(f"Extracting text transcripts from '{args.dataset}' (split=train)...")
    
    if args.dataset == "common_voice":
        dataset = loader.load_common_voice(split="train", max_samples=args.sample_count, streaming=True)
        text_column = "sentence"
    elif args.dataset == "fleurs":
        dataset = loader.load_fleurs(split="train", max_samples=args.sample_count, streaming=True)
        text_column = "transcription"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    sentences = [sample[text_column] for sample in dataset]
    logger.info(f"Extracted {len(sentences)} training sentences.")

    # Initialize and train tokenizer
    tokenizer = TurkishMorphoTokenizer()
    tokenizer.train_morfessor(sentences)

    # Save trained model and vocab
    tokenizer.save_pretrained(args.output_dir)

    logger.info("\n================ MORFOLOJİK BÖLÜMLEME ÖRNEKLERİ ================")
    test_words = [
        "gelmeyebileceklerini",
        "evlerimizden",
        "bilgisayarlaştırabildiklerimizdenmişsinizcesine",
        "arkadaşlarımızla",
        "konuşmalarında",
        "çalıştırılabilen"
    ]

    for word in test_words:
        morphemes = tokenizer._segment_word(word)
        logger.info(f"Kelime  : {word}")
        logger.info(f"Kök+Ek  : {' + '.join(morphemes)}")
        logger.info("-" * 45)

    test_sentence = "arkadaşlarımızla birlikte evlerimizden geldik"
    tokens = tokenizer._tokenize(test_sentence)
    token_ids = [tokenizer._convert_token_to_id(t) for t in tokens]
    reconstructed = tokenizer.convert_tokens_to_string(tokens)

    logger.info("\n================ KODLAMA & ÇÖZME TESTİ ================")
    logger.info(f"Orijinal Cümle : {test_sentence}")
    logger.info(f"Tokenler       : {tokens}")
    logger.info(f"Token ID'leri  : {token_ids}")
    logger.info(f"Geri Çözülen   : {reconstructed}")
    logger.info("==========================================================")

if __name__ == "__main__":
    main()
