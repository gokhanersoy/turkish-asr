"""
Training and Evaluation Pipeline for MorphoSpeech-LLM on Turkish Datasets (Common Voice & FLEURS).
"""

import argparse
import os
import shutil
import sys
import torch

# Auto-cleanup legacy folders if present to prevent Hugging Face datasets and tokenizers module shadowing
for legacy_dir in ["src/datasets", "src/tokenizers", "src/tokenizer"]:
    if os.path.exists(legacy_dir):
        shutil.rmtree(legacy_dir, ignore_errors=True)

import evaluate
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperProcessor,
)

from src.models.morpho_speech_llm import MorphoSpeechLLM
from src.data.dataset_loader import TurkishASRDatasetLoader, DataCollatorSpeechSeq2SeqWithPadding
from src.asr_tokenizer.morpho_tokenizer import MorphoTokenizer
from src.utils.metrics import compute_wer_cer
from src.utils.text_normalizer import TurkishTextNormalizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train and Evaluate MorphoSpeech-LLM for Turkish ASR")
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-large-v3", help="Base model identifier")
    parser.add_argument("--dataset_name", type=str, default="mozilla-foundation/common_voice_13_0", help="Dataset name on Hugging Face")
    parser.add_argument("--dataset_config", type=str, default="tr", help="Dataset language config code")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/morpho_speech_llm", help="Directory to save checkpoints")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Train batch size per GPU")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Eval batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for LoRA parameters")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total training epochs")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 mixed precision training")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Enable PEFT/LoRA fine-tuning")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only without training")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit number of eval samples for quick testing")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"==================================================")
    print(f" MorphoSpeech-LLM Turkish ASR Pipeline Launch")
    print(f" Base Model: {args.model_name_or_path}")
    print(f" Dataset: {args.dataset_name} ({args.dataset_config})")
    print(f" CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"==================================================")

    # 1. Initialize Normalizer, Processor & MorphoTokenizer
    normalizer = TurkishTextNormalizer()
    morpho_tokenizer = MorphoTokenizer(normalizer=normalizer)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language="turkish", task="transcribe")

    # 2. Load Dataset
    data_loader = TurkishASRDatasetLoader(
        dataset_name=args.dataset_name,
        language_code=args.dataset_config,
        morpho_tokenizer=morpho_tokenizer,
    )

    eval_dataset = data_loader.load_common_voice(split="test", max_samples=args.max_eval_samples)
    print(f"Loaded {len(eval_dataset)} test samples for evaluation.")

    eval_dataset = eval_dataset.map(
        lambda b: data_loader.prepare_sample(b, processor),
        remove_columns=eval_dataset.column_names,
        num_proc=1,
    )

    train_dataset = None
    if not args.eval_only:
        train_dataset = data_loader.load_common_voice(split="train")
        train_dataset = train_dataset.map(
            lambda b: data_loader.prepare_sample(b, processor),
            remove_columns=train_dataset.column_names,
            num_proc=4,
        )
        print(f"Loaded {len(train_dataset)} train samples.")

    # 3. Model & Data Collator Setup
    model = MorphoSpeechLLM(
        model_name_or_path=args.model_name_or_path,
        use_lora=args.use_lora,
        language="turkish",
        task="transcribe",
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.model.config.decoder_start_token_id,
    )

    # 4. Metric Computation Function
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 in labels with pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        metrics = compute_wer_cer(pred_str, label_str)
        return metrics

    # 5. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        evaluation_strategy="epoch" if train_dataset else "no",
        save_strategy="epoch" if train_dataset else "no",
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True if train_dataset else False,
        metric_for_best_model="wer" if train_dataset else None,
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    if not args.eval_only:
        print("Starting training...")
        trainer.train()
        print("Saving model to output directory...")
        trainer.save_model(args.output_dir)

    print("Running evaluation on test set...")
    eval_metrics = trainer.evaluate()
    print("==================================================")
    print(" Evaluation Results:")
    for key, value in eval_metrics.items():
        print(f"   {key}: {value}")
    print("==================================================")


if __name__ == "__main__":
    main()
