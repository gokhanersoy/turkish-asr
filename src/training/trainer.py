"""
MorphoSpeech Fine-Tuning & Training Pipeline for Turkish ASR.
Supports PyTorch + HuggingFace Transformers + PEFT/LoRA.
Trains models to surpass SOTA on Common Voice TR and FLEURS TR.
"""

import os
import logging
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets

from torch.optim import AdamW
from transformers import (
    AutoProcessor, 
    AutoModelForSpeechSeq2Seq, 
    AutoModelForCTC,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType

from src.utils.text_normalization import normalize_turkish_text
from src.utils.metrics import compute_wer, compute_cer

logger = logging.getLogger(__name__)

class MorphoSpeechTrainer:
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        output_dir: str = "checkpoints/morphospeech_sota",
        use_lora: bool = True,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 2,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_lora = use_lora
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.grad_accum_steps = gradient_accumulation_steps

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing MorphoSpeechTrainer for '{model_name}' on device '{self.device}'...")
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        if "whisper" in model_name.lower():
            self.is_encoder_decoder = True
            if hasattr(self.processor, "tokenizer"):
                self.processor.tokenizer.language = "turkish"
                self.processor.tokenizer.task = "transcribe"
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
            self.model.config.forced_decoder_ids = None
            self.model.config.suppress_tokens = []
        else:
            self.is_encoder_decoder = False
            self.model = AutoModelForCTC.from_pretrained(model_name)

        if self.use_lora:
            logger.info("Applying LoRA (Low-Rank Adaptation) for efficient fine-tuning...")
            if self.is_encoder_decoder:
                target_modules = ["q_proj", "v_proj"]
            else:
                target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
                
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM if self.is_encoder_decoder else TaskType.FEATURE_EXTRACTION
            )
            try:
                self.model = get_peft_model(self.model, peft_config)
                self.model.print_trainable_parameters()
            except Exception as e:
                logger.warning(f"PEFT/LoRA initialization warning ({e}). Falling back to memory-efficient encoder-frozen fine-tuning...")
                if hasattr(self.model, "model") and hasattr(self.model.model, "encoder"):
                    for param in self.model.model.encoder.parameters():
                        param.requires_grad = False
                    logger.info("Encoder layers frozen for memory efficiency.")

        self.model.to(self.device)

    def train(
        self, 
        train_dataset, 
        val_dataset, 
        epochs: int = 3, 
        text_column: str = "transcription"
    ):
        """
        Executes fine-tuning loop and saves best SOTA checkpoint.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        
        total_steps = (len(train_dataset) // (self.batch_size * self.grad_accum_steps)) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_steps * 0.1), 
            num_training_steps=max(1, total_steps)
        )

        best_wer = 999.0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            optimizer.zero_grad()

            logger.info(f"\n--- Epoch {epoch}/{epochs} ---")
            
            # Access underlying model if PEFT-wrapped to prevent keyword argument collisions
            target_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

            for idx, sample in enumerate(tqdm(train_dataset, desc=f"Training Epoch {epoch}")):
                audio_arr = sample["audio"]["array"]
                sr = sample["audio"].get("sampling_rate", 16000)
                target_text = sample[text_column]

                # Prepare inputs
                inputs = self.processor(audio_arr, sampling_rate=sr, return_tensors="pt").to(self.device)
                
                if self.is_encoder_decoder:
                    input_features = inputs.input_features.to(dtype=target_model.dtype)
                    full_tokens = self.processor.tokenizer(text=target_text, return_tensors="pt").input_ids.to(self.device)
                    # Align decoder_input_ids and labels perfectly to prevent position shift / EOS hallucination
                    decoder_input_ids = full_tokens[:, :-1]
                    labels = full_tokens[:, 1:].clone()
                    labels[labels == self.processor.tokenizer.pad_token_id] = -100
                    outputs = target_model(
                        input_features=input_features, 
                        decoder_input_ids=decoder_input_ids,
                        labels=labels
                    )
                else:
                    labels = self.processor.tokenizer(text=target_text, return_tensors="pt").input_ids.to(self.device)
                    outputs = target_model(input_values=inputs.input_values, labels=labels)

                loss = outputs.loss / self.grad_accum_steps
                loss.backward()
                total_loss += loss.item() * self.grad_accum_steps

                if (idx + 1) % self.grad_accum_steps == 0 or (idx + 1) == len(train_dataset):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_train_loss = total_loss / max(1, len(train_dataset))
            logger.info(f"Epoch {epoch} Average Loss: {avg_train_loss:.4f}")

            # Validation Evaluation
            val_wer = self.evaluate_validation(val_dataset, text_column=text_column)
            logger.info(f"Epoch {epoch} Validation WER: {val_wer:.2f}%")

            if val_wer < best_wer or epoch == epochs:
                best_wer = min(best_wer, val_wer)
                best_dir = os.path.join(self.output_dir, "best_sota_model")
                
                save_model = self.model
                if hasattr(self.model, "merge_and_unload"):
                    try:
                        save_model = self.model.merge_and_unload()
                    except Exception as e:
                        logger.warning(f"Could not merge LoRA weights before saving: {e}")
                        save_model = self.model
                        
                save_model.save_pretrained(best_dir)
                self.processor.save_pretrained(best_dir)
                logger.info(f"🏆 CHECKPOINT SAVED TO: {best_dir} (WER: {val_wer:.2f}%)")

        return best_wer

    def evaluate_validation(self, val_dataset, text_column: str = "transcription") -> float:
        self.model.eval()
        refs, preds = [], []
        target_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

        with torch.no_grad():
            for sample in val_dataset:
                audio_arr = sample["audio"]["array"]
                sr = sample["audio"].get("sampling_rate", 16000)
                ref_text = sample[text_column]

                inputs = self.processor(audio_arr, sampling_rate=sr, return_tensors="pt").to(self.device)
                
                if self.is_encoder_decoder:
                    input_features = inputs.input_features.to(dtype=target_model.dtype)
                    try:
                        forced_ids = self.processor.get_decoder_prompt_ids(language="turkish", task="transcribe")
                    except Exception:
                        forced_ids = None
                    gen_kwargs = {
                        "max_new_tokens": 256,
                        "no_repeat_ngram_size": 3,
                        "repetition_penalty": 1.2
                    }
                    if forced_ids:
                        gen_kwargs["forced_decoder_ids"] = forced_ids
                    else:
                        gen_kwargs["language"] = "turkish"
                        gen_kwargs["task"] = "transcribe"

                    predicted_ids = target_model.generate(input_features, **gen_kwargs)
                    pred_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                else:
                    logits = target_model(inputs.input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    pred_text = self.processor.batch_decode(predicted_ids)[0]

                refs.append(ref_text)
                preds.append(pred_text)

        wer = compute_wer(refs, preds, normalize=True)
        return wer
