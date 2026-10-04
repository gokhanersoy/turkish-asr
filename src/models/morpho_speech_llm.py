"""
MorphoSpeech-LLM Model Architecture.
Combines a pre-trained Speech Encoder with Auxiliary CTC Loss,
a Morpho-Aware Vocabulary Adapter, and a LoRA/QLoRA Fine-Tuned Audio-Decoder.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from peft import LoraConfig, get_peft_model, TaskType

from src.asr_tokenizer.morpho_tokenizer import MorphoTokenizer


class AuxiliaryCTCHead(nn.Module):
    """Auxiliary CTC Head attached to speech encoder representations for acoustic grounding."""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_states: torch.Tensor) -> torch.Tensor:
        x = self.dropout(torch.tanh(self.dense(encoder_states)))
        logits = self.out_proj(x)
        return logits


class MorphoSpeechLLM(nn.Module):
    """
    MorphoSpeech-LLM Architecture for High-Performance Turkish ASR.
    Supports Whisper Large-v3 backbone, PEFT/LoRA adaptation, and auxiliary CTC loss.
    """

    def __init__(
        self,
        model_name_or_path: str = "openai/whisper-large-v3",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        ctc_weight: float = 0.2,
        language: str = "turkish",
        task: str = "transcribe",
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.ctc_weight = ctc_weight
        self.language = language
        self.task = task

        # Load Base Model
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )

        # Configure forced decoder ids for Turkish transcription
        self.processor = WhisperProcessor.from_pretrained(
            model_name_or_path,
            language=language,
            task=task,
        )
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task=task
        )
        self.model.config.suppress_tokens = []

        # Auxiliary CTC Loss Head
        hidden_size = self.model.config.d_model
        vocab_size = self.model.config.vocab_size
        self.aux_ctc_head = AuxiliaryCTCHead(hidden_size, vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(zero_infinity=True)

        # Apply LoRA if requested
        if use_lora:
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def get_encoder(self):
        """Safely retrieve the speech encoder regardless of PEFT wrapper hierarchy."""
        if hasattr(self.model, "get_encoder"):
            return self.model.get_encoder()
        elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "get_encoder"):
            return self.model.base_model.get_encoder()
        elif hasattr(self.model, "model") and hasattr(self.model.model, "encoder"):
            return self.model.model.encoder
        return getattr(self.model, "encoder", None)

    def forward(
        self,
        input_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        # Remove duplicate keys from kwargs to avoid PEFT/Whisper keyword conflicts
        kwargs.pop("input_features", None)
        kwargs.pop("labels", None)
        kwargs.pop("decoder_input_ids", None)
        kwargs.pop("input_ids", None)

        forward_kwargs = {
            "output_hidden_states": True,
            "return_dict": True,
        }
        if input_features is not None:
            forward_kwargs["input_features"] = input_features
        if labels is not None:
            forward_kwargs["labels"] = labels
        if decoder_input_ids is not None:
            forward_kwargs["decoder_input_ids"] = decoder_input_ids

        outputs = self.model(**forward_kwargs, **kwargs)

        hidden_states = getattr(outputs, "encoder_last_hidden_state", None)
        if hidden_states is None and hasattr(outputs, "encoder_hidden_states") and outputs.encoder_hidden_states:
            hidden_states = outputs.encoder_hidden_states[-1]

        loss = outputs.loss if labels is not None else None

        # Compute Auxiliary CTC Loss if labels are provided during training
        if labels is not None and self.ctc_weight > 0.0 and hidden_states is not None:
            ctc_logits = self.aux_ctc_head(hidden_states)
            log_probs = torch.log_softmax(ctc_logits, dim=-1).transpose(0, 1) # [seq_len, batch_size, vocab_size]

            input_lengths = torch.full(
                (input_features.shape[0],),
                fill_value=hidden_states.shape[1],
                dtype=torch.long,
                device=input_features.device,
            )

            # Mask out -100 padding in labels for CTC
            target_labels = labels.clone()
            target_labels[target_labels == -100] = 0
            target_lengths = (labels != -100).sum(dim=-1)

            ctc_loss = self.ctc_loss_fn(log_probs, target_labels, input_lengths, target_lengths)
            loss = (1.0 - self.ctc_weight) * loss + self.ctc_weight * ctc_loss

        if not return_dict:
            return (loss, outputs.logits) if loss is not None else outputs.logits

        return {
            "loss": loss,
            "logits": outputs.logits,
            "encoder_last_hidden_state": hidden_states,
        }

    def generate(self, input_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate predicted transcript token IDs."""
        return self.model.generate(input_features=input_features, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegate gradient checkpointing enablement to inner base model."""
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """Delegate gradient checkpointing disablement to inner base model."""
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
