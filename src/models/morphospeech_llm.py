"""
MorphoSpeech-LLM: SOTA Turkish Automatic Speech Recognition Model Architecture.
Combines:
1. Morfessor-BPE Fusion Agglutinative Tokenizer (TurkishMorphoTokenizer)
2. Conformer / WavLM / Whisper Acoustic Encoder with Auxiliary CTC Loss
3. Cross-Attention Audio-LLM Adapter & Turkish Decoder Guidance
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModel,
    AutoModelForSpeechSeq2Seq,
    AutoModelForCTC,
    AutoConfig
)

from src.tokenizers.turkish_morpho_tokenizer import TurkishMorphoTokenizer

logger = logging.getLogger(__name__)


class MorphoSpeechConfig(PretrainedConfig):
    """
    Configuration dataclass for MorphoSpeech-LLM.
    """
    model_type = "morphospeech_llm"

    def __init__(
        self,
        acoustic_model_name: str = "facebook/wav2vec2-xls-r-300m",
        decoder_model_name: Optional[str] = None,
        vocab_size: int = 8000,
        encoder_dim: int = 1024,
        decoder_dim: int = 768,
        adapter_dim: int = 512,
        ctc_weight: float = 0.3,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.acoustic_model_name = acoustic_model_name
        self.decoder_model_name = decoder_model_name
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.adapter_dim = adapter_dim
        self.ctc_weight = ctc_weight
        self.freeze_encoder = freeze_encoder
        self.dropout = dropout


class AudioToLLMAdapter(nn.Module):
    """
    Convolutional & Linear Adapter to align acoustic feature frame rate 
    and feature dimensions with the Language Model Decoder embeddings.
    """

    def __init__(self, in_dim: int, out_dim: int, subsample_factor: int = 2, dropout: float = 0.1):
        super().__init__()
        self.conv_subsample = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=subsample_factor, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )
        self.projection = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, in_dim)
        x_trans = x.transpose(1, 2)  # (batch_size, in_dim, seq_len)
        x_conv = self.conv_subsample[0](x_trans)
        x_conv = self.conv_subsample[1](x_conv)
        x_conv = self.conv_subsample[2](x_conv)
        x_conv = self.conv_subsample[3](x_conv)
        x_conv = self.conv_subsample[4](x_conv)
        x_conv = x_conv.transpose(1, 2)  # (batch_size, seq_len_subsampled, out_dim)
        x_conv = self.conv_subsample[5](x_conv)
        return self.projection(x_conv)


class MorphoSpeechLLM(PreTrainedModel):
    """
    MorphoSpeech-LLM End-to-End PyTorch Model.
    Integrates acoustic representation, auxiliary CTC morpheme loss, 
    and agglutinative subword decoding.
    """

    config_class = MorphoSpeechConfig

    def __init__(self, config: MorphoSpeechConfig, tokenizer: Optional[TurkishMorphoTokenizer] = None):
        super().__init__(config)
        self.config = config
        self.tokenizer = tokenizer

        # 1. Acoustic Encoder Initialization
        logger.info(f"Initializing Acoustic Encoder from '{config.acoustic_model_name}'...")
        if config.acoustic_model_name.startswith("dummy"):
            self.encoder = nn.Linear(config.encoder_dim, config.encoder_dim)
        else:
            try:
                encoder_base = AutoModel.from_pretrained(config.acoustic_model_name, local_files_only=False)
                if hasattr(encoder_base, "encoder"):
                    self.encoder = encoder_base.encoder
                elif hasattr(encoder_base, "wav2vec2"):
                    self.encoder = encoder_base.wav2vec2
                else:
                    self.encoder = encoder_base
            except Exception as e:
                logger.warning(f"Could not load AutoModel for {config.acoustic_model_name}: {e}. Using local encoder fallback...")
                self.encoder = nn.Linear(config.encoder_dim, config.encoder_dim)

        if config.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            logger.info("Acoustic Encoder parameters frozen.")

        # Determine actual encoder output dimension
        encoder_cfg = getattr(self.encoder, "config", None)
        hidden_size = getattr(encoder_cfg, "hidden_size", getattr(encoder_cfg, "d_model", config.encoder_dim)) if encoder_cfg else config.encoder_dim
        self.config.encoder_dim = hidden_size

        # 2. Auxiliary CTC Projection Head over Morpheme Vocab
        self.ctc_head = nn.Linear(self.config.encoder_dim, config.vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(blank=config.pad_token_id, zero_infinity=True)

        # 3. Audio-to-LLM Adapter Layer
        self.adapter = AudioToLLMAdapter(
            in_dim=self.config.encoder_dim,
            out_dim=config.decoder_dim,
            dropout=config.dropout
        )

        # 4. Decoder Embedding & Linear Output Projection
        self.token_embeddings = nn.Embedding(config.vocab_size, config.decoder_dim, padding_idx=config.pad_token_id)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_dim,
            nhead=8,
            dim_feedforward=config.decoder_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        self.lm_head = nn.Linear(config.decoder_dim, config.vocab_size)

        # Initialize weights
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def extract_features(self, input_features_or_values: torch.Tensor) -> torch.Tensor:
        """
        Extracts frame-level acoustic features from input spectrogram / audio tensor.
        """
        encoder_outputs = self.encoder(input_features_or_values)
        if hasattr(encoder_outputs, "last_hidden_state"):
            hidden_states = encoder_outputs.last_hidden_state
        elif isinstance(encoder_outputs, tuple):
            hidden_states = encoder_outputs[0]
        else:
            hidden_states = encoder_outputs
        return hidden_states

    def forward(
        self,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """
        Forward pass with joint CTC + Cross-Attention Language Model Loss.
        """
        # 1. Acoustic Encoder Forward
        acoustic_feats = self.extract_features(input_features)  # (B, T, D_enc)
        batch_size, seq_len, _ = acoustic_feats.size()

        # 2. Auxiliary CTC Logits & Loss
        ctc_logits = self.ctc_head(acoustic_feats)  # (B, T, Vocab)
        
        ctc_loss = torch.tensor(0.0, device=input_features.device)
        lm_loss = torch.tensor(0.0, device=input_features.device)

        if labels is not None:
            # CTC Loss Computation
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (T, B, Vocab)
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=input_features.device)
            
            # Mask out invalid padding labels for CTC
            ctc_target = labels.clone()
            ctc_target[ctc_target == -100] = self.config.pad_token_id
            target_lengths = torch.sum((ctc_target != self.config.pad_token_id).long(), dim=-1)
            target_lengths = torch.clamp(target_lengths, min=1)

            ctc_loss = self.ctc_loss_fn(log_probs, ctc_target, input_lengths, target_lengths)

            # 3. LLM Decoder Forward & Cross-Entropy Loss
            if decoder_input_ids is None:
                # Prepare decoder_input_ids by shifting labels
                decoder_input_ids = labels.clone()
                decoder_input_ids[decoder_input_ids == -100] = self.config.pad_token_id
                # Insert BOS token at beginning
                bos = torch.full((batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=labels.device)
                decoder_input_ids = torch.cat([bos, decoder_input_ids[:, :-1]], dim=1)

            memory = self.adapter(acoustic_feats)  # (B, T_sub, D_dec)
            tgt_embed = self.token_embeddings(decoder_input_ids)  # (B, T_lbl, D_dec)
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_embed.size(1), device=tgt_embed.device)
            tgt_key_padding_mask = (decoder_input_ids == self.config.pad_token_id)

            dec_outputs = self.decoder(
                tgt=tgt_embed,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
            lm_logits = self.lm_head(dec_outputs)  # (B, T_lbl, Vocab)
            
            # Compute Cross-Entropy Loss
            active_loss = (labels != -100)
            shift_logits = lm_logits.view(-1, self.config.vocab_size)
            shift_labels = labels.view(-1)
            lm_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        # Combined Joint Loss
        total_loss = self.config.ctc_weight * ctc_loss + (1.0 - self.config.ctc_weight) * lm_loss

        if not return_dict:
            return (total_loss, ctc_logits, lm_logits)

        return {
            "loss": total_loss,
            "ctc_loss": ctc_loss,
            "lm_loss": lm_loss,
            "ctc_logits": ctc_logits,
            "lm_logits": lm_logits if labels is not None else None
        }

    @torch.no_grad()
    def generate_transcription(
        self,
        input_features: torch.Tensor,
        max_length: int = 128
    ) -> torch.Tensor:
        """
        Autoregressive Decoding for Inference.
        """
        self.eval()
        batch_size = input_features.size(0)
        acoustic_feats = self.extract_features(input_features)
        memory = self.adapter(acoustic_feats)

        ys = torch.full((batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=input_features.device)

        for _ in range(max_length - 1):
            tgt_embed = self.token_embeddings(ys)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1), device=ys.device)
            out = self.decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask)
            logits = self.lm_head(out[:, -1, :])
            next_word = torch.argmax(logits, dim=-1, keepdim=True)
            ys = torch.cat([ys, next_word], dim=1)
            
            # Stop if all sequences hit EOS
            if (next_word == self.config.eos_token_id).all():
                break

        return ys
