"""
Turkish ASR Model Benchmark Evaluator.
Direct PyTorch + Transformers Model & Processor inference for maximum stability and speed.
Supports Whisper (Encoder-Decoder) and Wav2Vec2 / HuBERT / MMS (CTC) models.
Completely bypasses torchcodec / ffmpeg pipeline dependencies.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
import numpy as np

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCTC, AutoConfig
from src.utils.metrics import evaluate_asr_predictions
from src.utils.text_normalization import normalize_turkish_text

logger = logging.getLogger(__name__)

class TurkishASREvaluator:
    def __init__(
        self, 
        model_name_or_path: str = "openai/whisper-small",
        device: Optional[str] = None,
        language: str = "turkish"
    ):
        self.model_name = model_name_or_path
        self.language = language
        
        # Determine target compute device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Loading Model & Processor for '{model_name_or_path}' on device '{self.device}'...")
        
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        
        # Check architecture type via AutoConfig
        try:
            config = AutoConfig.from_pretrained(model_name_or_path)
            model_type = getattr(config, "model_type", "").lower()
            archs = str(getattr(config, "architectures", [])).lower()
            if "whisper" in model_type or "whisper" in archs or "whisper" in model_name_or_path.lower():
                self.is_encoder_decoder = True
            else:
                self.is_encoder_decoder = False
        except Exception:
            self.is_encoder_decoder = "whisper" in model_name_or_path.lower()
        
        if self.is_encoder_decoder:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path)
            # Setup Turkish forced decoder prompt IDs for Whisper
            try:
                self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.language, task="transcribe")
            except Exception:
                self.forced_decoder_ids = None
        else:
            self.is_encoder_decoder = False
            self.model = AutoModelForCTC.from_pretrained(model_name_or_path)
            self.forced_decoder_ids = None
            
        self.model.to(self.device)
        self.model.eval()

    def transcribe_audio(self, audio_array: np.ndarray, sampling_rate: int = 16000) -> str:
        """
        Transcribes raw audio numpy array using direct PyTorch model inference.
        """
        inputs = self.processor(
            audio_array, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            if self.is_encoder_decoder:
                input_features = inputs.input_features.to(self.device)
                gen_kwargs = {"forced_decoder_ids": self.forced_decoder_ids} if self.forced_decoder_ids else {"language": self.language, "task": "transcribe"}
                predicted_ids = self.model.generate(
                    input_features, 
                    **gen_kwargs
                )
                transcription = self.processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )[0]
            else:
                input_values = inputs.input_values.to(self.device)
                logits = self.model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
                
        return transcription.strip()

    def evaluate_dataset(
        self, 
        dataset, 
        text_column: str = "sentence", 
        audio_column: str = "audio",
        normalize: bool = True
    ) -> Dict[str, Any]:
        """
        Runs inference over dataset samples and calculates WER and CER.
        """
        references = []
        predictions = []
        inference_times = []

        logger.info(f"Evaluating {len(dataset)} samples with model '{self.model_name}'...")

        for sample in tqdm(dataset, desc=f"Evaluating {self.model_name.split('/')[-1]}"):
            ref_text = sample[text_column]
            audio_data = sample[audio_column]
            
            # Handle audio dict structure
            if isinstance(audio_data, dict):
                audio_array = audio_data["array"]
                sampling_rate = audio_data.get("sampling_rate", 16000)
            else:
                audio_array = audio_data
                sampling_rate = 16000

            start_time = time.time()
            pred_text = self.transcribe_audio(audio_array, sampling_rate=sampling_rate)
            elapsed_time = time.time() - start_time

            references.append(ref_text)
            predictions.append(pred_text)
            inference_times.append(elapsed_time)

        # Compute metrics
        metrics = evaluate_asr_predictions(references, predictions, normalize=normalize)
        avg_latency = sum(inference_times) / max(1, len(inference_times))

        results = {
            "model_name": self.model_name,
            "sample_count": len(dataset),
            "wer_percentage": metrics["wer"],
            "cer_percentage": metrics["cer"],
            "avg_latency_seconds": round(avg_latency, 3),
            "predictions_sample": [
                {"ref": r, "pred": p} for r, p in zip(references[:5], predictions[:5])
            ]
        }

        return results
