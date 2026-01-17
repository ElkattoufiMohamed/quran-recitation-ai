from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

DEFAULT_MODEL_ID = "tarteel-ai/whisper-tiny-ar-quran"


def load_whisper_pipeline(model_id: str = DEFAULT_MODEL_ID, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(device)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if device == "cuda" else -1,
    )


def transcribe_audio(audio_path: str, model_id: str = DEFAULT_MODEL_ID, device: Optional[str] = None) -> str:
    asr = load_whisper_pipeline(model_id=model_id, device=device)
    result = asr(audio_path)
    return result["text"]
