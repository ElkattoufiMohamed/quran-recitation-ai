from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

DEFAULT_MODEL_ID = "tarteel-ai/whisper-tiny-ar-quran"


def load_whisper_model(model_id: str = DEFAULT_MODEL_ID, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=torch_dtype)
    model.to(device)
    model.eval()

    return processor, model, device


def transcribe_audio(
    audio_array,
    sampling_rate: int,
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
) -> str:
    processor, model, device = load_whisper_model(model_id=model_id, device=device)
    inputs = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
