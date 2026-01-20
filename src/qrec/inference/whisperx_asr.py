from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torchaudio
import whisperx


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


def _ensure_sample_rate(audio: np.ndarray, sr: int, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    if sr == target_sr:
        return audio, sr
    wav = torch.from_numpy(audio.astype(np.float32))
    if wav.ndim > 1:
        wav = wav.mean(dim=1)
    wav = wav.unsqueeze(0)
    resampled = torchaudio.functional.resample(wav, sr, target_sr).squeeze(0)
    return resampled.numpy(), target_sr


def transcribe_with_word_timestamps(
    audio_array: np.ndarray,
    sampling_rate: int,
    device: Optional[str] = None,
    model_size: str = "tiny",
    compute_type: str = "float32",
) -> tuple[str, List[WordTimestamp]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    audio_array, sampling_rate = _ensure_sample_rate(audio_array, sampling_rate)

    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    result = model.transcribe(audio_array)

    language = result.get("language", "ar")
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(result["segments"], align_model, metadata, audio_array, device)

    words: List[WordTimestamp] = []
    for item in aligned.get("word_segments", []):
        word = item.get("word", "").strip()
        if not word:
            continue
        words.append(WordTimestamp(word=word, start=float(item["start"]), end=float(item["end"])))

    text = " ".join(seg.get("text", "").strip() for seg in result.get("segments", [])).strip()
    return text, words
