from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

DEFAULT_MODEL_ID = "tarteel-ai/whisper-tiny-ar-quran"


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


def load_whisper_model(model_id: str = DEFAULT_MODEL_ID, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=torch_dtype)
    model.to(device)
    model.eval()

    return processor, model, device


def transcribe_with_word_timestamps(
    audio_array,
    sampling_rate: int,
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
) -> tuple[str, List[WordTimestamp]]:
    processor, model, device = load_whisper_model(model_id=model_id, device=device)
    if getattr(model.generation_config, "no_timestamps_token_id", None) is None:
        token_id = getattr(model.config, "no_timestamps_token_id", None)
        if token_id is None:
            token_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        model.generation_config.no_timestamps_token_id = token_id
    inputs = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

    with torch.no_grad():
        generated = model.generate(
            input_features,
            attention_mask=attention_mask,
            return_timestamps="word",
            return_dict_in_generate=True,
        )

    words: List[WordTimestamp] = []
    decoded_text = ""
    if isinstance(generated, dict) and "segments" in generated:
        for segment in generated.get("segments", []):
            decoded_text += segment.get("text", "")
            for item in segment.get("words", []):
                word = item.get("word", "").strip()
                if not word:
                    continue
                words.append(
                    WordTimestamp(
                        word=word,
                        start=float(item["start"]),
                        end=float(item["end"]),
                    )
                )
        decoded_text = decoded_text.strip()
    else:
        sequences = generated["sequences"] if isinstance(generated, dict) else generated.sequences
        decoded = processor.batch_decode(
            sequences,
            skip_special_tokens=True,
            output_offsets=True,
        )[0]
        decoded_text = decoded.get("text", "")
        for offset in decoded.get("offsets", []):
            word = offset.get("text", "").strip()
            if not word:
                continue
            words.append(
                WordTimestamp(
                    word=word,
                    start=float(offset["start_offset"]),
                    end=float(offset["end_offset"]),
                )
            )

    return decoded_text, words
