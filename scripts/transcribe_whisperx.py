import argparse
import sys
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qrec.inference.whisperx_asr import transcribe_with_word_timestamps  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe Quran audio with WhisperX.")
    ap.add_argument("--audio", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--model_size", type=str, default="tiny")
    ap.add_argument("--compute_type", type=str, default="float32")
    args = ap.parse_args()

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    text, _ = transcribe_with_word_timestamps(
        audio,
        sampling_rate=sr,
        device=args.device,
        model_size=args.model_size,
        compute_type=args.compute_type,
    )
    print("Transcription:")
    print(text)


if __name__ == "__main__":
    main()
