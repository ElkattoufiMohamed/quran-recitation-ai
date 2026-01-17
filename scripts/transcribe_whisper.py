import argparse
import sys
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qrec.inference.whisper_asr import transcribe_audio  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe Quran audio with Whisper.")
    ap.add_argument("--audio", type=str, required=True)
    ap.add_argument("--model_id", type=str, default="tarteel-ai/whisper-tiny-ar-quran")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    text = transcribe_audio(
        {"array": audio, "sampling_rate": sr},
        model_id=args.model_id,
        device=args.device,
    )
    print("Transcription:")
    print(text)


if __name__ == "__main__":
    main()
