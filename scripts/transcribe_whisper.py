import argparse

from src.qrec.inference.whisper_asr import transcribe_audio


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe Quran audio with Whisper.")
    ap.add_argument("--audio", type=str, required=True)
    ap.add_argument("--model_id", type=str, default="tarteel-ai/whisper-tiny-ar-quran")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    text = transcribe_audio(args.audio, model_id=args.model_id, device=args.device)
    print("Transcription:")
    print(text)


if __name__ == "__main__":
    main()
