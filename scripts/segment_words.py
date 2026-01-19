import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qrec.inference.whisperx_asr import (  # noqa: E402
    WordTimestamp,
    transcribe_with_word_timestamps,
)


def _safe_word(word: str) -> str:
    word = re.sub(r"\s+", "_", word.strip())
    word = re.sub(r"[^\w\-\u0600-\u06FF]+", "", word)
    return word or "word"


def _write_segment(audio, sr, start_s: float, end_s: float, out_path: Path) -> None:
    start_idx = max(0, int(start_s * sr))
    end_idx = min(len(audio), int(end_s * sr))
    if end_idx <= start_idx:
        return
    segment = audio[start_idx:end_idx]
    if out_path.suffix.lower() == ".wav":
        sf.write(out_path, segment, sr)
        return

    if out_path.suffix.lower() == ".mp3":
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise SystemExit("ffmpeg is required to write mp3 output.")
        tmp_wav = out_path.with_suffix(".tmp.wav")
        sf.write(tmp_wav, segment, sr)
        subprocess.run(
            [ffmpeg, "-y", "-i", str(tmp_wav), str(out_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tmp_wav.unlink(missing_ok=True)
        return

    raise SystemExit(f"Unsupported output format: {out_path.suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe audio and split into word clips.")
    ap.add_argument("--audio", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--model_size", type=str, default="tiny")
    ap.add_argument("--compute_type", type=str, default="float32")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--format", type=str, default="wav", choices=["wav", "mp3"])
    args = ap.parse_args()

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    text, words = transcribe_with_word_timestamps(
        audio,
        sampling_rate=sr,
        device=args.device,
        model_size=args.model_size,
        compute_type=args.compute_type,
    )
    if not words:
        tokens = [tok for tok in text.split() if tok.strip()]
        if tokens:
            total_dur = len(audio) / float(sr)
            step = total_dur / len(tokens)
            words = [
                WordTimestamp(token, i * step, (i + 1) * step) for i, token in enumerate(tokens)
            ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, word in enumerate(words, start=1):
        safe = _safe_word(word.word)
        out_path = output_dir / f"{idx:04d}_{safe}.{args.format}"
        _write_segment(audio, sr, word.start, word.end, out_path)

    transcript_path = output_dir / "transcription.txt"
    transcript_path.write_text(text, encoding="utf-8")
    print(f"Wrote {len(words)} word clips to {output_dir}")


if __name__ == "__main__":
    main()
