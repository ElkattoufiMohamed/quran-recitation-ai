import argparse
import json
from pathlib import Path
from typing import Iterable

import soundfile as sf
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError


def _infer_field(sample: dict, candidates: Iterable[str]) -> str:
    for key in candidates:
        if key in sample:
            return key
    return ""


def _normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _write_audio(audio, out_path: Path) -> None:
    if isinstance(audio, dict) and "array" in audio:
        sf.write(out_path, audio["array"], audio["sampling_rate"])
        return
    if isinstance(audio, (str, Path)):
        src = Path(audio)
        out_path.write_bytes(src.read_bytes())
        return
    raise ValueError("Unsupported audio field format. Provide --audio_field to match dataset schema.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="tarteel-ai/everyayah", required=False, help="HF dataset name")
    ap.add_argument("--config", type=str, default=None, help="Optional dataset config name")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--streaming", action="store_true", help="Stream samples to avoid full shard downloads")
    ap.add_argument("--out_dir", type=str, default="data/raw/everyayah_subset")
    ap.add_argument("--manifest_path", type=str, default="data/processed/everyayah_subset.jsonl")
    ap.add_argument("--audio_field", type=str, default="audio")
    ap.add_argument("--text_field", type=str, default="text")
    args = ap.parse_args()

    try:
        ds = load_dataset(args.dataset, args.config, split=args.split, streaming=args.streaming)
    except DatasetNotFoundError as exc:
        raise SystemExit(
            "Dataset not found on the Hugging Face Hub. "
            "Double-check the dataset name/config or search with:\n"
            "  python - <<'PY'\n"
            "  from datasets import list_datasets\n"
            "  print([d for d in list_datasets() if 'everyayah' in d.lower()])\n"
            "  PY\n"
            f"Requested dataset: {args.dataset}"
        ) from exc
    if args.seed is not None and not args.streaming:
        ds = ds.shuffle(seed=args.seed)

    if args.streaming:
        limit = args.limit
    else:
        limit = min(args.limit, len(ds))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_items = []
    if args.streaming:
        iterator = iter(ds)
        for idx in range(limit):
            sample = next(iterator)
            audio_field = args.audio_field
            text_field = args.text_field

            if audio_field not in sample:
                audio_field = _infer_field(sample, ["audio", "recitation", "wav", "speech"])
            if text_field not in sample:
                text_field = _infer_field(sample, ["text", "ayah", "verse", "transcript"])

            if not audio_field or not text_field:
                raise KeyError("Could not infer audio/text fields. Use --audio_field and --text_field.")

            audio = sample[audio_field]
            text = _normalize_text(sample[text_field])

            audio_path = out_dir / f"sample_{idx:04d}.wav"
            _write_audio(audio, audio_path)

            manifest_items.append({"audio_path": str(audio_path), "text": text})
    else:
        for idx in range(limit):
            sample = ds[idx]
            audio_field = args.audio_field
            text_field = args.text_field

            if audio_field not in sample:
                audio_field = _infer_field(sample, ["audio", "recitation", "wav", "speech"])
        if text_field not in sample:
            text_field = _infer_field(sample, ["text", "ayah", "verse", "transcript"])

            if not audio_field or not text_field:
                raise KeyError("Could not infer audio/text fields. Use --audio_field and --text_field.")

            audio = sample[audio_field]
            text = _normalize_text(sample[text_field])

            audio_path = out_dir / f"sample_{idx:04d}.wav"
            _write_audio(audio, audio_path)

            manifest_items.append({"audio_path": str(audio_path), "text": text})

    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for item in manifest_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(manifest_items)} samples to {out_dir} and manifest {manifest_path}")


if __name__ == "__main__":
    main()
