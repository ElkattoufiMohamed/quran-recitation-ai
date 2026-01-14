# Quran Recitation AI (Model-Only)

This repo contains:
- ASR (CNN-LSTM-CTC) for constrained Quran transcription
- Tajweed rule classifiers (segment-level audio classification)
- Makhraj detector (articulation classifier via transfer learning)

No API / frontend included.

## Quick ASR smoke test (EveryAyah subset)

1. Download a tiny subset and build an ASR manifest:
   ```bash
   python scripts/download_everyayah_subset.py \
     --dataset everyayah/recitations \
     --config ar \
     --split train \
     --limit 8
   ```
   - Replace the dataset/config values if you want a different EveryAyah variant.
   - Do **not** include angle brackets in your command; they are placeholders.
2. Split into train/dev:
   ```bash
   python scripts/make_splits.py --in_manifest data/processed/everyayah_subset.jsonl
   ```
3. Sanity check and train:
   ```bash
   python scripts/sanity_check.py --manifest data/processed/train.jsonl --kind asr
   python train_asr.py --train_manifest data/processed/train.jsonl --dev_manifest data/processed/dev.jsonl
   ```
