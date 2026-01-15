# Quran Recitation AI (Model-Only)

This repo contains:
- ASR (CNN-LSTM-CTC) for constrained Quran transcription
- Tajweed rule classifiers (segment-level audio classification)
- Makhraj detector (articulation classifier via transfer learning)

No API / frontend included.

## Quick ASR smoke test (EveryAyah subset)

1. Download a tiny subset and build an ASR manifest:
   ```bash
   git pull
   pip install -r requirements.txt
   python scripts/download_everyayah_subset.py \
     --dataset tarteel-ai/everyayah \
     --split train \
     --limit 8 \
     --streaming
   ```
   - You can omit `--dataset` to use the default (`tarteel-ai/everyayah`).
   - Use `--streaming` to avoid downloading large shards (audio is fetched as raw bytes).
   - If you want another dataset/config, you can pass:
   ```bash
   python scripts/download_everyayah_subset.py \
     --dataset <HF_DATASET_NAME> \
     --config <OPTIONAL_CONFIG> \
     --split train \
     --limit 8 \
     --streaming
   ```
   - Replace the dataset/config values with the actual EveryAyah dataset you want.
   - To discover available datasets/configs, you can run:
     ```bash
     python - <<'PY'
     from datasets import list_datasets
     print([d for d in list_datasets() if "everyayah" in d.lower()])
     PY
     ```
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
