# Quran Recitation AI (Model-Only)

This repo contains:
- ASR via a fine-tuned Whisper model for Quran transcription
- Tajweed rule classifiers (segment-level audio classification)

No API / frontend included.

## Whisper ASR (Quran transcription)

Use the fine-tuned Whisper model from Tarteel:

```bash
python scripts/transcribe_whisper.py --audio /path/to/audio.wav
```

If you run into `ModuleNotFoundError: No module named 'src'`, ensure the repo
root is on `PYTHONPATH`:
```bash
export PYTHONPATH=.
```

The script prints the transcription and uses `tarteel-ai/whisper-tiny-ar-quran`
by default. You can override the model with `--model_id`.

## Tajweed rule verification setup

The Tajweed stack includes rule definitions, feature extraction helpers, and
rule-specific detectors (LSTM/CNN) that can be trained once data arrives. See
`src/qrec/tajweed` for the rule definitions, feature extractor, detectors, and
verification pipeline.

### Prepare a tajweed manifest

You can build a JSONL manifest from a folder tree or from a CSV annotation file.
The manifest format is compatible with the segment-level training pipeline.

From a folder layout:
```
data/tajweed/
└── al_mad/
    ├── correct/
    └── incorrect/
```
Run:
```bash
python scripts/build_tajweed_manifest.py \
  --data_dir data/tajweed \
  --output data/processed/tajweed_manifest.jsonl
```

From a CSV annotation:
```bash
python scripts/build_tajweed_manifest.py \
  --annotations data/tajweed_annotations.csv \
  --base_dir data/tajweed \
  --output data/processed/tajweed_manifest.jsonl
```

### Train a tajweed CNN classifier

```bash
python train_tajweed.py \
  --train_manifest data/processed/train.jsonl \
  --dev_manifest data/processed/dev.jsonl \
  --rule_name al_mad
```

### Train a tajweed LSTM classifier (MFCC + delta)

```bash
python train_tajweed_lstm.py \
  --train_manifest data/processed/train.jsonl \
  --dev_manifest data/processed/dev.jsonl \
  --rule_name al_mad
```

### Simple feedback formatting

Use the verification pipeline output to build user-facing feedback:
```python
from src.qrec.tajweed.feedback import format_tajweed_feedback

feedback = format_tajweed_feedback(results)
print(feedback)
```
