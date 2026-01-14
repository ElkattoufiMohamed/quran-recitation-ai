#!/usr/bin/env bash
set -euo pipefail

ROOT="quran-recitation-ai"
mkdir -p "$ROOT"
cd "$ROOT"

# folders
mkdir -p configs data/{raw,interim,processed} scripts src/qrec/{utils,data,models,training,inference} tests runs/{asr,tajweed,makhraj}

# gitignore
cat > .gitignore <<'EOF'
__pycache__/
*.pyc
.venv/
.env
.ipynb_checkpoints/
runs/
data/interim/
data/processed/
.DS_Store
EOF

# requirements (minimal; add others as needed)
cat > requirements.txt <<'EOF'
torch>=2.1
torchaudio>=2.1
tqdm
pyyaml
numpy
soundfile
EOF

# pyproject
cat > pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "qrec"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []
EOF

# README
cat > README.md <<'EOF'
# Quran Recitation AI (Model-Only)

This repo contains:
- ASR (CNN-LSTM-CTC) for constrained Quran transcription
- Tajweed rule classifiers (segment-level audio classification)
- Makhraj detector (articulation classifier via transfer learning)

No API / frontend included.
EOF

# package init
cat > src/qrec/__init__.py <<'EOF'
__all__ = ["models", "data", "training", "inference", "utils"]
EOF

# basic configs
cat > configs/asr.yaml <<'EOF'
seed: 1337
sample_rate: 16000
n_mels: 80
win_length_ms: 25
hop_length_ms: 10

model:
  cnn_channels: [32, 64]
  lstm_hidden: 512
  lstm_layers: 3
  dropout: 0.1

train:
  batch_size: 16
  lr: 0.0005
  weight_decay: 0.0001
  max_epochs: 50
  grad_clip: 5.0
EOF

cat > configs/tajweed.yaml <<'EOF'
seed: 1337
sample_rate: 16000
n_mels: 80

model:
  base_channels: 64
  dropout: 0.2

train:
  batch_size: 32
  lr: 0.001
  max_epochs: 30

rules:
  - madd
  - ghunnah
  - qalqalah
  - idgham
  - ikhfa
EOF

cat > configs/makhraj.yaml <<'EOF'
seed: 1337
sample_rate: 16000
n_mels: 80

model:
  dropout: 0.2

train:
  batch_size: 32
  lr: 0.0005
  max_epochs: 30
EOF

echo "✅ Repo skeleton created at: $(pwd)"
