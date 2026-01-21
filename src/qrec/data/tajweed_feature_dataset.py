from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from .manifests import read_jsonl
from ..tajweed.features import TajweedFeatureConfig, TajweedFeatureExtractor


@dataclass
class TajweedFeatureCacheConfig:
    cache_dir: str | Path

    def ensure_dir(self) -> Path:
        path = Path(self.cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


def _feature_cache_path(cache_dir: str | Path, audio_path: str | Path, suffix: str) -> Path:
    cache_dir = Path(cache_dir)
    audio_path = Path(audio_path)
    safe = str(audio_path).replace("/", "_").replace(" ", "_")
    return cache_dir / f"{safe}.{suffix}.pt"


class TajweedFeatureDataset(Dataset):
    """
    Returns MFCC+delta+delta2 features shaped (T, F) for LSTM training.

    Manifest item:
      segment_path: str
      label: int
    """

    def __init__(
        self,
        manifest_path: str | Path,
        feature_cfg: TajweedFeatureConfig,
        cache_cfg: TajweedFeatureCacheConfig | None = None,
        label_key: str = "label",
    ) -> None:
        self.items: List[Dict[str, Any]] = read_jsonl(manifest_path)
        self.extractor = TajweedFeatureExtractor(feature_cfg)
        self.cache_cfg = cache_cfg
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.items)

    def _load_or_compute(self, segment_path: str) -> torch.Tensor:
        if not self.cache_cfg:
            features = self.extractor.extract_all_features(segment_path, rule=None)
            return features["mfcc_delta"].transpose(0, 1)

        cache_dir = self.cache_cfg.ensure_dir()
        cache_path = _feature_cache_path(cache_dir, segment_path, "mfccdelta")
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu")

        features = self.extractor.extract_all_features(segment_path, rule=None)
        mfcc_delta = features["mfcc_delta"].transpose(0, 1)
        torch.save(mfcc_delta, cache_path)
        return mfcc_delta

    def __getitem__(self, idx: int):
        item = self.items[idx]
        segment_path = item["segment_path"]
        label = int(item[self.label_key])

        feats = self._load_or_compute(segment_path)
        y = torch.tensor(label, dtype=torch.float32)
        return feats, y


def tajweed_feature_collate(batch):
    """
    Pads variable-length MFCC+delta sequences.
    batch item: feats (T,F), y ()
    """
    feats_list, ys = [], []
    for feats, y in batch:
        feats_list.append(feats)
        ys.append(y)

    feats_pad = torch.nn.utils.rnn.pad_sequence(feats_list, batch_first=True)
    ys = torch.stack(ys)
    return feats_pad, ys
