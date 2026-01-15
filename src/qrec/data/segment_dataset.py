from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset

from .manifests import read_jsonl
from ..utils.audio import AudioFeaturizer, get_or_compute_logmel


class SegmentDataset(Dataset):
    """
    For tajweed/makhraj segment-level training.

    Manifest item:
      segment_path: str
      label: int  (tajweed correct/incorrect OR makhraj class id)
    """
    def __init__(
        self,
        manifest_path: str | Path,
        featurizer: AudioFeaturizer,
        mel_cache_dir: str | Path,
        label_key: str = "label",
    ):
        self.items: List[Dict[str, Any]] = read_jsonl(manifest_path)
        self.feat = featurizer
        self.cache = mel_cache_dir
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        seg_path = it["segment_path"]
        y = int(it[self.label_key])

        mels, _ = get_or_compute_logmel(seg_path, self.cache, self.feat)
        y = torch.tensor(y, dtype=torch.long)
        return mels, y


def segment_collate(batch):
    """
    Pads variable-length segments.
    batch item: mels (n_mels,T), y ()
    """
    mels_list, ys = [], []
    for mels, y in batch:
        mels_list.append(mels.transpose(0, 1))  # (T,n_mels)
        ys.append(y)

    mels_pad = torch.nn.utils.rnn.pad_sequence(mels_list, batch_first=True)  # (B,T,n_mels)
    mels_pad = mels_pad.transpose(1, 2).contiguous()                        # (B,n_mels,T)
    ys = torch.stack(ys)
    return mels_pad, ys
