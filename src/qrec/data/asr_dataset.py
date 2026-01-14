from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset

from .manifests import read_jsonl
from .tokenizer import CharTokenizer
from qrec.utils.audio import AudioFeaturizer, get_or_compute_logmel


@dataclass
class ASRItem:
    audio_path: str
    text: str


class ASRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: CharTokenizer,
        featurizer: AudioFeaturizer,
        mel_cache_dir: str | Path,
    ):
        self.items: List[Dict[str, Any]] = read_jsonl(manifest_path)
        self.tok = tokenizer
        self.feat = featurizer
        self.cache = mel_cache_dir

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        audio_path = it["audio_path"]
        text = it["text"]

        mels, mel_len = get_or_compute_logmel(audio_path, self.cache, self.feat)
        target = torch.tensor(self.tok.encode(text), dtype=torch.long)
        tgt_len = torch.tensor([target.numel()], dtype=torch.long)

        return mels, torch.tensor([mel_len], dtype=torch.long), target, tgt_len


def asr_collate(batch):
    """
    Pads mels to max_T; CTC targets are concatenated.
    batch item:
      mels (n_mels,T), mel_len (1,), target (L,), tgt_len (1,)
    """
    mels_list, mel_lens, targets, tgt_lens = [], [], [], []
    for mels, mel_len, tgt, tl in batch:
        mels_list.append(mels.transpose(0, 1))  # (T,n_mels) for pad_sequence
        mel_lens.append(mel_len.squeeze(0))
        targets.append(tgt)
        tgt_lens.append(tl.squeeze(0))

    mels_pad = torch.nn.utils.rnn.pad_sequence(mels_list, batch_first=True)  # (B,T,n_mels)
    mels_pad = mels_pad.transpose(1, 2).contiguous()                        # (B,n_mels,T)

    mel_lens = torch.stack(mel_lens)    # (B,)
    targets_cat = torch.cat(targets)    # (sumL,)
    tgt_lens = torch.stack(tgt_lens)    # (B,)

    return mels_pad, mel_lens, targets_cat, tgt_lens
