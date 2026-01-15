from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import torch
import torchaudio
import soundfile as sf


@dataclass
class MelSpecConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    win_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    n_fft: Optional[int] = None  # if None, infer from win_length

    def win_length(self) -> int:
        return int(self.sample_rate * self.win_length_ms / 1000.0)

    def hop_length(self) -> int:
        return int(self.sample_rate * self.hop_length_ms / 1000.0)


class AudioFeaturizer:
    def __init__(self, cfg: MelSpecConfig):
        self.cfg = cfg
        win_len = cfg.win_length()
        hop_len = cfg.hop_length()
        n_fft = cfg.n_fft or 2 ** int((win_len - 1).bit_length())  # next pow2

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=n_fft,
            win_length=win_len,
            hop_length=hop_len,
            n_mels=cfg.n_mels,
            center=True,
            power=2.0,
        )
        self.amptodb = torchaudio.transforms.AmplitudeToDB(stype="power")

    def load_audio(self, path: str | Path) -> torch.Tensor:
        wav_np, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_np).transpose(0, 1)  # (C, T)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
        wav = wav.squeeze(0)  # (T,)
        return wav

    def wav_to_logmel(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (T,)
        m = self.melspec(wav.unsqueeze(0))     # (1, n_mels, frames)
        m = self.amptodb(m)                    # log power
        return m.squeeze(0)                    # (n_mels, frames)


def mel_cache_path(cache_dir: str | Path, audio_path: str | Path) -> Path:
    cache_dir = Path(cache_dir)
    audio_path = Path(audio_path)
    # stable relative-like key
    safe = str(audio_path).replace("/", "_").replace(" ", "_")
    return cache_dir / f"{safe}.pt"


def get_or_compute_logmel(
    audio_path: str | Path,
    cache_dir: str | Path,
    featurizer: AudioFeaturizer,
) -> Tuple[torch.Tensor, int]:
    """
    Returns: (logmel [n_mels, frames], frames_len)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = mel_cache_path(cache_dir, audio_path)

    if cpath.exists():
        m = torch.load(cpath, map_location="cpu")
        return m, m.size(-1)

    wav = featurizer.load_audio(audio_path)
    m = featurizer.wav_to_logmel(wav)
    torch.save(m, cpath)
    return m, m.size(-1)
