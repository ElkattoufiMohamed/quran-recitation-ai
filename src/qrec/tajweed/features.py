from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf

from .rules import TajweedRule


@dataclass
class TajweedFeatureConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_mfcc: int = 13
    win_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    fmin: float = 0.0
    fmax: Optional[float] = None

    def win_length(self) -> int:
        return int(self.sample_rate * self.win_length_ms / 1000.0)

    def hop_length(self) -> int:
        return int(self.sample_rate * self.hop_length_ms / 1000.0)


class TajweedFeatureExtractor:
    def __init__(self, cfg: Optional[TajweedFeatureConfig] = None):
        self.cfg = cfg or TajweedFeatureConfig()
        win_len = self.cfg.win_length()
        hop_len = self.cfg.hop_length()
        n_fft = 2 ** int((win_len - 1).bit_length())

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg.sample_rate,
            n_fft=n_fft,
            win_length=win_len,
            hop_length=hop_len,
            n_mels=self.cfg.n_mels,
            f_min=self.cfg.fmin,
            f_max=self.cfg.fmax,
            center=True,
            power=2.0,
        )
        self.amptodb = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.cfg.sample_rate,
            n_mfcc=self.cfg.n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": self.cfg.n_mels,
                "win_length": win_len,
                "hop_length": hop_len,
                "f_min": self.cfg.fmin,
                "f_max": self.cfg.fmax,
                "power": 2.0,
            },
        )

    def load_audio(self, path: str | Path) -> torch.Tensor:
        wav_np, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_np).transpose(0, 1)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
        return wav.squeeze(0)

    def _ensure_tensor(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(audio, torch.Tensor):
            wav = audio
        else:
            wav = torch.from_numpy(audio.astype(np.float32))
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        return wav

    def _frame_rms(self, wav: torch.Tensor) -> torch.Tensor:
        win_len = self.cfg.win_length()
        hop_len = self.cfg.hop_length()
        if wav.numel() < win_len:
            pad = win_len - wav.numel()
            wav = torch.nn.functional.pad(wav, (0, pad))
        frames = wav.unfold(0, win_len, hop_len)
        return torch.sqrt(frames.pow(2).mean(dim=-1) + 1e-10)

    def _align_feature_length(self, feat: torch.Tensor, target_len: int) -> torch.Tensor:
        if feat.numel() == 0:
            return torch.zeros(target_len)
        if feat.size(-1) == target_len:
            return feat
        if feat.size(-1) > target_len:
            return feat[..., :target_len]
        pad = target_len - feat.size(-1)
        return torch.nn.functional.pad(feat, (0, pad))

    def extract_all_features(
        self,
        audio: str | Path | np.ndarray | torch.Tensor,
        rule: TajweedRule | None,
    ) -> Dict[str, torch.Tensor | float | TajweedRule | None]:
        if isinstance(audio, (str, Path)):
            wav = self.load_audio(audio)
        else:
            wav = self._ensure_tensor(audio)

        duration = wav.numel() / float(self.cfg.sample_rate)

        mel = self.melspec(wav.unsqueeze(0))
        mel = self.amptodb(mel).squeeze(0)

        mfcc = self.mfcc(wav.unsqueeze(0)).squeeze(0)
        delta = torchaudio.functional.compute_deltas(mfcc)
        delta2 = torchaudio.functional.compute_deltas(delta)
        mfcc_delta = torch.cat([mfcc, delta, delta2], dim=0)

        pitch = torchaudio.functional.detect_pitch_frequency(
            wav,
            self.cfg.sample_rate,
            win_length=self.cfg.win_length(),
            hop_length=self.cfg.hop_length(),
        )
        energy = self._frame_rms(wav)

        frame_len = mfcc.size(-1)
        pitch = self._align_feature_length(pitch, frame_len)
        energy = self._align_feature_length(energy, frame_len)

        all_features = torch.cat([
            mfcc_delta,
            pitch.unsqueeze(0),
            energy.unsqueeze(0),
        ], dim=0)

        return {
            "rule": rule,
            "duration": duration,
            "mel_spectrogram": mel,
            "mfcc": mfcc,
            "mfcc_delta": mfcc_delta,
            "mfcc_delta2": delta2,
            "pitch": pitch,
            "energy": energy,
            "all_features": all_features,
        }
