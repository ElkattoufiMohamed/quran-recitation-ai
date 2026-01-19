from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from ..models.tajweed_cnn import TajweedCNN
from ..models.tajweed_lstm import TajweedRuleLSTM
from .features import TajweedFeatureExtractor
from .rules import TajweedRule


class AlMadDetector:
    """
    Al-Mad (Prolongation) Rule Detector.

    Detects if prolongation is too short/too long and checks model confidence.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = TajweedRuleLSTM(input_dim=39)
        if model_path:
            state = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.eval()

        self.feature_extractor = TajweedFeatureExtractor()
        self.mad_durations = {
            "asli": 0.4,
            "far3i": 0.8,
            "lazim": 1.0,
        }

    def detect(self, audio: np.ndarray | torch.Tensor, mad_type: str = "asli") -> Dict:
        features = self.feature_extractor.extract_all_features(audio, TajweedRule.AL_MAD)
        mfcc_delta = features["mfcc_delta"]
        x = torch.FloatTensor(mfcc_delta.T).unsqueeze(0)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()

        detected_duration = float(features["duration"])
        expected_duration = self.mad_durations.get(mad_type, 0.4)
        duration_tolerance = 0.15
        duration_correct = abs(detected_duration - expected_duration) < duration_tolerance

        is_correct = (prob > 0.5) and duration_correct

        return {
            "is_correct": is_correct,
            "confidence": prob,
            "detected_duration": detected_duration,
            "expected_duration": expected_duration,
            "duration_error": abs(detected_duration - expected_duration),
            "rule": "Al-Mad",
            "error_type": self._classify_error(detected_duration, expected_duration),
        }

    def _classify_error(self, detected: float, expected: float) -> Optional[str]:
        if abs(detected - expected) < 0.15:
            return None
        if detected < expected:
            return "too_short"
        return "too_long"


class GhunnahDetector:
    """
    Ghunnah (Nasalization) Rule Detector.

    Uses LSTM confidence + duration + nasal energy ratio.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = TajweedRuleLSTM(input_dim=39)
        if model_path:
            state = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.eval()

        self.feature_extractor = TajweedFeatureExtractor()
        self.expected_duration = 0.4

    def detect(self, audio: np.ndarray | torch.Tensor) -> Dict:
        features = self.feature_extractor.extract_all_features(audio, TajweedRule.GHUNNAH)
        mfcc_delta = features["mfcc_delta"]
        x = torch.FloatTensor(mfcc_delta.T).unsqueeze(0)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()

        detected_duration = float(features["duration"])
        duration_correct = abs(detected_duration - self.expected_duration) < 0.15
        nasality_score = self._check_nasality(audio)

        is_correct = (prob > 0.5) and duration_correct and (nasality_score > 0.6)

        return {
            "is_correct": is_correct,
            "confidence": prob,
            "detected_duration": detected_duration,
            "expected_duration": self.expected_duration,
            "nasality_score": nasality_score,
            "rule": "Ghunnah",
            "error_type": self._classify_error(prob, duration_correct, nasality_score > 0.6),
        }

    def _check_nasality(self, audio: np.ndarray | torch.Tensor) -> float:
        wav = audio
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
        if wav.ndim > 1:
            wav = np.mean(wav, axis=0)
        fft = np.fft.rfft(wav)
        freqs = np.fft.rfftfreq(len(wav), 1 / self.feature_extractor.cfg.sample_rate)
        nasal_range = (freqs >= 200) & (freqs <= 300)
        nasal_energy = np.sum(np.abs(fft[nasal_range]))
        total_energy = np.sum(np.abs(fft))
        return float(nasal_energy / (total_energy + 1e-10))

    def _classify_error(self, prob: float, duration_ok: bool, nasality_ok: bool) -> Optional[str]:
        if prob > 0.5 and duration_ok and nasality_ok:
            return None
        if not nasality_ok:
            return "insufficient_nasalization"
        if not duration_ok:
            return "wrong_duration"
        return "general_error"


class QalqalahDetector:
    """
    Qalqalah (Echo/Bounce) Rule Detector.

    Uses CNN on mel-spectrogram and checks for late energy burst.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = TajweedCNN(num_classes=2)
        if model_path:
            state = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.eval()

        self.feature_extractor = TajweedFeatureExtractor()

    def detect(self, audio: np.ndarray | torch.Tensor) -> Dict:
        features = self.feature_extractor.extract_all_features(audio, TajweedRule.QALQALAH)
        mel_spec = features["mel_spectrogram"]
        x = torch.FloatTensor(mel_spec).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=-1)[:, 1].item()

        energy = features["energy"]
        has_burst = self._detect_energy_burst(energy)
        is_correct = (prob > 0.5) and has_burst

        return {
            "is_correct": is_correct,
            "confidence": prob,
            "has_energy_burst": has_burst,
            "rule": "Qalqalah",
            "error_type": None if is_correct else "missing_bounce",
        }

    def _detect_energy_burst(self, energy: torch.Tensor, threshold: float = 0.7) -> bool:
        if energy.numel() < 3:
            return False
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-10)
        tail_start = int(len(energy_norm) * 0.7)
        tail_energy = energy_norm[tail_start:]
        return bool(tail_energy.max().item() > threshold)


class IdghamDetector:
    """
    Idgham (Merging) Rule Detector.

    Uses LSTM on MFCC deltas and checks for smooth transition energy.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = TajweedRuleLSTM(input_dim=39)
        if model_path:
            state = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.eval()
        self.feature_extractor = TajweedFeatureExtractor()

    def detect(self, audio: np.ndarray | torch.Tensor) -> Dict:
        features = self.feature_extractor.extract_all_features(audio, TajweedRule.IDGHAM)
        mfcc_delta = features["mfcc_delta"]
        x = torch.FloatTensor(mfcc_delta.T).unsqueeze(0)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()

        energy = features["energy"]
        smooth_transition = self._is_smooth_transition(energy)
        is_correct = (prob > 0.5) and smooth_transition

        return {
            "is_correct": is_correct,
            "confidence": prob,
            "smooth_transition": smooth_transition,
            "rule": "Idgham",
            "error_type": None if is_correct else "merge_error",
        }

    def _is_smooth_transition(self, energy: torch.Tensor, threshold: float = 0.25) -> bool:
        if energy.numel() < 3:
            return False
        diffs = torch.abs(energy[1:] - energy[:-1])
        return bool(torch.median(diffs).item() < threshold)


class IkhfaaDetector:
    """
    Ikhfaa (Concealment) Rule Detector.

    Uses LSTM confidence + nasal energy presence.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = TajweedRuleLSTM(input_dim=39)
        if model_path:
            state = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.eval()
        self.feature_extractor = TajweedFeatureExtractor()

    def detect(self, audio: np.ndarray | torch.Tensor) -> Dict:
        features = self.feature_extractor.extract_all_features(audio, TajweedRule.IKHFAA)
        mfcc_delta = features["mfcc_delta"]
        x = torch.FloatTensor(mfcc_delta.T).unsqueeze(0)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()

        nasality_score = self._check_nasality(audio)
        is_correct = (prob > 0.5) and (nasality_score > 0.5)

        return {
            "is_correct": is_correct,
            "confidence": prob,
            "nasality_score": nasality_score,
            "rule": "Ikhfaa",
            "error_type": None if is_correct else "concealment_error",
        }

    def _check_nasality(self, audio: np.ndarray | torch.Tensor) -> float:
        wav = audio
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
        if wav.ndim > 1:
            wav = np.mean(wav, axis=0)
        fft = np.fft.rfft(wav)
        freqs = np.fft.rfftfreq(len(wav), 1 / self.feature_extractor.cfg.sample_rate)
        nasal_range = (freqs >= 200) & (freqs <= 400)
        nasal_energy = np.sum(np.abs(fft[nasal_range]))
        total_energy = np.sum(np.abs(fft))
        return float(nasal_energy / (total_energy + 1e-10))
