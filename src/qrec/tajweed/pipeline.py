from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .detectors import (
    AlMadDetector,
    GhunnahDetector,
    QalqalahDetector,
    IdghamDetector,
    IkhfaaDetector,
)
from .rules import TajweedRule


class TajweedVerificationPipeline:
    def __init__(
        self,
        al_mad_model: Optional[str] = None,
        ghunnah_model: Optional[str] = None,
        qalqalah_model: Optional[str] = None,
        idgham_model: Optional[str] = None,
        ikhfaa_model: Optional[str] = None,
    ) -> None:
        self.detectors = {
            TajweedRule.AL_MAD: AlMadDetector(al_mad_model),
            TajweedRule.GHUNNAH: GhunnahDetector(ghunnah_model),
            TajweedRule.QALQALAH: QalqalahDetector(qalqalah_model),
            TajweedRule.IDGHAM: IdghamDetector(idgham_model),
            TajweedRule.IKHFAA: IkhfaaDetector(ikhfaa_model),
        }

    def verify_rule(
        self,
        rule: TajweedRule,
        audio: np.ndarray | torch.Tensor,
        **kwargs,
    ) -> Dict:
        detector = self.detectors[rule]
        return detector.detect(audio, **kwargs)

    def verify_all(self, audio: np.ndarray | torch.Tensor) -> Dict[TajweedRule, Dict]:
        return {rule: detector.detect(audio) for rule, detector in self.detectors.items()}
