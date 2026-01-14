import torch
import torch.nn as nn
import torch.nn.functional as F
from .asr_cnn_lstm_ctc import CNNLSTMCTC


class ASRFeatureExtractor(nn.Module):
    """
    Wrap CNNLSTMCTC but return hidden states instead of CTC logits.
    """
    def __init__(self, asr: CNNLSTMCTC):
        super().__init__()
        self.sub = asr.sub
        self.lstm = asr.lstm

    def forward(self, mels: torch.Tensor, mel_lens: torch.Tensor):
        x = mels.unsqueeze(1)
        x = self.sub(x)
        B, C, Freq, Tp = x.shape
        x = x.permute(3, 0, 1, 2).contiguous().view(Tp, B, C * Freq)

        out_lens = (mel_lens + 1) // 2
        out_lens = (out_lens + 1) // 2
        out_lens = torch.clamp(out_lens, min=1)

        h, _ = self.lstm(x)  # (T', B, 2H)
        return h, out_lens


class MakhrajDetector(nn.Module):
    """
    Articulation classifier from ASR hidden states.
    Assumes you have segment-level labels (or per-phoneme later).
    """
    def __init__(self, feature_extractor: ASRFeatureExtractor, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.feat = feature_extractor
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, mels: torch.Tensor, mel_lens: torch.Tensor):
        h, _ = self.feat(mels, mel_lens)       # (T', B, 2H)
        h = h.mean(dim=0)                      # mean over time -> (B, 2H)
        h = self.dropout(h)
        return self.head(h)                    # (B, num_classes)
