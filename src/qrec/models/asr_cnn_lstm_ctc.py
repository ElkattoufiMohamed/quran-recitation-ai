import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSubsampler(nn.Module):
    """
    Simple 2-layer CNN subsampler over (B, 1, n_mels, T) -> (B, C, n_mels', T')
    """
    def __init__(self, channels=(32, 64)):
        super().__init__()
        c1, c2 = channels
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, stride=(2, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=(2, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class CNNLSTMCTC(nn.Module):
    """
    CNN -> flatten freq -> BiLSTM -> linear logits for CTC.
    """
    def __init__(self, vocab_size: int, n_mels: int = 80,
                 cnn_channels=(32, 64),
                 lstm_hidden: int = 512,
                 lstm_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.sub = ConvSubsampler(cnn_channels)
        c_out = cnn_channels[-1]

        # after 2 strides (2,2) twice => n_mels // 4 (approx), T // 4 (approx)
        n_mels_out = (n_mels + 1) // 2
        n_mels_out = (n_mels_out + 1) // 2  # rough for padding/stride
        feat_dim = c_out * n_mels_out

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=False,
        )
        self.proj = nn.Linear(2 * lstm_hidden, vocab_size)

    def forward(self, mels: torch.Tensor, mel_lens: torch.Tensor):
        """
        mels: (B, n_mels, T)
        mel_lens: (B,) lengths in frames
        returns:
          log_probs: (T', B, V)
          out_lens: (B,)
        """
        x = mels.unsqueeze(1)  # (B, 1, n_mels, T)
        x = self.sub(x)        # (B, C, n_mels', T')
        B, C, Freq, Tp = x.shape

        # flatten freq
        x = x.permute(3, 0, 1, 2).contiguous()      # (T', B, C, Freq)
        x = x.view(Tp, B, C * Freq)                 # (T', B, feat_dim)

        # update lengths after 2 time subsampling strides
        out_lens = (mel_lens + 1) // 2
        out_lens = (out_lens + 1) // 2
        out_lens = torch.clamp(out_lens, min=1)

        x, _ = self.lstm(x)                         # (T', B, 2H)
        logits = self.proj(x)                       # (T', B, V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, out_lens
