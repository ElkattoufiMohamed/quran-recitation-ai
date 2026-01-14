import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def train_asr(model, dataloader, optimizer, ctc_loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="train_asr", leave=False):
        mels, mel_lens, targets, tgt_lens = batch
        mels = mels.to(device)
        mel_lens = mel_lens.to(device)
        targets = targets.to(device)
        tgt_lens = tgt_lens.to(device)

        optimizer.zero_grad(set_to_none=True)
        log_probs, out_lens = model(mels, mel_lens)
        loss = ctc_loss_fn(log_probs, targets, out_lens, tgt_lens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))
