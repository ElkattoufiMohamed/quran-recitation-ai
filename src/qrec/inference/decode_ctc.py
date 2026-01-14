import torch

def greedy_ctc_decode(log_probs, blank_id: int):
    """
    log_probs: (T, B, V)
    returns: list[list[int]] token ids per batch.
    """
    pred = log_probs.argmax(dim=-1)  # (T,B)
    pred = pred.transpose(0, 1)      # (B,T)
    out = []
    for seq in pred:
        collapsed = []
        prev = None
        for t in seq.tolist():
            if t != blank_id and t != prev:
                collapsed.append(t)
            prev = t
        out.append(collapsed)
    return out
