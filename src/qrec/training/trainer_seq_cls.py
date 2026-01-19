import torch
from tqdm import tqdm


def train_sequence_classifier(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for feats, y in tqdm(dataloader, desc="train_seq", leave=False):
        feats = feats.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(feats)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = (torch.sigmoid(logits) > 0.5).float()
        correct += (pred == y).sum().item()
        n += y.numel()

    acc = correct / max(1, n)
    return total_loss / max(1, len(dataloader)), acc


@torch.no_grad()
def eval_sequence_classifier(model, dataloader, device):
    model.eval()
    correct = 0
    n = 0
    for feats, y in dataloader:
        feats = feats.to(device)
        y = y.to(device)
        logits = model(feats)
        pred = (torch.sigmoid(logits) > 0.5).float()
        correct += (pred == y).sum().item()
        n += y.numel()
    return correct / max(1, n)
