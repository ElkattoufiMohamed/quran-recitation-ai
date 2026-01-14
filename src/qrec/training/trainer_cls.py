import torch
from tqdm import tqdm


def train_classifier(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for batch in tqdm(dataloader, desc="train_cls", leave=False):
        mels, y = batch
        mels = mels.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(mels)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        n += y.numel()

    acc = correct / max(1, n)
    return total_loss / max(1, len(dataloader)), acc
