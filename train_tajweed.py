import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from src.qrec.utils.seed import set_seed
from src.qrec.utils.audio import MelSpecConfig, AudioFeaturizer
from src.qrec.data.segment_dataset import SegmentDataset, segment_collate
from src.qrec.models.tajweed_cnn import TajweedCNN
from src.qrec.training.trainer_cls import train_classifier
from src.qrec.utils.logging import make_run_dir, write_metrics


@torch.no_grad()
def eval_classifier(model, dataloader, device):
    model.eval()
    correct = 0
    n = 0
    for mels, y in dataloader:
        mels = mels.to(device)
        y = y.to(device)
        logits = model(mels)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        n += y.numel()
    return correct / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/tajweed.yaml")
    ap.add_argument("--train_manifest", type=str, required=True)
    ap.add_argument("--dev_manifest", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, default="data/interim/mels_segments")
    ap.add_argument("--runs_dir", type=str, default="runs/tajweed")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--rule_name", type=str, default="tajweed_rule")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    melcfg = MelSpecConfig(sample_rate=cfg["sample_rate"], n_mels=cfg["n_mels"])
    featurizer = AudioFeaturizer(melcfg)

    train_ds = SegmentDataset(args.train_manifest, featurizer, args.cache_dir, label_key="label")
    dev_ds = SegmentDataset(args.dev_manifest, featurizer, args.cache_dir, label_key="label")

    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=segment_collate, num_workers=2)
    dev_dl = DataLoader(dev_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=segment_collate, num_workers=2)

    model = TajweedCNN(n_mels=cfg["n_mels"], base_channels=cfg["model"]["base_channels"], num_classes=args.num_classes, dropout=cfg["model"]["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()

    run_dir = make_run_dir(args.runs_dir, args.rule_name)
    best = 0.0

    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        tr_loss, tr_acc = train_classifier(model, train_dl, optimizer, loss_fn, device)
        dv_acc = eval_classifier(model, dev_dl, device)

        metrics = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "dev_acc": dv_acc}
        write_metrics(run_dir / "metrics.jsonl", metrics)
        print(metrics)

        if dv_acc > best:
            best = dv_acc
            torch.save({"model": model.state_dict()}, run_dir / "best.pt")

    print(f"✅ Done. Best dev acc={best:.4f}. Saved to {run_dir}/best.pt")


if __name__ == "__main__":
    main()
