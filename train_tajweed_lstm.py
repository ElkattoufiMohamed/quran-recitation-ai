import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.qrec.utils.seed import set_seed
from src.qrec.tajweed.features import TajweedFeatureConfig
from src.qrec.data.tajweed_feature_dataset import (
    TajweedFeatureCacheConfig,
    TajweedFeatureDataset,
    tajweed_feature_collate,
)
from src.qrec.models.tajweed_lstm import TajweedRuleLSTM
from src.qrec.training.trainer_seq_cls import train_sequence_classifier, eval_sequence_classifier
from src.qrec.utils.logging import make_run_dir, write_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/tajweed_lstm.yaml")
    ap.add_argument("--train_manifest", type=str, required=True)
    ap.add_argument("--dev_manifest", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, default="data/interim/mfcc_delta")
    ap.add_argument("--runs_dir", type=str, default="runs/tajweed_lstm")
    ap.add_argument("--rule_name", type=str, default="tajweed_rule")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    feat_cfg = TajweedFeatureConfig(
        sample_rate=cfg["sample_rate"],
        n_mels=cfg["n_mels"],
        n_mfcc=cfg["n_mfcc"],
    )

    cache_cfg = TajweedFeatureCacheConfig(args.cache_dir)

    train_ds = TajweedFeatureDataset(args.train_manifest, feat_cfg, cache_cfg, label_key="label")
    dev_ds = TajweedFeatureDataset(args.dev_manifest, feat_cfg, cache_cfg, label_key="label")

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=tajweed_feature_collate,
        num_workers=2,
    )
    dev_dl = DataLoader(
        dev_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=tajweed_feature_collate,
        num_workers=2,
    )

    input_dim = cfg["n_mfcc"] * 3
    model = TajweedRuleLSTM(
        input_dim=input_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    run_dir = make_run_dir(args.runs_dir, args.rule_name)
    best = 0.0

    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        tr_loss, tr_acc = train_sequence_classifier(model, train_dl, optimizer, loss_fn, device)
        dv_acc = eval_sequence_classifier(model, dev_dl, device)

        metrics = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "dev_acc": dv_acc}
        write_metrics(run_dir / "metrics.jsonl", metrics)
        print(metrics)

        if dv_acc > best:
            best = dv_acc
            torch.save({"model": model.state_dict()}, run_dir / "best.pt")

    print(f"✅ Done. Best dev acc={best:.4f}. Saved to {run_dir}/best.pt")


if __name__ == "__main__":
    main()
