import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from src.qrec.utils.seed import set_seed
from src.qrec.utils.audio import MelSpecConfig, AudioFeaturizer
from src.qrec.data.tokenizer import CharTokenizer
from src.qrec.data.asr_dataset import ASRDataset, asr_collate
from src.qrec.models.asr_cnn_lstm_ctc import CNNLSTMCTC
from src.qrec.utils.logging import make_run_dir, write_metrics
from src.qrec.training.trainer_asr import train_asr
from src.qrec.inference.decode_ctc import greedy_ctc_decode
from src.qrec.utils.metrics import wer


@torch.no_grad()
def eval_asr(model, dataloader, device, tokenizer, blank_id: int):
    model.eval()
    wers = []
    for mels, mel_lens, targets_cat, tgt_lens in dataloader:
        mels = mels.to(device)
        mel_lens = mel_lens.to(device)

        log_probs, out_lens = model(mels, mel_lens)
        pred_ids = greedy_ctc_decode(log_probs.cpu(), blank_id=blank_id)

        # rebuild refs per sample from concatenated targets
        refs = []
        offset = 0
        for L in tgt_lens.tolist():
            ref_ids = targets_cat[offset:offset+L].tolist()
            refs.append(ref_ids)
            offset += L

        for ref_ids, hyp_ids in zip(refs, pred_ids):
            ref_txt = tokenizer.decode(ref_ids)
            hyp_txt = tokenizer.decode(hyp_ids)
            # WER expects word tokens; for Arabic text you can split on spaces if you keep them
            ref_words = ref_txt.split()
            hyp_words = hyp_txt.split()
            wers.append(wer(ref_words, hyp_words))
    return sum(wers) / max(1, len(wers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/asr.yaml")
    ap.add_argument("--train_manifest", type=str, required=True)
    ap.add_argument("--dev_manifest", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, default="data/processed/tokenizer.json")
    ap.add_argument("--cache_dir", type=str, default="data/interim/mels_asr")
    ap.add_argument("--runs_dir", type=str, default="runs/asr")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    melcfg = MelSpecConfig(
        sample_rate=cfg["sample_rate"],
        n_mels=cfg["n_mels"],
        win_length_ms=cfg["win_length_ms"],
        hop_length_ms=cfg["hop_length_ms"],
    )
    featurizer = AudioFeaturizer(melcfg)

    # Build or load tokenizer based on TRAIN texts
    tok_path = Path(args.tokenizer_path)
    if tok_path.exists():
        tokenizer = CharTokenizer.load(tok_path)
    else:
        import json
        from src.qrec.data.manifests import read_jsonl
        train_items = read_jsonl(args.train_manifest)
        texts = [it["text"] for it in train_items]
        tokenizer = CharTokenizer.build_from_texts(texts)
        tokenizer.save(tok_path)

    train_ds = ASRDataset(args.train_manifest, tokenizer, featurizer, args.cache_dir)
    dev_ds = ASRDataset(args.dev_manifest, tokenizer, featurizer, args.cache_dir)

    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=asr_collate, num_workers=2)
    dev_dl = DataLoader(dev_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=asr_collate, num_workers=2)

    model = CNNLSTMCTC(
        vocab_size=len(tokenizer.char2id),
        n_mels=cfg["n_mels"],
        cnn_channels=tuple(cfg["model"]["cnn_channels"]),
        lstm_hidden=cfg["model"]["lstm_hidden"],
        lstm_layers=cfg["model"]["lstm_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    ctc_loss = torch.nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    run_dir = make_run_dir(args.runs_dir, "asr")
    best = 1e9

    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        tr_loss = train_asr(model, train_dl, optimizer, ctc_loss, device)
        dv_wer = eval_asr(model, dev_dl, device, tokenizer, blank_id=tokenizer.blank_id)

        metrics = {"epoch": epoch, "train_loss": tr_loss, "dev_wer": dv_wer}
        write_metrics(run_dir / "metrics.jsonl", metrics)
        print(metrics)

        if dv_wer < best:
            best = dv_wer
            torch.save({"model": model.state_dict(), "tokenizer": tokenizer.char2id}, run_dir / "best.pt")

    print(f"✅ Done. Best dev WER={best:.4f}. Saved to {run_dir}/best.pt")


if __name__ == "__main__":
    main()
