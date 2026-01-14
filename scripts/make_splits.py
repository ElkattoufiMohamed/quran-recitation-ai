import argparse
import random
from pathlib import Path
from src.qrec.data.manifests import read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    args = ap.parse_args()

    items = read_jsonl(args.in_manifest)
    random.seed(args.seed)
    random.shuffle(items)

    n = len(items)
    n_train = int(n * args.train_ratio)
    n_dev = int(n * args.dev_ratio)
    train = items[:n_train]
    dev = items[n_train:n_train + n_dev]
    test = items[n_train + n_dev:]

    out = Path(args.out_dir)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "dev.jsonl", dev)
    write_jsonl(out / "test.jsonl", test)

    print(f"✅ Wrote: {len(train)} train, {len(dev)} dev, {len(test)} test to {out}")


if __name__ == "__main__":
    main()
