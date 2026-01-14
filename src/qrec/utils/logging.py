from __future__ import annotations
from pathlib import Path
import json
import time


def make_run_dir(root: str | Path, prefix: str) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run = root / f"{prefix}-{ts}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def write_metrics(path: str | Path, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")
 