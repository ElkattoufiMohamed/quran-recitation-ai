import argparse
from src.qrec.data.manifests import read_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--kind", type=str, choices=["asr", "segment"], required=True)
    args = ap.parse_args()

    items = read_jsonl(args.manifest)
    assert len(items) > 0, "Empty manifest"

    if args.kind == "asr":
        for it in items[:10]:
            assert "audio_path" in it and "text" in it
    else:
        for it in items[:10]:
            assert "segment_path" in it and "label" in it

    print("✅ Sanity check passed")

if __name__ == "__main__":
    main()
