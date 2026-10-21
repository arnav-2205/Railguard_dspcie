"""Download the 20 Rail-VIVID run CSVs (~1.9 GB) from Hugging Face.
Skips files already present with the right size. Usage: python scripts/download_data.py"""
import json, sys, urllib.request
from pathlib import Path

REPO = "saluslab/Rail-VIVID"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "runs"


def main():
    top = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/tree/main"))
    runs = [e["path"] for e in top if e["type"] == "directory" and e["path"][:4] in ("AtoB", "BtoA")]
    for run in sorted(runs):
        csv = f"{run}/{run}.csv"
        info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/tree/main/{run}"))
        size = next(f["size"] for f in info if f["path"] == csv)
        dst = OUT / run / f"{run}.csv"
        if dst.exists() and dst.stat().st_size == size:
            print("have", run); continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print("get ", run, f"{size/1e6:.0f} MB", flush=True)
        urllib.request.urlretrieve(f"https://huggingface.co/datasets/{REPO}/resolve/main/{csv}", dst)
        assert dst.stat().st_size == size, f"size mismatch {run}"


if __name__ == "__main__":
    sys.exit(main())
