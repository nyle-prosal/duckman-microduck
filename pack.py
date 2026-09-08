"""Build submission.zip with source, small assets, pins, tests, README, licences and the inference checkpoint only."""
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INCLUDE = ["duckman", "tests", "assets", "checkpoints", "training", "results", "evidence", "vendor", "examples_api", "docs/hero.png", "SUBMISSION.md", "run.sh",
           "requirements.txt", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "pack.py"]
EXCLUDE_DIRS = {"__pycache__", ".venv", "runs", ".git", "spikes", "examples", "microduck_rl", "microduck"}
EXCLUDE_SUFFIX = {".mp4", ".pyc", ".log"}


def files():
    for item in INCLUDE:
        p = ROOT / item
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                rel = f.relative_to(ROOT).parts
                if rel[0] == "results" and f.suffix not in (".json", ".md"):
                    continue                     # measurements only; preview frames and videos stay out
                if f.is_file() and rel[0] not in EXCLUDE_DIRS and "__pycache__" not in rel and f.suffix not in EXCLUDE_SUFFIX:
                    yield f


def main():
    out = ROOT / "submission.zip"
    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files():
            z.write(f, f.relative_to(ROOT))
            total += f.stat().st_size
            print(f"{f.stat().st_size / 1024:9.1f} KB  {f.relative_to(ROOT)}")
    print(f"\n{len(list(files()))} files, {total / 1e6:.1f} MB uncompressed, {out.stat().st_size / 1e6:.1f} MB zipped -> {out}")
    assert out.stat().st_size < 40e6, "zip too large"


if __name__ == "__main__":
    main()
