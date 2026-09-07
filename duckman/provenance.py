"""Write assets/PROVENANCE.json: sha256, size and upstream origin for every asset and checkpoint we ship."""
import hashlib
import json
from datetime import date
from pathlib import Path
from .constants import ROOT

UPSTREAM = {
    "assets/microduck/": ("https://github.com/pollen-robotics/microduck_rl/tree/29e887ecfbf5d37144759e5a9f8a176dfb83d547/src/mjlab_microduck/robot/microduck",
                          "pollen-robotics/microduck_rl @ 29e887e", "2026-09-03", "Apache-2.0 (XML), CC BY-SA-NC 4.0 (meshes)"),
    "assets/policies/alpha_": ("https://huggingface.co/pollen-robotics/microduck-policies/tree/main", "pollen-robotics/microduck-policies, 2026-09-02 revision", "2026-09-03", "Apache-2.0"),
    "assets/policies/standup.onnx": ("trained for this entry: Mjlab-StandUp-Flat-MicroDuck, microduck_rl @ 29e887e, HIM Arena RTX 4090, exported with scripts/export.py from model_4499.pt",
                                     "this entry", "2026-09-04", "Apache-2.0"),
    "checkpoints/": ("trained for this entry: duckman/imitate.py then duckman/train_es.py (run4, generation 20 elite)", "this entry", "2026-09-04", "Apache-2.0"),
    "vendor/": ("https://github.com/Rhoban/bam/tree/62bd8ce12154340be97e06f7f41a0ca8f116d967", "Rhoban/bam @ 62bd8ce (branch mjlab_frictionloss), built with pip wheel", "2026-09-07", "MIT"),
}


def origin(rel):
    for k, v in UPSTREAM.items():
        if rel.startswith(k):
            return v
    return ("this entry", "this entry", str(date.today()), "Apache-2.0")


def main():
    out = []
    for folder in ("assets", "checkpoints", "vendor"):
        for f in sorted((ROOT / folder).rglob("*")):
            if f.is_file() and f.suffix != ".json":
                rel = f.relative_to(ROOT).as_posix()
                url, rev, when, lic = origin(rel)
                out.append({"path": rel, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size,
                            "upstream": url, "revision": rev, "obtained": when, "license": lic})
    (ROOT / "assets/PROVENANCE.json").write_text(json.dumps(out, indent=1))
    print(f"wrote assets/PROVENANCE.json with {len(out)} entries")


if __name__ == "__main__":
    main()
