"""Ghost League: {learned, planner} Duck-Men x {scripted, learned} ghosts on held-out seeds.
python -m duckman.league --seeds 20 --workers 6"""
import argparse
import json
from multiprocessing import Pool
import numpy as np
from .constants import ROOT
from .eval import run_eval


def _one(args):
    pac, ghosts, seed, ck = args
    if pac == "planner":
        r = run_eval("planner", seed, None, ghosts=ghosts)
    else:
        r = run_eval("learned", seed, ck, ghosts=ghosts)
    return {"pac": pac, "ghosts": ghosts, "seed": seed, "score": r["score"], "coins": r["coins"], "caught": r["ghosts"], "lives_lost": r["lives_lost"], "end": r["end"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pac2", help="optional second Duck-Man checkpoint (e.g. round 2, trained against mixed ghosts)")
    a = ap.parse_args()
    pacs = [("learned", str(ROOT / "checkpoints/strategy_final.npz")), ("planner", None)]
    if a.pac2:
        pacs.insert(1, ("learned2", a.pac2))
    jobs = [(p, g, s, ck) for p, ck in pacs for g in ("scripted", "learned") for s in range(a.seeds)]
    with Pool(a.workers) as pool:
        rows = pool.map(_one, jobs, chunksize=1)
    json.dump(rows, open(ROOT / "results/league.json", "w"), indent=1)
    lines = ["| Duck-Man \\ ghosts | scripted ghosts (arcade personalities) | learned ghosts (co-evolved vs the learned Duck-Man) |", "|---|---|---|"]
    names = {"learned": "learned strategy, round 1 (trained vs scripted ghosts)", "learned2": "learned strategy, round 2 (trained vs scripted + learned ghosts)", "planner": "scripted planner"}
    for p, _ in pacs:
        name = names[p]
        cells = []
        for g in ("scripted", "learned"):
            rs = [r for r in rows if r["pac"] == p and r["ghosts"] == g]
            cells.append(f"**{np.mean([r['score'] for r in rs]):.0f}** ± {np.std([r['score'] for r in rs]):.0f} (tags {np.mean([r['lives_lost'] for r in rs]):.2f}, catches {np.mean([r['caught'] for r in rs]):.2f})")
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    t = "\n".join(lines) + f"\n\nMean Duck-Man score ± std over held-out seeds 0-{a.seeds - 1}; tags = lives lost per round; catches = ghosts caught per round.\n"
    (ROOT / "results/league.md").write_text(t)
    print(t)


if __name__ == "__main__":
    main()
