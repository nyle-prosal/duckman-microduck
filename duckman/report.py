"""Fill the README results table from results/*.json.  python -m duckman.report results/learned_seed0.json ..."""
import json
import sys
from pathlib import Path
from .constants import ROOT

HEADER = ("| Policy | Score | Coins | Pellets | Ghosts caught | Lives lost | End | Time | Falls |\n"
          "|---|---|---|---|---|---|---|---|---|\n")


def table(paths):
    rows = []
    for p in paths:
        r = json.load(open(p))
        rows.append(f"| {r['labels']['P_']} | **{r['score']}** | {r['coins']} | {r['pellets']} | {r['ghosts']} | "
                    f"{r['lives_lost']} | {r['end']} | {r['t']} s | {r.get('falls', 0)} |")
    return HEADER + "\n".join(rows) + "\n"


def main():
    t = table(sys.argv[1:])
    readme = ROOT / "README.md"
    s = readme.read_text()
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    block = f"{start}\n{t}{end}"
    if start in s:
        s = s[:s.index(start)] + block + s[s.index(end) + len(end):]
    else:
        s = s.replace("RESULTS_TABLE", block)
    readme.write_text(s)
    print(t)


if __name__ == "__main__":
    main()
