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


def _fill(s, start, end, placeholder, body):
    block = f"{start}\n{body.rstrip()}\n{end}"
    if start in s and end in s:
        return s[:s.index(start)] + block + s[s.index(end) + len(end):]
    return s.replace(placeholder, block)


def main():
    readme = ROOT / "README.md"
    s = readme.read_text()
    if len(sys.argv) > 1:
        s = _fill(s, "<!-- results:start -->", "<!-- results:end -->", "RESULTS_TABLE", table(sys.argv[1:]))
    bench = ROOT / "results/bench.md"
    if bench.exists():
        s = _fill(s, "<!-- bench:start -->", "<!-- bench:end -->", "BENCH_TABLE", bench.read_text())
    readme.write_text(s)
    print("README updated")


if __name__ == "__main__":
    main()
