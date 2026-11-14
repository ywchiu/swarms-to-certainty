"""B1 mechanism (sprint): D4c tie blocks per window, with camp purity. Detector only, no LLM calls.

For each window, lists the largest groups of scored organic accounts that share one D4c value, and how many of
each group support each front-runner (>=1 supportive message). -> data/politics/b1_tie_blocks.json (aggregates only).

  python b1_tie_blocks.py
"""
import json
from collections import Counter

import run_politics as R

NAMES = ("沈伯洋", "蔣萬安")  # challenger, incumbent
TOP = 4

out = {}
ops = R.operating_points()
for win in ("dev", "heldout"):
    org, cl = R.load_win(win)
    scored, d4, d4c = R.scores(org, cl)
    sup = {n: set() for n in NAMES}
    for m in org:
        for n in NAMES:
            if m["account"] in d4 and R.stance_of(m, cl, n) > 0:
                sup[n].add(m["account"])
    vals = Counter(round(d4c[a], 6) for a in scored)
    blocks = []
    for v, size in vals.most_common(TOP):
        members = {a for a in scored if round(d4c[a], 6) == v}
        blocks.append(dict(d4c=round(v, 1), size=size,
                           challenger_supporters=len(members & sup["沈伯洋"]),
                           incumbent_supporters=len(members & sup["蔣萬安"])))
    out[win] = dict(n_scored=len(scored), tea_D4c=round(ops["tea"]["D4c"], 1), blocks=blocks)
    print(f"== {win}: scored {len(scored)}, transferred tea D4c threshold {ops['tea']['D4c']:.1f}")
    for b in blocks:
        print(f"  D4c {b['d4c']:>7}: {b['size']:>4} accounts | challenger supporters {b['challenger_supporters']}, "
              f"incumbent supporters {b['incumbent_supporters']}")
json.dump(out, open(R.P / "b1_tie_blocks.json", "w"), ensure_ascii=False, indent=1)
