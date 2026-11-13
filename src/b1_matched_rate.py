"""B1 (sprint, 13 Nov): camp disparity of detector alerts at matched overall alert rates.
Sweeps in-domain budgets (dev organic quantiles, strict >) and compares the transferred 'tea' point at the same
overall organic alert rate. Detector only, no LLM calls. -> data/politics/b1_matched_rate.json"""
import json
import numpy as np
from common import DATA, D
import run_politics as R

P = DATA / "politics"
BUD = [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3]


def supporters(org, cl, name):
    out = set()
    for m in org:
        if R.stance_of(m, cl, name) > 0: out.add(m["account"])
    return out


res = {}
dev_org, dev_cl = R.load_win("dev")
_, d4dev, d4cdev = R.scores(dev_org, dev_cl)
sc_dev = sorted(d4dev)
thr = {b: (float(np.quantile([d4dev[a] for a in sc_dev], 1 - b)), float(np.quantile([d4cdev[a] for a in sc_dev], 1 - b))) for b in BUD}
thr["tea"] = (R.TEA_THR["D4"], R.TEA_THR["D4c"])
for win in ("dev", "heldout"):
    org, cl = R.load_win(win)
    scored, d4, d4c = R.scores(org, cl)
    sup = {n: supporters(org, cl, n) & set(scored) for n in ("沈伯洋", "蔣萬安")}
    rows = []
    for b, (t4, t4c) in thr.items():
        fl = {a for a in scored if d4[a] > t4 or d4c[a] > t4c}
        r = dict(op=str(b), alert=len(fl) / len(scored),
                 shen=len(fl & sup["沈伯洋"]) / len(sup["沈伯洋"]), chiang=len(fl & sup["蔣萬安"]) / len(sup["蔣萬安"]),
                 n_scored=len(scored), n_shen=len(sup["沈伯洋"]), n_chiang=len(sup["蔣萬安"]))
        r["ratio"] = r["shen"] / r["chiang"] if r["chiang"] else None
        rows.append(r)
    tea = next(r for r in rows if r["op"] == "tea")
    sw = sorted([r for r in rows if r["op"] != "tea"], key=lambda r: r["alert"])
    xs = [r["alert"] for r in sw]
    tea["matched_in_domain"] = {k: float(np.interp(tea["alert"], xs, [r[k] for r in sw])) for k in ("shen", "chiang")}
    res[win] = rows
    print(f"== {win}: scored {len(scored)}, supporters 沈 {len(sup['沈伯洋'])} 蔣 {len(sup['蔣萬安'])}")
    for r in rows:
        print(f"  {r['op']:>7} alert {r['alert']:.3f}  沈 {r['shen']:.3f}  蔣 {r['chiang']:.3f}  ratio {r['ratio'] if r['ratio'] is None else round(r['ratio'],2)}")
    m = tea["matched_in_domain"]
    print(f"  tea at alert {tea['alert']:.3f}: 沈 {tea['shen']:.3f} 蔣 {tea['chiang']:.3f} | in-domain at same alert: 沈 {m['shen']:.3f} 蔣 {m['chiang']:.3f} ratio {m['shen']/m['chiang']:.2f}")
json.dump(res, open(P / "b1_matched_rate.json", "w"), ensure_ascii=False, indent=1)
