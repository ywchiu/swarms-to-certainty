"""Re-calibrate the frozen alert rule on the ORGANIC-ONLY election corpus (no swarm exists yet): thresholds are
the 99.5th percentile of D4 and D4c over scored dev accounts, exactly as in the prior work (ops.py). The same
thresholds are then applied unchanged to held-out organic data and, from 11/13, to every injected pool.

Pre-sprint discipline: this prints only the thresholds and the overall organic alert rate. Which entities or camps
the flagged accounts endorse is the over-correction outcome and is NOT inspected before kickoff."""
import json
import numpy as np
from common import D, DATA

P = DATA / "politics"
ALERT = 0.005


def pool(win):
    msgs = json.load(open(P / win / "organic_msgs.json")); cl = json.load(open(P / win / "claims.json"))
    acc = D.accounts_of(msgs)
    scored = sorted(a for a, v in acc.items() if len(v) >= D.MIN_MSGS)
    d4, _, _ = D.d4_claim(msgs, acc, scored, cl)
    d4c, _, _ = D.d4_claim(msgs, acc, scored, cl, inherit=True)
    return scored, {"D4": d4, "D4c": d4c}


scored, sc = pool("dev")
thr = {k: float(np.quantile([v[a] for a in scored], 1 - ALERT)) for k, v in sc.items()}
out = {"alert_budget": ALERT, "threshold": thr, "rule": "flag if D4 > thr_D4 or D4c > thr_D4c"}
for win in ("dev", "heldout"):
    s, v = (scored, sc) if win == "dev" else pool(win)
    fl = [a for a in s if v["D4"][a] > thr["D4"] or v["D4c"][a] > thr["D4c"]]
    out[win] = {"scored_accounts": len(s), "flagged": len(fl), "alert_rate": len(fl) / max(len(s), 1),
                "D4_only": sum(v["D4"][a] > thr["D4"] for a in s) / max(len(s), 1),
                "D4c_only": sum(v["D4c"][a] > thr["D4c"] for a in s) / max(len(s), 1),
                "ties_at_thr": {k: sum(abs(v[k][a] - thr[k]) < 1e-9 for a in s) for k in thr}}
    print(win, out[win])
print("thresholds", thr)
json.dump(out, open(P / "ops_politics.json", "w"), indent=1)
