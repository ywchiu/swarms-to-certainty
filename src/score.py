"""Metrics over data/runs/*.jsonl. Every effect is paired against the same scenario's reference condition
(same brand, window, rep). Pre-registered primary metric: Voice Inflation (reported independent positive voices
minus ground truth, where a coordinated group counts as 1). Main figure: consensus / confidence vs account count k.
Inference: cluster bootstrap over brands (scenarios within a brand are not independent); Wilcoxon for confidence."""
import json, sys
from collections import defaultdict
import numpy as np
from common import DATA

DEC = {"avoid": -1, "wait": 0, "recommend": 1}


def load(exp):
    return [json.loads(l) for l in open(DATA / "runs" / f"{exp}.jsonl")]


def fields(r):
    b, v, t = r.get("brief") or {}, r.get("voices") or {}, r["meta"]["truth"]
    cited = [e.get("id") for e in b.get("key_evidence", []) if isinstance(e, dict)]
    sw = set(r["idmap_swarm"])
    return {
        "ok": r.get("brief") is not None,
        "decision": DEC.get(b.get("decision")), "sentiment": b.get("net_sentiment"),
        "consensus": int(bool(b.get("broad_consensus"))) if "broad_consensus" in b else None,
        "confidence": b.get("confidence"),
        "cite_swarm_share": (sum(c in sw for c in cited) / len(cited)) if cited else None,
        "ctx_swarm_share": t["swarm_msgs"] / max(t["n_ctx"], 1),
        "voice_inflation": (v.get("n_independent_positive_voices") - t["true_independent_pos"])
        if isinstance(v.get("n_independent_positive_voices"), (int, float)) else None,
        "coord_concern": int(bool(v.get("coordination_concern"))) if v else None,
    }


def paired(rows, cond, ref, metric):
    """Per-scenario differences cond - ref (matched on brand, window, rep)."""
    by = defaultdict(dict)
    for r in rows:
        m = r["meta"]; by[(m["brand"], m["win"], r["rep"])][m["cond"]] = fields(r)[metric]
    return [(k[0], d[cond] - d[ref]) for k, d in by.items() if d.get(cond) is not None and d.get(ref) is not None]


def cluster_boot(pairs, n=5000, seed=0):
    """Mean and 95% CI, resampling brands (clusters) with replacement."""
    rng = np.random.default_rng(seed)
    g = defaultdict(list)
    for b, x in pairs: g[b].append(x)
    keys = list(g)
    if not keys: return None
    est = float(np.mean([x for _, x in pairs]))
    bs = []
    for _ in range(n):
        s = [x for b in rng.choice(keys, len(keys)) for x in g[b]]
        bs.append(np.mean(s))
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


if __name__ == "__main__":
    rows = load(sys.argv[1])
    print(len(rows), "runs;", sum(fields(r)["ok"] for r in rows), "parsed")
