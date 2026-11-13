"""Tables for the election experiments -> data/politics/results.json and printed summaries.
Clusters for the bootstrap: (window, swarm seed) for the fictional candidate; (window, rep) for over-correction."""
import json, sys
from collections import defaultdict
import numpy as np
import pandas as pd
from common import DATA

import os
RUNS = DATA / "runs"
SUF = os.environ.get("RUN_SUFFIX", "")  # "_gemini" for the cross-family replication
DEC = {"decline": -1, "wait": 0, "invite": 1}


def rows(exp):
    f = RUNS / f"pol_{exp}{SUF}.jsonl"
    out = []
    for l in open(f):
        r = json.loads(l); m = r["meta"]; b = r.get("brief") or {}; v = r.get("voices") or {}
        cited = [e.get("id") for e in b.get("key_evidence", []) if isinstance(e, dict)]
        sw = set(r["idmap_swarm"])
        nv = v.get("n_independent_positive_voices")
        out.append(dict(
            exp=m["exp"], win=m["win"], cond=m["cond"], cand=m["cand"], defense=m["defense"], op=m["op"], rep=r["rep"],
            k=m.get("k"), dose=m.get("dose"), seed=m.get("seed", m["cond"]),
            ok=r.get("brief") is not None, ok_v=r.get("voices") is not None,
            invite=(b.get("decision") == "invite") if b else None, decision=DEC.get(b.get("decision")),
            poll=bool(b.get("include_in_poll")) if "include_in_poll" in b else None,
            momentum=b.get("grassroots_momentum"), sentiment=b.get("net_sentiment"),
            consensus=bool(b.get("broad_consensus")) if "broad_consensus" in b else None,
            confidence=b.get("confidence"),
            voices=nv if isinstance(nv, (int, float)) else None,
            voice_infl=(nv - m["truth"]["true_independent_pos"]) if isinstance(nv, (int, float)) else None,
            coord=bool(v.get("coordination_concern")) if v else None,
            cite_sw=(sum(c in sw for c in cited) / len(cited)) if cited else None,
            ctx_sw=m["truth"]["swarm_msgs"] / max(m["truth"]["n_ctx"], 1),
            n_flag=m["n_flagged_ctx"], n_flag_pos=m.get("n_flag_pos"), ctx=m.get("ctx"), n_flag_sw=m["n_flagged_swarm_ctx"],
            org_pos=m["truth"]["organic_pos_accounts"]))
    df = pd.DataFrame(out)
    for c in ("invite", "poll", "consensus", "coord"):
        df[c] = df[c].astype(float)
    return df


def boot(x, clusters, n=5000, seed=0):
    """Mean and 95% CI resampling clusters."""
    rng = np.random.default_rng(seed)
    g = defaultdict(list)
    for c, v in zip(clusters, x):
        if v is not None and not (isinstance(v, float) and np.isnan(v)): g[c].append(v)
    keys = list(g)
    if not keys: return (np.nan, np.nan, np.nan)
    est = float(np.mean([v for k in keys for v in g[k]]))
    bs = [np.mean([v for k in rng.choice(keys, len(keys)) for v in g[k]]) for _ in range(n)]
    return (round(est, 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3))


MET = ["invite", "poll", "momentum", "confidence", "voice_infl", "coord", "cite_sw"]


def table_nat():
    df = rows("nat"); df["attack"] = df["cond"].str[:2]
    df["cl"] = df["win"] + "|" + df["cond"]
    out = {}
    for (att, dfn, op), g in df.groupby(["attack", "defense", "op"]):
        out[f"{att}|{dfn}|{op}"] = {m: boot(g[m].tolist(), g["cl"].tolist()) for m in MET} | {
            "n": len(g), "parsed": int(g["ok"].sum()), "flag_sw_ctx": round(g["n_flag_sw"].mean(), 2),
            "flag_org_ctx": round((g["n_flag"] - g["n_flag_sw"]).mean(), 2), "ctx_sw": round(g["ctx_sw"].mean(), 3)}
    return out


def table_k():
    df = rows("k"); df["cl"] = df["win"] + "|" + df["seed"]
    out = {}
    for (k, dfn, op), g in df.groupby(["k", "defense", "op"]):
        out[f"k={k}|{dfn}|{op}"] = {m: boot(g[m].tolist(), g["cl"].tolist()) for m in MET} | {
            "n": len(g), "flag_sw_ctx": round(g["n_flag_sw"].mean(), 2)}
    # slope of momentum / invite / voices on log k (no defense), per cluster
    nd = df[df.defense == "none"]
    sl = {}
    for m in ("momentum", "invite", "voice_infl", "confidence"):
        s = [np.polyfit(np.log(g["k"].astype(float)), g[m].astype(float), 1)[0] for _, g in nd.dropna(subset=[m]).groupby("cl") if g["k"].nunique() > 1]
        sl[m] = (round(float(np.mean(s)), 3), len(s))
    out["slope_logk_nodefense"] = sl
    return out


def table_dose():
    df = rows("dose"); df["cl"] = df["win"] + "|" + df["seed"]
    return {f"n={n}": {m: boot(g[m].tolist(), g["cl"].tolist()) for m in MET} | {"n": len(g)} for n, g in df.groupby("dose")}


def table_oc():
    df = rows("oc"); df["cl"] = df["win"] + "|" + df["rep"].astype(str)
    base = df[df.defense == "none"].set_index(["cand", "win", "rep"])
    out = {}
    for (cand, dfn, op), g in df[df.defense != "none"].groupby(["cand", "defense", "op"]):
        d = {}
        for m in ("momentum", "voices", "sentiment", "confidence", "invite", "coord"):
            diffs = [(r.cl, r[m] - base.loc[(cand, r.win, r.rep), m]) for _, r in g.iterrows()
                     if pd.notna(r[m]) and pd.notna(base.loc[(cand, r.win, r.rep), m])]
            d["d_" + m] = boot([x for _, x in diffs], [c for c, _ in diffs])
        d["flag_ctx"] = round(g["n_flag"].mean(), 2)
        out[f"{cand}|{dfn}|{op}"] = d
    for cand, g in base.reset_index().groupby("cand"):
        out[f"{cand}|none"] = {m: round(float(g[m].astype(float).mean()), 3) for m in ("momentum", "voices", "sentiment", "confidence", "invite", "coord", "org_pos")}
    return out


def table_oc2():
    """Stance-matched contexts (24 pos / 24 neg / 12 neutral). Unit = context (cand, win, ctx), reps averaged.
    Delta = variant - none within context; asymmetry = mean delta(沈伯洋) - mean delta(蔣萬安), bootstrap over contexts."""
    df = rows("oc2")
    mets = ["momentum", "voices", "sentiment", "confidence", "invite", "coord"]
    g = df.groupby(["cand", "win", "ctx", "defense", "op"])[mets + ["n_flag", "n_flag_pos"]].mean().reset_index()
    base = g[g.defense == "none"].set_index(["cand", "win", "ctx"])
    out = {"base": {c: {m: round(float(x[m].mean()), 3) for m in mets} | {"n_ctx": len(x)} for c, x in base.reset_index().groupby("cand")}}
    rng = np.random.default_rng(0)
    for (dfn, op), x in g[g.defense != "none"].groupby(["defense", "op"]):
        row = {}
        d = {}
        for cand, y in x.groupby("cand"):
            y = y.set_index(["cand", "win", "ctx"])
            d[cand] = {m: (y[m] - base.loc[y.index, m]).dropna().values for m in mets}
            row[cand] = {"d_" + m: boot(list(d[cand][m]), list(range(len(d[cand][m])))) for m in ("momentum", "voices", "sentiment", "confidence", "invite")}
            row[cand]["flagged_ctx"] = round(float(y["n_flag"].mean()), 2)
            row[cand]["flagged_supportive_of_24"] = round(float(y["n_flag_pos"].mean()), 2)
        if len(d) == 2:
            asym = {}
            for m in ("momentum", "voices", "sentiment", "invite"):
                a, b = d["沈伯洋"][m], d["蔣萬安"][m]
                bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(5000)]
                asym[m] = (round(float(a.mean() - b.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3))
            row["asym_shen_minus_chiang"] = asym
        out[f"{dfn}|{op}"] = row
    return out


def table_chain():
    """Single chained call: independence assessment and decision in one JSON (same contexts as nat)."""
    df = rows("chain"); df["attack"] = df["cond"].str[:2]; df["cl"] = df["win"] + "|" + df["cond"]
    return {f"{a}|{dfn}|{op}": {m: boot(g[m].tolist(), g["cl"].tolist()) for m in MET} | {"n": len(g), "parsed": int(g["ok"].sum())}
            for (a, dfn, op), g in df.groupby(["attack", "defense", "op"])}


def table_chainctl():
    """Control for the chained call: same single call and order, step 1 only counts posts."""
    out = {}
    df = rows("chainctl"); df["attack"] = df["cond"].str[:2]; df["cl"] = df["win"] + "|" + df["cond"]
    for a, g in df.groupby("attack"):
        out[a] = {m: boot(g[m].tolist(), g["cl"].tolist()) for m in ("invite", "poll", "momentum", "confidence")} | {"n": len(g), "parsed": int(g["ok"].sum())}
    df = rows("chainctloc"); df["cl"] = df["win"] + "|" + df["ctx"].astype(str)
    for c, g in df.groupby("cand"):
        out[c] = {m: boot(g[m].tolist(), g["cl"].tolist()) for m in ("invite", "poll", "momentum")} | {"n": len(g)}
    return out


def table_chainoc():
    """Chained call on the oc2 real-candidate contexts (no defense), next to the two-call oc2 baseline."""
    out = {}
    for nm, df in (("chain", rows("chainoc")), ("two_call", rows("oc2").query("defense == 'none'"))):
        df = df.assign(cl=df["win"] + "|" + df["ctx"].astype(str))
        for cand, g in df.groupby("cand"):
            out[f"{nm}|{cand}"] = {m: boot(g[m].tolist(), g["cl"].tolist()) for m in ("invite", "poll", "momentum", "voices", "coord")} | {"n": len(g)}
    return out


def table_bn():
    """Benign control (sprint): genuine independent support transplanted onto the fictional candidate (n accounts)."""
    out = {}
    for e in ("bn", "chainbn", "chainctlbn"):
        d = rows(e); d["n"] = d["cond"].str.extract(r"n(\d+)")[0].astype(int); d["cl"] = d["win"] + "|" + d["seed"]
        for n, g in d.groupby("n"):
            out[f"{e}|n{n}"] = {m: boot(g[m].tolist(), g["cl"].tolist()) for m in ("invite", "poll", "momentum", "voices", "voice_infl", "coord")} | {"n": len(g), "parsed": int(g["ok"].sum())}
    return out


def table_note():
    df = rows("note"); df["attack"] = df["cond"].str[:2]; df["cl"] = df["win"] + "|" + df["cond"]
    return {f"{a}|note-only": {m: boot(g[m].tolist(), g["cl"].tolist()) for m in MET} | {"n": len(g)} for a, g in df.groupby("attack")}


if __name__ == "__main__":
    exps = sys.argv[1].split(",") if len(sys.argv) > 1 else ["nat", "k", "dose", "oc", "oc2", "note", "chain", "chainoc", "chainctl", "bn"]
    res = {}
    for e in exps:
        try:
            res[e] = {"nat": table_nat, "k": table_k, "dose": table_dose, "oc": table_oc, "oc2": table_oc2, "note": table_note, "chain": table_chain, "chainoc": table_chainoc, "chainctl": table_chainctl, "bn": table_bn}[e]()
        except FileNotFoundError:
            continue
        print(f"== {e}")
        for k, v in res[e].items():
            print(k, json.dumps(v, ensure_ascii=False))
    f = DATA / "politics" / f"results{SUF}.json"
    old = json.load(open(f)) if f.exists() else {}
    json.dump(old | res, open(f, "w"), ensure_ascii=False, indent=1)
