"""Sprint 2 (15 Nov) statistics -> data/politics/sprint2_numbers.json (aggregates only).

Three robustness experiments, Gemma (H200) and Gemini (agy):
  fresh  independent replication on six fresh, thread-disjoint organic backgrounds (f1-f6), each with a newly
         generated S4 swarm: k-ablation endpoints, same-call assessment (chained) and count-only control, and the
         renamed real-support control (renamed30).            files: pol_<exp>_fresh / _gemini_fresh
  mt     presentation-matched pair: 30 swarm payload messages vs 30 renamed real supportive messages matched on
         platform and length, one per account, no thread roots; two calls / chained / count-only.
  reg    candidate-registry check: stylised official list with the fictional candidate listed (L) or not (U), k = 1 vs 30.

Units. A cluster is (background, swarm seed). The original two backgrounds (dev, heldout) carry three swarm seeds each
and share their organic filler, so we also report CIs that resample BACKGROUNDS (8 = dev, heldout, f1-f6), averaging
the seeds within a background first. Paired contrasts take within-unit differences of unit means.

  python sprint2_numbers.py
"""
import json, re
import numpy as np
import pandas as pd
import score_politics as S
from common import DATA

RUNS = DATA / "runs"
R = lambda x, d=3: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)
MODELS = {"gemma": "", "gemini": "_gemini"}
METS = ("invite", "poll", "voices", "coord")


def load(exp, suf):
    f = RUNS / f"pol_{exp}{suf}.jsonl"
    if not f.exists() or f.stat().st_size == 0:
        return None
    old = S.SUF; S.SUF = suf
    try:
        df = S.rows(exp)
    finally:
        S.SUF = old
    df["sd"] = df["cond"].map(lambda c: (re.search(r"s(\d)", c) or re.search(r"(A)", c)).group(1))
    df["cl"] = df["win"] + "|s" + df["sd"]
    df["bg"] = df["win"]
    return df


def both(model, exp, fresh_only=False, old_only=False):
    """Original-background rows (pol_<exp><suf>) and fresh rows (pol_<exp><suf>_fresh) for one model."""
    suf = MODELS[model]
    parts = [] if fresh_only else [load(exp, suf)]
    if not old_only:
        parts.append(load(exp, suf + "_fresh"))
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    if old_only:
        df = df[~df["win"].str.startswith("f")]
    if fresh_only:
        df = df[df["win"].str.startswith("f")]
    return df


def boot_mean(v, n=5000, seed=0):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return {"diff": R(v.mean()), "lo": R(np.percentile(bs, 2.5)), "hi": R(np.percentile(bs, 97.5)), "units": int(len(v)),
            "up": int((v > 0).sum()), "down": int((v < 0).sum())}


def unit_means(df, met, unit):
    if unit == "bg":  # seeds averaged within background first
        return df.groupby(["bg", "cl"])[met].mean().groupby("bg").mean()
    return df.groupby("cl")[met].mean()


def paired(a, b, met, unit="cl"):
    if a is None or b is None or not len(a) or not len(b):
        return None
    d = (unit_means(a, met, unit) - unit_means(b, met, unit)).dropna()
    return boot_mean(d.values)


def did(a1, a0, b1, b0, met, unit="cl"):
    """(a1 - a0) - (b1 - b0), paired within unit: e.g. how much more the chained call cuts the swarm than real support."""
    parts = [unit_means(x, met, unit) for x in (a1, a0, b1, b0)]
    d = ((parts[0] - parts[1]) - (parts[2] - parts[3])).dropna()
    return boot_mean(d.values)


def rate(g, met):
    v = g[met].dropna()
    if not len(v):
        return None
    cm = g.groupby("cl")[met].mean().dropna()
    rng = np.random.default_rng(0)
    bs = [rng.choice(cm.values, len(cm)).mean() for _ in range(5000)]
    return {"mean": R(v.mean()), "lo": R(np.percentile(bs, 2.5)), "hi": R(np.percentile(bs, 97.5)), "n": int(len(v))}


def cellsum(g):
    return {m: rate(g, m) for m in METS if m in g} | {"briefs": int(len(g)), "parsed": int(g["ok"].sum())}


OUT = {}

# ------------------------------------------------------------------ fresh-background replication
for model in MODELS:
    o = OUT.setdefault(f"fresh|{model}", {})
    k = both(model, "k", fresh_only=True)
    if k is None:
        continue
    k = k[k["defense"] == "none"]
    for kv in sorted(k["k"].dropna().unique()):
        o[f"k{int(kv)}"] = cellsum(k[k["k"] == kv])
    k1, k30 = k[k["k"] == 1], k[k["k"] == k["k"].max()]
    for met in ("invite", "poll"):
        o[f"k30_minus_k1|{met}"] = paired(k30, k1, met)
    # pooled with the original six clusters (12 clusters / 8 backgrounds)
    ko = both(model, "k", old_only=True)
    if ko is not None:
        ko = ko[ko["defense"] == "none"]
        allk = pd.concat([ko, k], ignore_index=True)
        a1, a30 = allk[allk["k"] == 1], allk[allk["k"] == 30]
        for met in ("invite", "poll"):
            o[f"pooled_k30_minus_k1|{met}|clusters"] = paired(a30, a1, met, "cl")
            o[f"pooled_k30_minus_k1|{met}|backgrounds"] = paired(a30, a1, met, "bg")
    ch, ct = both(model, "chaink", fresh_only=True), both(model, "chainctlk", fresh_only=True)
    bn, cbn, ctbn = (both(model, e, fresh_only=True) for e in ("bn", "chainbn", "chainctlbn"))
    for name, d in (("chaink", ch), ("chainctlk", ct)):
        if d is not None:
            for kv in (1, 30):
                o[f"{name}{kv}"] = cellsum(d[d["k"] == kv])
    for name, d in (("bn30", bn), ("chainbn30", cbn), ("chainctlbn30", ctbn)):
        if d is not None:
            o[name] = cellsum(d)
    if ch is not None and bn is not None and cbn is not None:
        sw2, sw1 = ch[ch["k"] == 30], k30
        for met in ("invite", "poll"):
            o[f"swarm30_chain_minus_twocall|{met}"] = paired(sw2, sw1, met)
            o[f"real30_chain_minus_twocall|{met}"] = paired(cbn, bn, met)
            o[f"chain_real_minus_swarm|{met}"] = paired(cbn, sw2, met, "bg")  # one swarm per fresh background; f1 uses seed 1

# ------------------------------------------------------------------ natural S4 swarm: same-call assessment on fresh backgrounds
for model in MODELS:
    o = OUT.setdefault(f"fresh_nat|{model}", {})
    d = {e: both(model, e, fresh_only=True) for e in ("nat", "chain", "chainctl")}
    if d["nat"] is None:
        continue
    d = {e: (df[(df["defense"] == "none")] if df is not None else None) for e, df in d.items()}
    for e, df in d.items():
        if df is not None:
            o[f"{e}|A"] = cellsum(df[df["cond"] == "A"])
            o[f"{e}|S4"] = cellsum(df[df["cond"].str.startswith("S4")])
    s4 = {e: (df[df["cond"].str.startswith("S4")] if df is not None else None) for e, df in d.items()}
    bn, cbn = both(model, "bn", fresh_only=True), both(model, "chainbn", fresh_only=True)
    for met in ("invite", "poll"):
        if s4["chain"] is not None:
            o[f"S4_chain_minus_twocall|{met}"] = paired(s4["chain"], s4["nat"], met)
            if bn is not None and cbn is not None:
                o[f"chain_real30_minus_S4|{met}"] = paired(cbn, s4["chain"], met, "bg")
                o[f"did_S4cut_minus_realcut|{met}"] = did(s4["chain"], s4["nat"], cbn, bn, met, "bg")
        if s4["chainctl"] is not None:
            o[f"S4_chain_minus_countonly|{met}"] = paired(s4["chain"], s4["chainctl"], met)
    # renamed30 real support pooled over all eight backgrounds (original: 3 samples per window)
    pb, pc = both(model, "bn"), both(model, "chainbn")
    if pb is not None and pc is not None:
        pb, pc = pb[pb["cond"].str.startswith("BNn30")], pc[pc["cond"].str.startswith("BNn30")]
        o["pooled|real30_twocall"] = cellsum(pb); o["pooled|real30_chain"] = cellsum(pc)
        for met in ("invite", "poll"):
            o[f"pooled|real30_chain_minus_twocall|{met}|backgrounds"] = paired(pc, pb, met, "bg")
    # pooled with the original backgrounds (S4 only, no defense)
    on, oc = both(model, "nat", old_only=True), both(model, "chain", old_only=True)
    if on is not None and oc is not None and s4["chain"] is not None:
        on = on[(on["defense"] == "none") & on["cond"].str.startswith("S4")]
        oc = oc[(oc["defense"] == "none") & oc["cond"].str.startswith("S4")]
        an, ac = pd.concat([on, s4["nat"]]), pd.concat([oc, s4["chain"]])
        o["pooled|S4_twocall"] = cellsum(an); o["pooled|S4_chain"] = cellsum(ac)
        for met in ("invite", "poll"):
            o[f"pooled|S4_chain_minus_twocall|{met}|clusters"] = paired(ac, an, met, "cl")
            o[f"pooled|S4_chain_minus_twocall|{met}|backgrounds"] = paired(ac, an, met, "bg")

# ------------------------------------------------------------------ presentation-matched pair (all backgrounds)
for model in MODELS:
    o = OUT.setdefault(f"matched|{model}", {})
    d = {e: both(model, e) for e in ("mt", "chainmt", "chainctlmt")}
    if d["mt"] is None:
        continue
    for e, df in d.items():
        if df is None:
            continue
        for side in ("MSW", "MBN"):
            o[f"{e}|{side}"] = cellsum(df[df["cond"].str.startswith(side)])
    sw = {e: (df[df["cond"].str.startswith("MSW")] if df is not None else None) for e, df in d.items()}
    bn = {e: (df[df["cond"].str.startswith("MBN")] if df is not None else None) for e, df in d.items()}
    for met in ("invite", "poll", "voices"):
        o[f"twocall_real_minus_swarm|{met}"] = paired(bn["mt"], sw["mt"], met)
        if d["chainmt"] is not None:
            o[f"chain_real_minus_swarm|{met}"] = paired(bn["chainmt"], sw["chainmt"], met)
            o[f"chain_real_minus_swarm|{met}|backgrounds"] = paired(bn["chainmt"], sw["chainmt"], met, "bg")
            o[f"swarm_chain_minus_twocall|{met}"] = paired(sw["chainmt"], sw["mt"], met)
            o[f"real_chain_minus_twocall|{met}"] = paired(bn["chainmt"], bn["mt"], met)
            o[f"did_swarm_cut_minus_real_cut|{met}"] = did(sw["chainmt"], sw["mt"], bn["chainmt"], bn["mt"], met)
        if d["chainctlmt"] is not None:
            o[f"countonly_real_minus_swarm|{met}"] = paired(bn["chainctlmt"], sw["chainctlmt"], met)
    raw = d["mt"]
    o["units"] = {"clusters": int(raw["cl"].nunique()), "backgrounds": int(raw["bg"].nunique())}

# ------------------------------------------------------------------ candidate-registry check (all backgrounds)
for model in MODELS:
    o = OUT.setdefault(f"registry|{model}", {})
    r = both(model, "reg")
    if r is None:
        continue
    r["arm"] = r["cond"].str[-1]
    for arm in ("L", "U"):
        for kv in (1, 30):
            o[f"{arm}|k{kv}"] = cellsum(r[(r["arm"] == arm) & (r["k"] == kv)])
        a = r[r["arm"] == arm]
        for met in ("invite", "poll"):
            o[f"{arm}|k30_minus_k1|{met}"] = paired(a[a["k"] == 30], a[a["k"] == 1], met)
            o[f"{arm}|k30_minus_k1|{met}|backgrounds"] = paired(a[a["k"] == 30], a[a["k"] == 1], met, "bg")
    for met in ("invite", "poll"):
        for kv in (1, 30):
            o[f"U_minus_L|k{kv}|{met}"] = paired(r[(r["arm"] == "U") & (r["k"] == kv)], r[(r["arm"] == "L") & (r["k"] == kv)], met)
    o["units"] = {"clusters": int(r["cl"].nunique()), "backgrounds": int(r["bg"].nunique())}

json.dump(OUT, open(DATA / "politics" / "sprint2_numbers.json", "w"), ensure_ascii=False, indent=1)


def show(d, ind=""):
    for k, v in d.items():
        if isinstance(v, dict) and not ({"mean", "diff"} & set(v)):
            print(f"{ind}{k}:"); show(v, ind + "  ")
        else:
            print(f"{ind}{k}: {v}")


show(OUT)
