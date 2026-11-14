"""Recomputes the briefing statistics quoted in the report from the run logs (corpus totals: build_politics.py; tie blocks: b1_tie_blocks.py; operating-point sweep: b1_matched_rate.py) -> data/politics/report_numbers.json (aggregates only).

  python report_numbers.py            # prints and writes report_numbers.json

Units: Gemma swarm / benign / k cells are 6 clusters (window x swarm seed) x 3 reps; real-candidate cells are
16 contexts per candidate (2 windows x 8 stance-matched contexts) x 2 reps. Paired CIs resample clusters (or contexts)
and take the mean of within-cluster differences. Summary keyword match: footnote 1 of the report.
"""
import json, os, re
import numpy as np
import pandas as pd
import score_politics as S
from common import DATA

KW = re.compile("協同|水軍|帶風向|操作|人為")
RUNS = DATA / "runs"
OUT = {}
R = lambda x, d=3: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def load(exp, suf=""):
    """score_politics.rows() plus the brief summary and a keyword flag, in file order."""
    f = RUNS / f"pol_{exp}{suf}.jsonl"
    if not f.exists():
        return None
    old = S.SUF; S.SUF = suf
    try:
        df = S.rows(exp)
    finally:
        S.SUF = old
    summ = []
    for l in open(f):
        b = json.loads(l).get("brief") or {}
        summ.append(str(b.get("summary", "")) if b else None)
    df["summary"] = summ
    df["kw"] = [None if s is None else bool(KW.search(s)) for s in summ]
    return df


def ci(x, cl, n=5000):
    return S.boot(list(x), list(cl), n=n)


def paired(a, b, key, met, n=5000, seed=0):
    """Mean over units of (mean_a - mean_b) within unit `key`; 95% bootstrap over units."""
    ga = a.groupby(key)[met].mean(); gb = b.groupby(key)[met].mean()
    d = (ga - gb).dropna()
    rng = np.random.default_rng(seed)
    bs = [rng.choice(d.values, len(d)).mean() for _ in range(n)]
    return {"diff": R(d.mean()), "lo": R(np.percentile(bs, 2.5)), "hi": R(np.percentile(bs, 97.5)), "units": int(len(d)),
            "units_up": int((d > 0).sum()), "units_down": int((d < 0).sum())}


def unpaired(a, b, n=5000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    rng = np.random.default_rng(seed)
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return {"diff": R(a.mean() - b.mean()), "lo": R(np.percentile(bs, 2.5)), "hi": R(np.percentile(bs, 97.5)), "n_a": len(a), "n_b": len(b)}


def cell(g, cl="cl", mets=("invite", "poll", "momentum", "voices", "voice_infl", "coord")):
    out = {m: ci(g[m], g[cl]) for m in mets if m in g}
    out |= {"briefs": int(len(g)), "parsed": int(g["ok"].sum()), "voices_valid": int(g["voices"].notna().sum()),
            "kw": f"{int(g['kw'].fillna(False).sum())}/{int(g['kw'].notna().sum())}"}
    return out


def cl_swarm(df):
    df = df.copy(); df["attack"] = df["cond"].str[:2]; df["cl"] = df["win"] + "|" + df["cond"].str[:4]
    return df


def cl_oc(df):
    df = df.copy(); df["cl"] = df["win"] + "|" + df["ctx"].astype(str); df["unit"] = df["cand"] + "|" + df["cl"]
    return df


# ------------------------------------------------------------------ Table 1: chained vs two-call (Gemma and Gemini)
def table1():
    t = {}
    for model, suf in (("gemma", ""), ("gemini", "_gemini")):
        nat, chain, ctl = load("nat", suf), load("chain", suf), load("chainctl", suf)
        if nat is None:
            continue
        nat, chain, ctl = cl_swarm(nat), cl_swarm(chain), cl_swarm(ctl)
        for att in ("S4", "S5"):
            t[f"{model}|{att}|two_call"] = cell(nat[(nat.attack == att) & (nat.defense == "none")])
            t[f"{model}|{att}|chained_count_only"] = cell(ctl[ctl.attack == att])
            t[f"{model}|{att}|chained_independence"] = cell(chain[(chain.attack == att) & (chain.defense == "none")])
            for d in ("annotate", "collapse"):
                g = chain[(chain.attack == att) & (chain.defense == d)]
                if len(g): t[f"{model}|{att}|chained_independence+{d}@b5"] = cell(g)
        both = lambda d: d[d.attack.isin(["S4", "S5"])]
        t[f"{model}|S4+S5|two_call"] = cell(both(nat[nat.defense == "none"]))
        t[f"{model}|S4+S5|chained_count_only"] = cell(both(ctl))
        t[f"{model}|S4+S5|chained_independence"] = cell(both(chain[chain.defense == "none"]))
        oc2, coc, cctl = load("oc2", suf), load("chainoc", suf), load("chainctloc", suf)
        oc2, coc, cctl = cl_oc(oc2), cl_oc(coc), cl_oc(cctl)
        for cand in ("蔣萬安", "沈伯洋"):
            t[f"{model}|{cand}|two_call"] = cell(oc2[(oc2.cand == cand) & (oc2.defense == "none")])
            t[f"{model}|{cand}|chained_count_only"] = cell(cctl[cctl.cand == cand])
            t[f"{model}|{cand}|chained_independence"] = cell(coc[coc.cand == cand])
            # paired: chained independence minus count-only, over contexts
            t[f"{model}|{cand}|chain_minus_countonly"] = {m: paired(coc[coc.cand == cand], cctl[cctl.cand == cand], "cl", m)
                                                           for m in ("momentum", "invite", "poll")}
        t[f"{model}|real|kw"] = {"two_call": cell(oc2[oc2.defense == "none"])["kw"], "count_only": cell(cctl)["kw"], "chained": cell(coc)["kw"]}
    # benign control (Gemma only)
    for e, nm in (("bn", "two_call"), ("chainctlbn", "chained_count_only"), ("chainbn", "chained_independence")):
        d = load(e); d["n"] = d["cond"].str.extract(r"n(\d+)")[0].astype(int); d["cl"] = d["win"] + "|" + d["seed"]
        for n, g in d.groupby("n"):
            t[f"gemma|renamed{n}|{nm}"] = cell(g)
    return t


# ------------------------------------------------------------------ Table 2: defenses
def table2():
    t = {}
    for model, suf in (("gemma", ""), ("gemini", "_gemini")):
        nat = load("nat", suf)
        nat = cl_swarm(nat)
        for (att, dfn, op), g in nat.groupby(["attack", "defense", "op"]):
            c = cell(g)
            c |= {"flag_sw_ctx": R(g["n_flag_sw"].mean(), 2), "flag_org_ctx": R((g["n_flag"] - g["n_flag_sw"]).mean(), 2),
                  "briefs_inflating": f"{int((g['voice_infl'] > 0).sum())}/{int(g['voice_infl'].notna().sum())}"}
            t[f"{model}|{att}|{dfn}|{op}"] = c
        for att in ("S4", "S5"):  # collapse - annotate at the same operating point, paired within cluster
            for op in ("b0.5", "b5", "tea"):
                a = nat[(nat.attack == att) & (nat.defense == "collapse") & (nat.op == op)]
                b = nat[(nat.attack == att) & (nat.defense == "annotate") & (nat.op == op)]
                if len(a) and len(b):
                    t[f"{model}|{att}|collapse_minus_annotate|{op}"] = {m: paired(a, b, "cl", m) for m in ("invite", "poll", "voice_infl")}
        note = load("note", suf)
        if note is not None:
            note = cl_swarm(note)
            for att, g in note.groupby("attack"):
                t[f"{model}|{att}|note_only"] = cell(g) | {"briefs_inflating": f"{int((g['voice_infl'] > 0).sum())}/{int(g['voice_infl'].notna().sum())}"}
    # Gemini partial-flag pooled count: briefs inflating among flagged (annotate/collapse @b5) vs note-only
    nat = cl_swarm(load("nat", "_gemini")); nat = nat[nat.attack != "A"]
    fl = nat[(nat.op == "b5")]
    t["gemini|partial_flag_pooled"] = {"flagged_b5_inflating": f"{int((fl['voice_infl'] > 0).sum())}/{int(fl['voice_infl'].notna().sum())}",
                                       "range": [R(fl.loc[fl.voice_infl > 0, "voice_infl"].min()), R(fl.loc[fl.voice_infl > 0, "voice_infl"].max())]}
    nn = nat[nat.op == "noflag"]
    t["gemini|note_only_pooled"] = f"{int((nn['voice_infl'] > 0).sum())}/{int(nn['voice_infl'].notna().sum())}"
    return t


# ------------------------------------------------------------------ k-ablation
def table_k():
    t = {}
    for model, suf in (("gemma", ""), ("gemini", "_gemini")):
        k = load("k", suf)
        k["cl"] = k["win"] + "|" + k["seed"]
        for (kk, dfn, op), g in k.groupby(["k", "defense", "op"]):
            t[f"{model}|k={kk}|{dfn}|{op}"] = cell(g, mets=("invite", "poll", "momentum", "voices", "voice_infl")) | {"flag_sw_ctx": R(g["n_flag_sw"].mean(), 2)}
        nd = k[k.defense == "none"]
        t[f"{model}|paired_k3_minus_k1"] = {m: paired(nd[nd.k == 3], nd[nd.k == 1], "cl", m) for m in ("invite", "poll", "momentum")}
        t[f"{model}|paired_k30_minus_k10"] = {m: paired(nd[nd.k == 30], nd[nd.k == 10], "cl", m) for m in ("invite", "poll", "momentum")}
    # how many of the k=3 / k=10 / k=30 swarm accounts Gemma counts as independent (no defense)
    g = load("k"); g = g[g.defense == "none"]
    t["gemma|k_voices_dist"] = {int(kk): sorted(x["voices"].dropna().astype(int).tolist()) for kk, x in g.groupby("k")}
    return t


# ------------------------------------------------------------------ over-correction: baselines, asymmetry, missingness
def table_oc():
    t = {}
    for model, suf in (("gemma", ""), ("gemini", "_gemini")):
        oc2 = cl_oc(load("oc2", suf))
        base = oc2[oc2.defense == "none"]
        pc = base.groupby(["cand", "cl"])[["voices", "momentum"]].mean().reset_index()
        t[f"{model}|baseline"] = {c: {m: R(x[m].mean()) for m in ("voices", "momentum")} for c, x in pc.groupby("cand")}
        t[f"{model}|baseline_asym_shen_minus_chiang"] = {m: unpaired(pc[pc.cand == "沈伯洋"][m], pc[pc.cand == "蔣萬安"][m]) for m in ("voices", "momentum")}
        # missingness: valid voices answers per (cand, win, condition)
        miss = oc2.assign(cond_=oc2.defense + "|" + oc2.op).groupby(["cand", "win", "cond_"]).agg(
            n=("ok", "size"), voices_valid=("voices", lambda x: int(x.notna().sum()))).reset_index()
        t[f"{model}|oc2_voices_valid"] = {"total": f"{int(oc2['voices'].notna().sum())}/{len(oc2)}",
                                         "by_cell": {f"{r.cand}|{r.win}|{r.cond_}": f"{r.voices_valid}/{r.n}" for r in miss.itertuples()}}
        # per-condition voice deltas (variant - none, within context) and the camp asymmetry: available-case and
        # complete-case (contexts where every condition has a valid voices answer in every rep)
        mets = ["voices"]
        cc_ctx = oc2.groupby("unit")["voices"].apply(lambda x: x.notna().all())
        cc_ctx = set(cc_ctx[cc_ctx].index)
        for nm, d in (("available", oc2), ("complete_case", oc2[oc2.unit.isin(cc_ctx)])):
            g = d.groupby(["cand", "unit", "defense", "op"])["voices"].mean().reset_index()
            b = g[g.defense == "none"].set_index("unit")["voices"]
            row = {"contexts": {c: int(x.unit.nunique()) for c, x in g.groupby("cand")}}
            for (dfn, op), x in g[g.defense != "none"].groupby(["defense", "op"]):
                x = x.assign(d=x["voices"].values - b.reindex(x.unit).values)
                row[f"{dfn}|{op}"] = {c: ci(y["d"], y["unit"]) for c, y in x.groupby("cand")} | {
                    "asym": unpaired(x[x.cand == "沈伯洋"]["d"], x[x.cand == "蔣萬安"]["d"])}
            pcb = g[g.defense == "none"]
            row["baseline_asym"] = unpaired(pcb[pcb.cand == "沈伯洋"]["voices"], pcb[pcb.cand == "蔣萬安"]["voices"])
            t[f"{model}|oc2_{nm}"] = row
        # flagged supportive messages per context
        t[f"{model}|flagged_supportive_of_24"] = {f"{c}|{op}|{w}": R(x["n_flag_pos"].mean(), 2)
                                                 for (c, op, w), x in oc2[oc2.defense != "none"].groupby(["cand", "op", "win"])}
    return t


# ------------------------------------------------------------------ sprint controls E1 (chained on k contexts) and E2 (B3)
def table_sprint():
    t = {}
    for e in ("chaink", "chainctlk"):
        d = load(e)
        if d is None: continue
        d["cl"] = d["win"] + "|" + d["seed"]
        for kk, g in d.groupby("k"):
            t[f"{e}|k={kk}"] = cell(g, mets=("invite", "poll", "momentum", "voices", "voice_infl", "coord"))
    k = load("k"); k["cl"] = k["win"] + "|" + k["seed"]
    ch = load("chaink")
    if ch is not None:
        ch["cl"] = ch["win"] + "|" + ch["seed"]
        for kk in (1, 30):
            a, b = ch[ch.k == kk], k[(k.k == kk) & (k.defense == "none")]
            t[f"chaink_minus_twocall|k={kk}"] = {m: paired(a, b, "cl", m) for m in ("invite", "poll", "momentum")}
    b3 = load("b3")
    if b3 is not None:
        b3 = cl_swarm(b3); nat = cl_swarm(load("nat"))
        for (att, dfn, op), g in b3.groupby(["attack", "defense", "op"]):
            c = cell(g) | {"flag_sw_ctx": R(g["n_flag_sw"].mean(), 2), "flag_org_ctx": R((g["n_flag"] - g["n_flag_sw"]).mean(), 2)}
            t[f"b3|{att}|{dfn}|{op}"] = c
            ref = nat[(nat.attack == att) & (nat.defense == dfn) & (nat.op == "b5")]
            t[f"b3|{att}|{dfn}|detector_b5_minus_{op}"] = {m: paired(ref, g, "cl", m) for m in ("voice_infl", "invite", "poll")}
    b3k = load("b3k")
    if b3k is not None:
        b3k["cl"] = b3k["win"] + "|" + b3k["seed"]
        for dfn, g in b3k.groupby("defense"):
            t[f"b3k|k=30|{dfn}|nf5"] = cell(g, mets=("invite", "poll", "momentum", "voices", "voice_infl"))
            for op in ("b5", "tea"):
                ref = k[(k.k == 30) & (k.defense == dfn) & (k.op == op)]
                t[f"b3k|k=30|{dfn}|{op}_minus_nf5"] = {m: paired(ref, g, "cl", m) for m in ("voice_infl",)}
            ref = k[(k.k == 30) & (k.defense == "none")]
            t[f"b3k|k=30|{dfn}|nf5_minus_none"] = {m: paired(g, ref, "cl", m) for m in ("voice_infl", "invite")}
    return t


# ------------------------------------------------------------------ sprint (14 Nov): more briefing models, core experiments, no defense
MODELS = (("gemma", ""), ("gemini", "_gemini"), ("qwen", "_qwen"), ("gpt", "_gpt"), ("oss", "_oss"), ("haiku", "_haiku"))


def table_models():
    t = {}
    for model, suf in MODELS:
        k = load("k", suf)
        if k is None:
            continue
        k["cl"] = k["win"] + "|" + k["seed"]; nd = k[k.defense == "none"]
        for kk, g in nd.groupby("k"):
            t[f"{model}|k={kk}|two_call"] = cell(g, mets=("invite", "poll", "momentum", "voices", "voice_infl"))
        t[f"{model}|paired_k3_minus_k1"] = {m: paired(nd[nd.k == 3], nd[nd.k == 1], "cl", m) for m in ("invite", "poll", "momentum")}
        t[f"{model}|paired_k30_minus_k1"] = {m: paired(nd[nd.k == nd.k.max()], nd[nd.k == 1], "cl", m) for m in ("invite", "poll", "momentum")}
        for e in ("chaink", "chainctlk"):
            d = load(e, suf)
            if d is None: continue
            d["cl"] = d["win"] + "|" + d["seed"]
            for kk, g in d.groupby("k"):
                t[f"{model}|k={kk}|{e}"] = cell(g)
        nat, chain, ctl = load("nat", suf), load("chain", suf), load("chainctl", suf)
        if nat is not None and chain is not None and ctl is not None:
            nat, chain, ctl = cl_swarm(nat), cl_swarm(chain), cl_swarm(ctl)
            for att in ("A", "S4", "S5"):
                t[f"{model}|{att}|two_call"] = cell(nat[(nat.attack == att) & (nat.defense == "none")])
                t[f"{model}|{att}|chained_count_only"] = cell(ctl[ctl.attack == att])
                t[f"{model}|{att}|chained_independence"] = cell(chain[(chain.attack == att) & (chain.defense == "none")])
            sw = lambda d: d[d.attack.isin(["S4", "S5"])]
            t[f"{model}|S4+S5|chain_minus_countonly"] = {m: paired(sw(chain[chain.defense == "none"]), sw(ctl), "cl", m) for m in ("invite", "poll")}
            t[f"{model}|S4+S5|chain_minus_twocall"] = {m: paired(sw(chain[chain.defense == "none"]), sw(nat[nat.defense == "none"]), "cl", m) for m in ("invite", "poll")}
        for e, nm in (("bn", "two_call"), ("chainctlbn", "chained_count_only"), ("chainbn", "chained_independence")):
            d = load(e, suf)
            if d is None: continue
            d["n"] = d["cond"].str.extract(r"n(\d+)")[0].astype(int); d["cl"] = d["win"] + "|" + d["seed"]
            for n, g in d.groupby("n"):
                t[f"{model}|renamed{n}|{nm}"] = cell(g)
            if e == "chainbn" and (b := load("bn", suf)) is not None:
                b["n"] = b["cond"].str.extract(r"n(\d+)")[0].astype(int); b["cl"] = b["win"] + "|" + b["seed"]
                for n in sorted(d.n.unique()):
                    t[f"{model}|renamed{n}|chain_minus_twocall"] = {m: paired(d[d.n == n], b[b.n == n], "cl", m) for m in ("invite", "poll")}
    return t


def rep_agreement():
    """Share of no-defense contexts with 3 parsed reps in which all reps give the same decision (App. A)."""
    import collections
    exps = ("nat", "k", "chain", "chainctl", "chaink", "chainctlk", "bn", "chainbn", "chainctlbn")
    t = {}
    for model, suf in MODELS:
        agree = {"invite": [0, 0], "poll": [0, 0]}
        for e in exps:
            f = RUNS / f"pol_{e}{suf}.jsonl"
            if not f.exists(): continue
            g = collections.defaultdict(list)
            for line in open(f):
                r = json.loads(line)
                if e in ("nat", "k") and r["meta"].get("defense") not in (None, "none"): continue
                if r.get("brief"): g[r["id"]].append(r["brief"])
            for bs in g.values():
                if len(bs) < 3: continue
                for key in agree:
                    vals = [(b.get("decision") == "invite") if key == "invite" else bool(b.get("include_in_poll")) for b in bs]
                    agree[key][0] += len(set(vals)) == 1; agree[key][1] += 1
        t[model] = {k: {"agree": v[0], "contexts": v[1], "rate": round(v[0] / v[1], 3)} for k, v in agree.items() if v[1]}
    return t


if __name__ == "__main__":
    for nm, f in (("table1_chain", table1), ("table2_defense", table2), ("k_ablation", table_k), ("overcorrection", table_oc), ("sprint_controls", table_sprint), ("models", table_models), ("rep_agreement", rep_agreement)):
        OUT[nm] = f()
        print(f"== {nm}")
        for k, v in OUT[nm].items():
            print(k, json.dumps(v, ensure_ascii=False, default=str))
    b1 = DATA / "politics" / "b1_matched_rate.json"
    if b1.exists():
        OUT["b1_operating_points"] = json.load(open(b1))
    json.dump(OUT, open(DATA / "politics" / "report_numbers.json", "w"), ensure_ascii=False, indent=1, default=str)
