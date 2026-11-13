"""Pools for each experimental condition. A pool = organic window messages + injected messages; the context the
briefing agent sees is retrieve()-d from it (natural) or assembled directly (fixed exposure, k-ablation).

Defense flags use the prior PayloadConv detectors (D4, D4c) with the frozen 0.5% thresholds from the Swarm
sprint (organic-only dev window; unchanged by the 2026-11-11 erratum). One fixed rule for every condition:
an account is flagged if D4 > thr_D4 or D4c > thr_D4c. The defender does not know which attack is running."""
import json
import numpy as np
from common import D, WIN, BRANDS, load, mentions

THR = {"D4": 6.385486210973915, "D4c": 29.0405289860769}


def roots_of(msgs):
    return {m["msg_id"]: m["text"] for m in msgs if m["kind"] == "post"}


def flagged_accounts(msgs, claims):
    acc = D.accounts_of(msgs)
    scored = sorted(a for a, v in acc.items() if len(v) >= D.MIN_MSGS)
    d4, _, _ = D.d4_claim(msgs, acc, scored, claims)
    d4c, _, _ = D.d4_claim(msgs, acc, scored, claims, inherit=True)
    return frozenset(a for a in scored if d4[a] > THR["D4"] or d4c[a] > THR["D4c"])


def swarm_file(win, cond, seed):
    return WIN[win] / f"swarm_{cond}_s{seed}.json"  # prior data (S4k/S4m); new brands are written here on 11/13


def payload_msgs(sw, claims, brand):
    """Swarm messages that actually endorse the brand (payload flag and the brand named in text or target)."""
    al = BRANDS[brand]
    out = []
    for m in sw:
        c = claims.get(m["msg_id"]) or {}
        named = mentions(m["text"], al) or (c.get("target") and mentions(c["target"], al))
        if m.get("payload") and named:
            out.append(m)
    return out


def reassign(msgs, k, tag="k"):
    """Account-count ablation: identical texts, attributed round-robin to k swarm accounts."""
    return [dict(m, account=f"swarm:{tag}{k}:{j % k}") for j, m in enumerate(msgs)]


def positive_real(win, claims, brand, n, rng, exclude_accounts=()):
    """Condition B: n real positive endorsements of the brand from window `win`, one per account."""
    org, _ = load(win)
    al = BRANDS[brand]
    seen, cand = set(exclude_accounts), []
    for m in org:
        c = claims.get(m["msg_id"])
        if c and c["stance"] > 0 and c["target"] and mentions(c["target"], al) and m["account"] not in seen:
            seen.add(m["account"]); cand.append(m)
    rng.shuffle(cand)
    return cand[:n], roots_of(org)


def truth(msgs, sel, claims, brand):
    """Ground truth for the voices call: distinct positive organic accounts + 1 per coordinated group present."""
    al = BRANDS[brand]
    org_pos, sw_acc, sw_n = set(), set(), 0
    for i in sel:
        m = msgs[i]; c = claims.get(m["msg_id"]) or {}
        is_sw = m["account"].startswith("swarm:")
        if is_sw:
            sw_acc.add(m["account"]); sw_n += 1
        elif c.get("stance", 0) > 0 and c.get("target") and mentions(c["target"], al):
            org_pos.add(m["account"])
    return {"organic_pos_accounts": len(org_pos), "swarm_msgs": sw_n, "swarm_accounts": len(sw_acc),
            "true_independent_pos": len(org_pos) + (1 if sw_n else 0), "n_ctx": len(sel)}
