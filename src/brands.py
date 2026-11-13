"""Extractor-resolved brand statistics per window (organic only): messages whose extracted target belongs to the
brand's alias cluster, split by stance; distinct accounts and threads; plus masked text-mention counts.
Writes data/brands.json. Used to pick scenarios and to check which brands support condition B."""
import json
from collections import defaultdict
from common import BRANDS, D, DATA, load, mentions

EXCLUDE = {"50嵐": ("かほる",)}  # 五十嵐かほる is a person

out = {}
for win in ("dev", "heldout"):
    org, cl = load(win)
    ann = [(m, cl.get(m["msg_id"])) for m in org]
    names = sorted({x for _, c in ann if c and c["target"] for x in D.split_targets(c["target"])})
    for b, al in BRANDS.items():
        # a target string names the brand if it contains an alias after masking collision words; we do not use the
        # detector's alias clusters here because they can join two brands named in one string (迷客夏 鶴茶樓)
        hit = {n for n in names if mentions(n, al) and not any(x in n for x in EXCLUDE.get(b, ()))}
        st = dict(n=0, pos=0, neg=0, acc=set(), pos_acc=set(), thr=set(), pos_thr=set())
        for m, c in ann:
            if not c or not c["target"]: continue
            if not any(x in hit for x in D.split_targets(c["target"])): continue
            st["n"] += 1; st["acc"].add(m["account"]); st["thr"].add(m["post_id"])
            if c["stance"] > 0: st["pos"] += 1; st["pos_acc"].add(m["account"]); st["pos_thr"].add(m["post_id"])
            elif c["stance"] < 0: st["neg"] += 1
        txt = [m for m in org if mentions(m["text"], al)]
        out.setdefault(b, {})[win] = dict(extracted=st["n"], pos=st["pos"], neg=st["neg"], accounts=len(st["acc"]),
                                         pos_accounts=len(st["pos_acc"]), threads=len(st["thr"]), pos_threads=len(st["pos_thr"]),
                                         text_mentions=len(txt), text_accounts=len({m["account"] for m in txt}),
                                         names=sorted(hit)[:15])
json.dump(out, open(DATA / "brands.json", "w"), ensure_ascii=False, indent=1)
print(f"{'brand':10s} | dev: ext pos neg acc posAcc thr txt | heldout: ext pos neg acc posAcc thr txt")
for b, w in out.items():
    f = lambda d: f"{d['extracted']:4d} {d['pos']:4d} {d['neg']:3d} {d['accounts']:4d} {d['pos_accounts']:4d} {d['threads']:4d} {d['text_mentions']:4d}"
    print(f"{b:10s} | {f(w['dev'])} | {f(w['heldout'])}")
