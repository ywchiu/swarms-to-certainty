"""Briefing agent: two calls per context (1: decision brief, 2: evidence independence), so the brief itself is not
primed to look for coordination. Results are appended to data/runs/<exp>.jsonl.

  python brief.py --dry-run          # build one context and print the prompts, no LLM call
"""
import asyncio, json, sys, time
from pathlib import Path
import httpx
from common import ROOT, DATA, llm

P = ROOT / "prompts"
SYS = (P / "brief_system.txt").read_text()
U1 = (P / "brief_user.txt").read_text()
U2 = (P / "voices_user.txt").read_text()
RUNS = DATA / "runs"; RUNS.mkdir(exist_ok=True)


def prompts(brand, period, ctx, note):
    s = SYS.format(defense_note=note).strip()
    return ([{"role": "system", "content": s}, {"role": "user", "content": U1.format(brand=brand, period=period, context=ctx)}],
            [{"role": "system", "content": s}, {"role": "user", "content": U2.format(brand=brand, period=period, context=ctx)}])


async def _one(client, job, temp):
    m1, m2 = prompts(job["brand"], job["period"], job["ctx"], job["note"])
    salt = f"efs|{job['id']}|rep{job['rep']}"
    out = dict(id=job["id"], rep=job["rep"], model=llm.MODEL, meta=job["meta"], idmap_swarm=job["idmap_swarm"])
    for name, msgs in (("brief", m1), ("voices", m2)):
        try:
            t = await llm.chat(client, msgs, temp=temp, max_tokens=900, salt=salt + name)
            out[name] = llm.parse_json(t); out[name + "_raw"] = t
        except Exception as e:  # parse failure or refusal: kept and counted, never silently dropped
            out[name] = None; out[name + "_err"] = repr(e)[:200]
    return out


async def run(jobs, exp, temp=0.7):
    async with httpx.AsyncClient() as client:
        res = await asyncio.gather(*[_one(client, j, temp) for j in jobs])
    with open(RUNS / f"{exp}.jsonl", "a") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return res


if __name__ == "__main__" and "--dry-run" in sys.argv:
    import numpy as np
    from common import load
    from context import retrieve, format_context
    from conditions import roots_of, flagged_accounts, swarm_file, truth
    org, cl = load("dev"); sw = json.load(open(swarm_file("dev", "S4m", 0)))
    msgs = org + sw; roots = roots_of(msgs)
    sel = retrieve(msgs, cl, "迷客夏")
    fl = flagged_accounts(msgs, cl)
    ctx, note, idmap = format_context(msgs, sel, roots, fl, defense="annotate")
    m1, m2 = prompts("迷客夏", "2026/10/18–10/26", ctx, note)
    print(m1[0]["content"], "\n----\n", m1[1]["content"][:3000], "\n...")
    print("truth", truth(msgs, sel, cl, "迷客夏"), "flagged swarm/organic:",
          sum(a.startswith("swarm:") for a in fl), sum(not a.startswith("swarm:") for a in fl))
