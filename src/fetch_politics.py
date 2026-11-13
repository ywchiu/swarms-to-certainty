"""Fetch the 2026 Taipei mayoral race discussion (posts + comments) from InfoMiner -> data/politics/organic_raw.jsonl.
Pre-sprint data collection (disclosed). Reuses the prior InfoMiner client (src/im.py of the Swarm sprint)."""
import json, sys, time
from common import PRIOR, DATA
sys.path.insert(0, str(PRIOR / "src")); import os; os.chdir(PRIOR)  # im.py reads PRIOR/.env (key never printed)
import im
KW = "(蔣萬安 OR 沈伯洋 OR 台北市長 OR 臺北市長 OR 北市長)"
START, END = "2026-10-25", "2026-11-11"
KEEP = {"ptt", "threads", "dcard", "mobile01", "fans", "fb_group", "youtube"}
start = int(time.mktime(time.strptime(START, "%Y-%m-%d"))); end = int(time.mktime(time.strptime(END, "%Y-%m-%d")))
out = open(DATA / "politics" / "organic_raw.jsonl", "w"); seen = set(); n_posts = n_push = 0
t = start
while t < end:
    t2 = min(t + 86400, end); page = 1
    while True:
        d = im.qsearch(f"{KW} && tm: (>={t} && <={t2})", page=page, size=100)
        rows = d.get("data", [])
        for x in rows:
            if x["id"] in seen: continue
            seen.add(x["id"]); src = x.get("source", "")
            if not any(src == k or src.startswith(k) for k in KEEP): continue
            rec = {k: x.get(k) for k in ("id", "source", "source_name", "author", "author_id", "datetime", "tm", "title", "content", "url", "comment_count", "like_count")}
            rec["kind"] = "post"; out.write(json.dumps(rec, ensure_ascii=False) + "\n"); n_posts += 1
            if src == "ptt" or int(x.get("comment_count") or 0) > 0:
                for p in range(1, 6):
                    try: pd = im.pushs(x["id"], p).get("data", [])
                    except Exception as e: print("push err", e); break
                    for c in pd:
                        c2 = {k: c.get(k) for k in ("id", "author", "author_id", "content", "push_type", "datetime", "tm", "source")}
                        c2.update(kind="comment", post_id=x["id"], post_source=src)
                        out.write(json.dumps(c2, ensure_ascii=False) + "\n"); n_push += 1
                    if len(pd) < 20: break
        if len(rows) < 100 or page >= 20: break
        page += 1
    print(time.strftime("%m-%d", time.localtime(t)), "posts", n_posts, "comments", n_push, flush=True)
    t = t2
print("DONE posts", n_posts, "comments", n_push)
