"""Raw InfoMiner election rows -> two message tables with the same schema as the Swarm sprint
(data/politics/{dev,heldout}/organic_msgs.json).

Windows split by the thread's root-post date: dev = 10/25-11/02, heldout = 11/03-11/10 (Taipei time).
The full corpus (~0.5M messages) is ~15x the tea windows, so each window is a random sample of WHOLE threads
(fixed seed) until it reaches TARGET messages. Sampling whole threads keeps reply structure intact, which the
detector's unsolicited gate needs, and keeps the 0.5% alert threshold on the same population scale as the
prior work. The sample is drawn once, before any swarm exists, and never re-drawn."""
import collections, json, os, random, time
from common import DATA

P = DATA / "politics"
TARGET = int(os.environ.get("TARGET", 40000))
SPLIT = time.mktime(time.strptime("2026-11-03", "%Y-%m-%d"))

rows = [json.loads(l) for l in open(P / "organic_raw.jsonl")]
posts = {r["id"]: r for r in rows if r["kind"] == "post"}
msgs, seen = [], set()
for r in rows:
    if r["id"] in seen: continue
    seen.add(r["id"])
    text = ((r.get("title") or "") + "\n" + (r.get("content") or "")).strip() if r["kind"] == "post" else (r.get("content") or "").strip()
    if not text or text == "None": continue
    src = r.get("source") if r["kind"] == "post" else r.get("post_source")
    aid = r.get("author_id") or r.get("author")
    if not aid or aid == "None": continue
    try: tm = int(float(r.get("tm")))
    except Exception: continue
    pid = r["id"] if r["kind"] == "post" else r["post_id"]
    if pid not in posts: continue
    msgs.append(dict(msg_id=r["id"], account=f"{src}:{aid}", text=text[:600], tm=tm, post_id=pid,
                     kind=r["kind"], source=src, label=0, cond="organic"))
print("full corpus: msgs", len(msgs), "threads", len({m["post_id"] for m in msgs}),
      "accounts", len({m["account"] for m in msgs}))

threads = collections.defaultdict(list)
for m in msgs:
    threads[m["post_id"]].append(m)
rng = random.Random(0)
stats = {}
for win, keep in (("dev", lambda t: t < SPLIT), ("heldout", lambda t: t >= SPLIT)):
    tids = sorted(t for t in threads if keep(float(posts[t]["tm"])))
    rng.shuffle(tids)
    out, n = [], 0
    for t in tids:
        if n >= TARGET: break
        out += threads[t]; n += len(threads[t])
    d = P / win; d.mkdir(exist_ok=True)
    json.dump(out, open(d / "organic_msgs.json", "w"), ensure_ascii=False)
    acc = collections.Counter(m["account"] for m in out)
    dl = sorted(m["tm"] - int(float(posts[m["post_id"]]["tm"])) for m in out if m["kind"] == "comment")
    dl = [x for x in dl if 0 < x < 30 * 86400]
    json.dump(dl, open(d / "organic_delays.json", "w"))
    stats[win] = {"threads_in_window": len(tids), "threads_sampled": len({m["post_id"] for m in out}), "msgs": len(out),
                  "posts": sum(m["kind"] == "post" for m in out), "accounts": len(acc),
                  "accounts_ge2": sum(v >= 2 for v in acc.values()),
                  "by_source": dict(collections.Counter(m["source"] for m in out).most_common())}
    print(win, stats[win])
json.dump(stats, open(P / "build_stats.json", "w"), ensure_ascii=False, indent=1)
