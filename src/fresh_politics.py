"""Sprint 2 (15 Nov): six fresh organic backgrounds for an independent replication.

The original experiments share two organic backgrounds (dev, heldout). Here we draw six more whole-thread samples of
the same size (TARGET messages each) from the same corpus: f1-f3 from the dev period, f4-f6 from the held-out period.
Threads already used by dev/heldout, or by an earlier fresh window, are excluded, so all eight backgrounds are
thread-disjoint. Each fresh window then gets one newly generated S4 swarm (swarm_politics.py <win> S4 0) and claims
for the subset the experiments read (messages naming a front-runner, their thread roots, and the swarm).

  python fresh_politics.py sample          # writes data/politics/f{1..6}/organic_msgs.json + organic_delays.json
  python fresh_politics.py claims f1,...   # after swarm generation: claims.json for the subset (H200 Gemma, temp 0)
"""
import asyncio, collections, json, os, random, sys, time
from common import DATA

P = DATA / "politics"
TARGET = int(os.environ.get("TARGET", 40000))
SPLIT = time.mktime(time.strptime("2026-11-03", "%Y-%m-%d"))
FRESH = {"f1": "dev", "f2": "dev", "f3": "dev", "f4": "heldout", "f5": "heldout", "f6": "heldout"}
NAMES = ["蔣萬安", "萬安", "蔣市長", "小蔣", "沈伯洋", "PUMA", "Puma", "puma", "沈伯"]


def load_corpus():
    """Same message construction as build_politics.py."""
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
    return msgs, posts


def sample():
    msgs, posts = load_corpus()
    threads = collections.defaultdict(list)
    for m in msgs:
        threads[m["post_id"]].append(m)
    used = set()
    for w in ("dev", "heldout"):
        used |= {m["post_id"] for m in json.load(open(P / w / "organic_msgs.json"))}
    rng = random.Random(2026)
    stats = {}
    for period, keep in (("dev", lambda t: t < SPLIT), ("heldout", lambda t: t >= SPLIT)):
        tids = sorted(t for t in threads if keep(float(posts[t]["tm"])) and t not in used)
        rng.shuffle(tids)
        for win in [w for w, p in FRESH.items() if p == period]:
            out, n = [], 0
            while tids and n < TARGET:
                t = tids.pop()
                out += threads[t]; n += len(threads[t])
            d = P / win; d.mkdir(exist_ok=True)
            json.dump(out, open(d / "organic_msgs.json", "w"), ensure_ascii=False)
            dl = sorted(m["tm"] - int(float(posts[m["post_id"]]["tm"])) for m in out if m["kind"] == "comment")
            json.dump([x for x in dl if 0 < x < 30 * 86400], open(d / "organic_delays.json", "w"))
            acc = collections.Counter(m["account"] for m in out)
            stats[win] = {"period": period, "threads_sampled": len({m["post_id"] for m in out}), "msgs": len(out),
                          "accounts": len(acc), "accounts_ge2": sum(v >= 2 for v in acc.values())}
            print(win, stats[win], flush=True)
    json.dump(stats, open(P / "fresh_stats.json", "w"), ensure_ascii=False, indent=1)


def claims(wins):
    import ast  # extract_politics.py runs on import, so read its PROMPT constant without executing it
    tree = ast.parse(open(os.path.join(os.path.dirname(__file__), "extract_politics.py")).read())
    prompt = next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "PROMPT")
    for win in wins:
        os.environ["DATA_DIR"] = str(P / win)
        sys.modules.pop("extract", None)
        import extract  # prior work; DATA is read from DATA_DIR at import
        extract.PROMPT = prompt
        org = json.load(open(P / win / "organic_msgs.json"))
        sw = [m for f in sorted((P / win).glob("swarm_*_s[0-9].json")) for m in json.load(open(f))]
        sub = [m for m in org if any(a in m["text"] for a in NAMES)]
        roots = {m["post_id"] for m in sub if m["kind"] == "comment"}
        ids = {m["msg_id"] for m in sub}
        sub += [m for m in org if m["kind"] == "post" and m["msg_id"] in roots and m["msg_id"] not in ids]
        done = asyncio.run(extract.extract_all(sub + sw))
        print(win, "subset", len(sub), "swarm", len(sw), "claims", len(done), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        sample()
    else:
        claims(sys.argv[2].split(","))
