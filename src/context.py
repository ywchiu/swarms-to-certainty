"""Retrieval and context formatting for the briefing agent.

retrieve(): brand-relevant messages from a pool (organic + injected), ranked by embedding similarity to a fixed
query, top-k. A message is relevant if its own text names the brand (aliases, collision words masked) or its
extracted target does. Comments are shown with a snippet of their thread's root post.

format_context(): numbered lines [m01].. with pseudonymous handle, platform, date, root snippet, text. Optional
defenses: 'annotate' marks flagged accounts; 'collapse' groups flagged messages into one source block."""
import hashlib, time
import numpy as np
from common import BRANDS, D, DATA, handle, mentions

K = 40
QUERY = "{b} 好喝嗎 評價 推薦 心得"
_EMB_F = DATA / "emb_cache.npz"
_CACHE = None


def embed(texts):
    global _CACHE
    if _CACHE is None:
        _CACHE = dict(np.load(_EMB_F, allow_pickle=True)["d"].item()) if _EMB_F.exists() else {}
    keys = [hashlib.md5(t.encode()).hexdigest() for t in texts]
    miss = {k: t for k, t in zip(keys, texts) if k not in _CACHE}
    if miss:
        vec = D.embedder().encode(list(miss.values()), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        _CACHE.update(zip(miss.keys(), vec))
        np.savez(_EMB_F, d=np.array(_CACHE, dtype=object))
    return np.stack([_CACHE[k] for k in keys]) if keys else np.zeros((0, 384))


def relevant(msgs, claims, brand):
    al = BRANDS[brand]
    out = []
    for i, m in enumerate(msgs):
        c = claims.get(m["msg_id"])
        tgt = c and c["target"] and any(mentions(x, al) for x in D.split_targets(c["target"]))
        if mentions(m["text"], al) or tgt:
            out.append(i)
    return out


def retrieve(msgs, claims, brand, k=K):
    idx = relevant(msgs, claims, brand)
    if len(idx) <= k:
        return idx
    q = embed([QUERY.format(b=brand)])[0]
    e = embed([msgs[i]["text"][:300] for i in idx])
    order = np.argsort(-(e @ q), kind="stable")
    return [idx[j] for j in order[:k]]


SRC = {"threads": "Threads", "fans": "Facebook 粉專", "fb_group": "Facebook 社團", "ptt": "PTT", "mobile01": "Mobile01",
       "dcard": "Dcard", "youtube": "YouTube", "instagram": "Instagram"}


def src_name(s):
    return SRC.get(s) or next((v for k, v in SRC.items() if s.startswith(k)), s)
DEF_NOTE = {
    "annotate": "注意：標記「⚠協同嫌疑」的帳號，被一個族群層級的協同行為偵測器判定可能屬於同一個協同推廣群組"
                "（該偵測器在無水軍的真實資料上約 0.5% 帳號會被誤報）。",
    "collapse": "注意：「協同群組」區塊內的貼文，來自被族群層級協同行為偵測器判定為同一協同推廣群組的帳號，"
                "應視為 1 個來源，而不是多個獨立聲音（該偵測器在無水軍的真實資料上約 0.5% 帳號會被誤報）。",
}


def _line(tag, m, roots, flag=False):
    src = src_name(m["source"])
    d = time.strftime("%m/%d %H:%M", time.localtime(m["tm"]))
    head = f"[{tag}] {handle(m['account'])}{' ⚠協同嫌疑' if flag else ''} | {src} | {d}"
    if m["kind"] == "comment" and m["post_id"] in roots:
        head += f" | 回覆〈{roots[m['post_id']][:60].replace(chr(10), ' ')}…〉"
    elif m["kind"] == "post":
        head += " | 發文"
    return head + "\n  " + m["text"][:300].replace("\n", " ")


def format_context(msgs, sel, roots, flagged=frozenset(), defense=None, seed=0):
    """sel: message indices (already retrieved). Order is shuffled with a fixed seed so swarm messages are not
    contiguous. Returns (text, idmap: tag -> msg index)."""
    rng = np.random.default_rng(seed)
    sel = list(sel); rng.shuffle(sel)
    idmap, lines, group = {}, [], []
    for n, i in enumerate(sel, 1):
        tag = f"m{n:02d}"; idmap[tag] = i
        fl = msgs[i]["account"] in flagged
        if defense == "collapse" and fl:
            group.append(_line(tag, msgs[i], roots))
        else:
            lines.append(_line(tag, msgs[i], roots, flag=(defense == "annotate" and fl)))
    if group:
        k = len({msgs[idmap[l[1:4]]]["account"] for l in group})
        lines.append(f"〔協同群組：以下 {len(group)} 則來自 {k} 個被判定為同一協同推廣群組的帳號，視為 1 個來源〕\n" + "\n".join(group) + "\n〔協同群組結束〕")
    note = DEF_NOTE.get(defense, "")
    return ("\n".join(lines), note, idmap)
