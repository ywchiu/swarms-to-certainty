"""Claim extraction for the election windows (H200 Gemma, temp 0), reusing the prior extractor's code path with a
political prompt: who does the message support or attack (politician / candidate / party). The output schema is
unchanged ({target, stance, claim}), so PayloadConv/D4c run as-is. The fictional candidate's swarm messages are
extracted later with this exact prompt.

  python extract_politics.py dev heldout
"""
import asyncio, json, os, sys
from common import DATA

P = DATA / "politics"
PROMPT = """你是社群內容分析器。讀下面這則台灣社群留言，判斷它主要在「表態支持、推薦或批評某位具體的政治人物／候選人／政黨」。
輸出 JSON：{{"target": "被表態的政治人物/候選人/政黨名稱（盡量寫成正式全名；沒有就空字串）", "stance": 1 或 0 或 -1（對該 target 的態度：支持為 1、批評為 -1；沒有 target 則 0）, "claim": "對該 target 的核心主張，15字內；沒有就空字串"}}
網路常見暱稱：「PUMA」指沈伯洋，「蔣市長」「萬安」指蔣萬安；其他暱稱請依上下文判斷，不確定就照原文寫。
只輸出 JSON。
{context}
留言：{text}"""

for win in sys.argv[1:] or ["dev", "heldout"]:
    os.environ["DATA_DIR"] = str(P / win)
    sys.modules.pop("extract", None)
    import extract  # prior work; DATA is read from DATA_DIR at import
    extract.PROMPT = PROMPT
    msgs = json.load(open(P / win / "organic_msgs.json"))
    for f in sorted((P / win).glob("swarm_*_s[0-9].json")):
        msgs += json.load(open(f))
    done = asyncio.run(extract.extract_all(msgs))
    pos = sum(1 for c in done.values() if c["stance"] > 0)
    print(win, "claims", len(done), "with target", sum(1 for c in done.values() if c["target"]), "positive", pos, flush=True)
