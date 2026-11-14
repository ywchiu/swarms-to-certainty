"""Shared paths and loaders. Reuses the Swarm-sprint sandbox (prior work, disclosed): data and detector code
live in ~/hackathon/apart_swarm and are imported read-only."""
import hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIOR = Path(os.environ.get("PRIOR_DIR", Path.home() / "hackathon" / "apart_swarm"))
sys.path.insert(0, str(PRIOR / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # a bundled llm.py (public release) takes precedence over the prior client
import detect as D  # noqa: E402  (prior work: PayloadConv / D4c, canon_targets, COLLIDE fix)
import llm  # noqa: E402  (prior work: H200 Gemma client with disk cache; GEN_BACKEND=agy for Gemini)

WIN = {"dev": PRIOR / "data", "heldout": PRIOR / "data" / "heldout"}
DATA = ROOT / "data"

# Brand aliases (strings that name the chain). Short aliases that collide with everyday words are handled by
# masking D.COLLIDE (e.g. 可不可以, 都可以) before matching.
BRANDS = {
    "迷客夏": ["迷客夏", "Milksha", "milksha"],
    "50嵐": ["50嵐", "五十嵐", "50 嵐"],
    "八曜和茶": ["八曜和茶", "八曜", "8yotea"],
    "可不可": ["可不可熟成紅茶", "可不可", "KEBUKE", "kebuke"],
    "星巴克": ["星巴克", "Starbucks", "starbucks"],
    "茶湯會": ["茶湯會"],
    "萬波": ["萬波"],
    "再睡5分鐘": ["再睡5分鐘", "再睡五分鐘"],
    "清心福全": ["清心福全", "清心"],
    "CoCo都可": ["CoCo", "coco", "都可"],
    "麻古茶坊": ["麻古茶坊", "麻古"],
    "得正": ["得正"],
    "路易莎": ["路易莎", "Louisa"],
    "春水堂": ["春水堂"],
}


def load(win):
    org = json.load(open(WIN[win] / "organic_msgs.json"))
    cl = json.load(open(WIN[win] / "claims.json"))
    return org, cl


def masked(text):
    for w in D.COLLIDE:
        text = text.replace(w, " ")
    return text


def mentions(text, aliases):
    t = masked(text)
    return any(a in t for a in aliases)


_SALT = os.environ.get("HANDLE_SALT", "").encode() or os.urandom(16)


def handle(account):
    """Stable pseudonymous handle shown to the briefing model. Swarm and organic accounts get the same format,
    so the account ID never leaks the label. The salt is per-run unless HANDLE_SALT is set (never written)."""
    return "@u" + hashlib.blake2b(account.encode(), key=_SALT, digest_size=3).hexdigest()
