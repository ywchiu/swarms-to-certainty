"""Async client for the self-hosted Gemma-4 31B (vLLM, OpenAI-compatible) with disk cache."""
import asyncio, hashlib, json, os, re
from pathlib import Path
import httpx
URL = os.environ.get("CHAT_URL", "http://127.0.0.1:8001/v1/chat/completions")
MODEL = os.environ.get("CHAT_MODEL", "H200-gemma-4-31B")
# Cross-family generation: GEN_BACKEND=agy routes chat() to Gemini through the `agy` CLI (no temperature / max_tokens control).
BACKEND = os.environ.get("GEN_BACKEND", "")
if BACKEND == "agy":
    MODEL = os.environ.get("AGY_MODEL", "gemini-3.8-flash-medium")
# GEN_BACKEND=codex routes chat() to an OpenAI model through `codex exec` (ChatGPT account; read-only sandbox in an empty dir).
if BACKEND == "codex":
    MODEL = os.environ.get("CODEX_MODEL", "gpt-6-luna")
# GEN_BACKEND=claude routes chat() to Anthropic through Claude Code (`claude -p`): our system prompt replaces Claude Code's,
# no tools, no MCP, no settings, no session persistence (no temperature / max_tokens control).
if BACKEND == "claude":
    MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")
AGY_CWD = os.environ.get("AGY_CWD") or "/tmp"  # run agy from an empty directory so it never sees project files
CACHE = Path(__file__).resolve().parent.parent / "data" / "llm_cache"
CACHE.mkdir(parents=True, exist_ok=True)
_SEMS = {}
def _sem():  # one semaphore per event loop (detect.py calls asyncio.run several times)
    loop = asyncio.get_running_loop()
    if loop not in _SEMS: _SEMS[loop] = asyncio.Semaphore(int(os.environ.get("LLM_CONC", "24")))
    return _SEMS[loop]

def _key(msgs, temp, max_tokens, salt):
    return hashlib.sha256(json.dumps([msgs, temp, max_tokens, salt, MODEL], ensure_ascii=False).encode()).hexdigest()

async def chat(client, msgs, temp=0.9, max_tokens=300, salt=""):
    k = _key(msgs, temp, max_tokens, salt); f = CACHE / f"{k}.json"
    if f.exists():
        return json.loads(f.read_text())["text"]
    if BACKEND == "agy":
        return await _agy_chat(msgs, f)
    if BACKEND == "codex":
        return await _codex_chat(msgs, f)
    if BACKEND == "claude":
        return await _claude_chat(msgs, f)
    async with _sem():
        for i in range(4):
            try:
                body = {"model": MODEL, "messages": msgs, "temperature": temp, "max_tokens": max_tokens}
                if os.environ.get("CHAT_TOKENS_FIELD"):  # e.g. max_completion_tokens for OpenAI reasoning models
                    body[os.environ["CHAT_TOKENS_FIELD"]] = body.pop("max_tokens")
                body |= json.loads(os.environ.get("CHAT_EXTRA", "{}"))  # e.g. {"chat_template_kwargs": {"enable_thinking": false}}
                hdr = {"Authorization": f"Bearer {os.environ[os.environ['CHAT_KEY_ENV']]}"} if os.environ.get("CHAT_KEY_ENV") else {}
                r = await client.post(URL, json=body, headers=hdr, timeout=int(os.environ.get("CHAT_TIMEOUT", "120")))
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"].strip()
                f.write_text(json.dumps({"text": text}, ensure_ascii=False)); return text
            except Exception as e:
                if i == 3: raise
                await asyncio.sleep(2 * (i + 1))

async def _agy_chat(msgs, f):
    prompt = "\n\n".join(m["content"] if m["role"] == "user" else f"[系統設定]\n{m['content']}\n[任務]" for m in msgs)
    async with _sem():
        for i in range(4):
            try:
                p = await asyncio.create_subprocess_exec("agy", "-p", prompt, "--model", MODEL, "--print-timeout", "300s", cwd=AGY_CWD,
                                                         stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                out, _ = await asyncio.wait_for(p.communicate(), 400)
                text = out.decode().strip()
                if not text: raise ValueError("empty")
                f.write_text(json.dumps({"text": text}, ensure_ascii=False)); return text
            except Exception as e:
                if i == 3: raise
                await asyncio.sleep(3 * (i + 1))

async def _codex_chat(msgs, f):
    import tempfile
    prompt = "\n\n".join(m["content"] if m["role"] == "user" else f"[系統設定]\n{m['content']}\n[任務]" for m in msgs)
    prompt += "\n\n只輸出所要求的 JSON，不要執行任何指令或讀取任何檔案。"
    async with _sem():
        for i in range(4):
            try:
                with tempfile.TemporaryDirectory() as td:
                    out = Path(td) / "out.md"
                    p = await asyncio.create_subprocess_exec("codex", "exec", "-m", MODEL, "-s", "read-only", "-C", AGY_CWD,
                                                             "--skip-git-repo-check", "-o", str(out), "-",
                                                             stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL,
                                                             stderr=asyncio.subprocess.DEVNULL)
                    await asyncio.wait_for(p.communicate(prompt.encode()), 600)
                    text = out.read_text().strip() if out.exists() else ""
                if not text: raise ValueError("empty")
                f.write_text(json.dumps({"text": text}, ensure_ascii=False)); return text
            except Exception as e:
                if i == 3: raise
                await asyncio.sleep(3 * (i + 1))

async def _claude_chat(msgs, f):
    sys_p = "\n\n".join(m["content"] for m in msgs if m["role"] == "system") or "You are a careful analyst."
    user = "\n\n".join(m["content"] for m in msgs if m["role"] != "system")
    async with _sem():
        for i in range(4):
            try:
                p = await asyncio.create_subprocess_exec("claude", "-p", "--model", MODEL, "--system-prompt", sys_p, "--tools", "",
                                                         "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
                                                         "--output-format", "json", cwd=AGY_CWD, stdin=asyncio.subprocess.PIPE,
                                                         stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
                out, _ = await asyncio.wait_for(p.communicate(user.encode()), 600)
                d = json.loads(out.decode())
                text = (d.get("result") or "").strip()
                if d.get("is_error") or not text: raise ValueError("empty or error")
                f.write_text(json.dumps({"text": text}, ensure_ascii=False)); return text
            except Exception as e:
                if i == 3: raise
                await asyncio.sleep(3 * (i + 1))

def parse_json(text):
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    s = m.group(1) if m else text
    i = min([p for p in (s.find("{"), s.find("[")) if p >= 0], default=-1)
    if i < 0: raise ValueError("no json")
    s = s[i:]
    for end in range(len(s), 0, -1):
        try: return json.loads(s[:end])
        except Exception: continue
    raise ValueError("bad json")

async def run_all(coros_fn):
    async with httpx.AsyncClient() as client:
        return await coros_fn(client)
