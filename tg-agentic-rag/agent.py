"""The agent: Claude + tools, run in a loop until it has enough to answer."""
import ipaddress
import json
import logging
import socket
from urllib.parse import urlparse

import anthropic
import httpx
from bs4 import BeautifulSoup

import config
from rag import KnowledgeBase, chunk_text

log = logging.getLogger(__name__)

SYSTEM = """You are a research assistant inside a Telegram chat. You can search the user's \
private knowledge base (documents they uploaded) and fetch web pages.

How to work:
- For any question that might depend on the user's documents, call search_knowledge_base first.
- If the first results are weak or incomplete, rewrite the query (synonyms, narrower or broader \
terms) and search again. Break multi-part questions into separate searches.
- Use fetch_url only when the user gives a URL or the knowledge base clearly lacks the answer \
and a specific page would help. Use save_url_to_knowledge_base only if the user asks to save a page.
- Ground answers in retrieved text and cite sources as [filename]. If the knowledge base does \
not contain the answer, say so plainly instead of guessing.
- Keep replies concise and plain text (no markdown tables); Telegram is a chat app."""

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "Semantic search over the user's uploaded documents. Returns the top matching passages with source filenames.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query"},
                "top_k": {"type": "integer", "description": "Number of passages (1-10)", "default": config.TOP_K},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_sources",
        "description": "List the documents currently in the user's knowledge base.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "fetch_url",
        "description": "Download a public web page and return its readable text (truncated).",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "save_url_to_knowledge_base",
        "description": "Fetch a public web page and store it in the knowledge base for future searches.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
]


def _is_public_http_url(url: str) -> bool:
    """Basic SSRF guard: http(s) only, and the host must resolve to public IPs."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        for info in socket.getaddrinfo(p.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global:
                return False
    except socket.gaierror:
        return False
    return True


async def _fetch_page_text(url: str) -> str:
    if not _is_public_http_url(url):
        raise ValueError("URL must be a public http(s) address")
    async with httpx.AsyncClient(follow_redirects=False, timeout=20) as client:
        r = await client.get(url, headers={"User-Agent": "tg-rag-bot/1.0"})
        r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(" ").split())


class Agent:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        self.history: dict[int, list[dict]] = {}

    def forget(self, chat_id: int) -> None:
        self.history.pop(chat_id, None)

    async def _run_tool(self, chat_id: int, name: str, args: dict) -> str:
        import asyncio

        try:
            if name == "search_knowledge_base":
                k = max(1, min(int(args.get("top_k", config.TOP_K)), 10))
                hits = await asyncio.to_thread(self.kb.search, chat_id, args["query"], k)
                if not hits:
                    return "No results. The knowledge base may be empty."
                return json.dumps(hits, ensure_ascii=False)
            if name == "list_sources":
                srcs = await asyncio.to_thread(self.kb.list_sources, chat_id)
                return json.dumps(srcs) if srcs else "Knowledge base is empty."
            if name == "fetch_url":
                text = await _fetch_page_text(args["url"])
                return text[:8000] + (" …[truncated]" if len(text) > 8000 else "")
            if name == "save_url_to_knowledge_base":
                text = await _fetch_page_text(args["url"])
                n = await asyncio.to_thread(self.kb.add_document, chat_id, args["url"], text)
                return f"Saved {n} chunks from {args['url']}."
            return f"Unknown tool: {name}"
        except Exception as e:  # report errors to the model so it can adapt
            log.warning("tool %s failed: %s", name, e)
            return f"Error: {e}"

    async def answer(self, chat_id: int, user_text: str) -> str:
        past = self.history.setdefault(chat_id, [])
        messages = past + [{"role": "user", "content": user_text}]

        final = ""
        for _ in range(config.MAX_AGENT_STEPS):
            resp = await self.client.messages.create(
                model=config.MODEL,
                max_tokens=1500,
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})
            final = "".join(b.text for b in resp.content if b.type == "text").strip()

            if resp.stop_reason != "tool_use":
                break

            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    log.info("chat %s tool %s %s", chat_id, block.name, block.input)
                    out = await self._run_tool(chat_id, block.name, block.input)
                    results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": out}
                    )
            messages.append({"role": "user", "content": results})
        else:
            final = final or "I hit my step limit before finishing. Try a more specific question."

        # keep only plain text turns in memory (tool traces are rebuilt each question)
        past.append({"role": "user", "content": user_text})
        past.append({"role": "assistant", "content": final or "(no answer)"})
        del past[: -2 * config.HISTORY_TURNS]
        return final or "I couldn't produce an answer."
