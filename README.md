# Agentic RAG Telegram Bot

A Telegram bot that answers questions from documents you send it. Claude runs as an
agent: it decides when to search, rewrites weak queries, searches again, and can fetch
web pages before answering.

```
Telegram ──> bot.py ──> agent.py (Claude tool-use loop)
                            ├─ search_knowledge_base ─> rag.py (ChromaDB, per-chat)
                            ├─ list_sources
                            ├─ fetch_url
                            └─ save_url_to_knowledge_base
```

## Setup

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy the token.
2. Install and configure:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env   # fill in TELEGRAM_BOT_TOKEN and ANTHROPIC_API_KEY
   python bot.py
   ```

## Usage

- Send a PDF/TXT/MD/CSV file; the bot chunks, embeds, and stores it.
- Ask questions in plain text. Sources are cited as [filename].
- `/sources`, `/delete <name>`, `/forget`

## Notes

- Each chat has its own isolated knowledge base (filtered by chat ID).
- Embeddings use Chroma's default local MiniLM model (downloaded on first use).
- Set `ALLOWED_USER_IDS` to restrict access; otherwise anyone who finds the bot can use your API key.
- `fetch_url` blocks non-public addresses as a basic SSRF guard.
- To scale: swap Chroma for pgvector/Qdrant, add a reranker, and move chat history to Redis.
