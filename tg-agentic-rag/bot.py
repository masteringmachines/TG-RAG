import asyncio
import logging
import tempfile
from pathlib import Path

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config
from agent import Agent
from rag import KnowledgeBase, extract_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

kb = KnowledgeBase()
agent = Agent(kb)

SUPPORTED = {".txt", ".md", ".pdf", ".csv", ".json", ".log", ".py"}


def allowed(update: Update) -> bool:
    if not config.ALLOWED_USER_IDS:
        return True
    user = update.effective_user
    return bool(user and user.id in config.ALLOWED_USER_IDS)


async def reply(update: Update, text: str) -> None:
    # Telegram caps messages at 4096 chars; send plain text to avoid markup errors.
    for i in range(0, len(text), 4000):
        await update.effective_message.reply_text(text[i : i + 4000])


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    await reply(
        update,
        "Hi! Send me documents (PDF, TXT, MD, CSV...) and ask questions about them.\n\n"
        "/sources - list your documents\n"
        "/delete <name> - remove a document\n"
        "/forget - clear conversation memory",
    )


async def sources(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    srcs = await asyncio.to_thread(kb.list_sources, update.effective_chat.id)
    if not srcs:
        await reply(update, "Your knowledge base is empty. Send me a file.")
        return
    await reply(update, "\n".join(f"- {name} ({n} chunks)" for name, n in srcs.items()))


async def delete(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    if not ctx.args:
        await reply(update, "Usage: /delete <filename>")
        return
    name = " ".join(ctx.args)
    await asyncio.to_thread(kb.delete_source, update.effective_chat.id, name)
    await reply(update, f"Removed {name} (if it existed).")


async def forget(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    agent.forget(update.effective_chat.id)
    await reply(update, "Conversation memory cleared. Your documents are untouched.")


async def on_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    doc = update.message.document
    name = doc.file_name or "untitled.txt"
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED:
        await reply(update, f"Unsupported type {suffix or '(none)'}. Try: {', '.join(sorted(SUPPORTED))}")
        return
    if doc.file_size and doc.file_size > config.MAX_FILE_BYTES:
        await reply(update, "File is too large (20 MB limit).")
        return

    await update.effective_chat.send_action(ChatAction.TYPING)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(name).name
        tg_file = await ctx.bot.get_file(doc.file_id)
        await tg_file.download_to_drive(path)
        try:
            text = await asyncio.to_thread(extract_text, path)
            n = await asyncio.to_thread(kb.add_document, update.effective_chat.id, name, text)
        except Exception as e:
            log.exception("ingest failed")
            await reply(update, f"Couldn't read that file: {e}")
            return
    if n == 0:
        await reply(update, "I couldn't extract any text (scanned PDF?).")
    else:
        await reply(update, f"Added {name} ({n} chunks). Ask me anything about it.")


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    chat_id = update.effective_chat.id

    async def keep_typing():
        while True:
            await update.effective_chat.send_action(ChatAction.TYPING)
            await asyncio.sleep(4)

    typing = asyncio.create_task(keep_typing())
    try:
        answer = await agent.answer(chat_id, update.message.text)
    except Exception:
        log.exception("agent failed")
        answer = "Something went wrong while answering. Please try again."
    finally:
        typing.cancel()
    await reply(update, answer)


def main() -> None:
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("sources", sources))
    app.add_handler(CommandHandler("delete", delete))
    app.add_handler(CommandHandler("forget", forget))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    log.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
