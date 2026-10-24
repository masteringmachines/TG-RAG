import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

ALLOWED_USER_IDS = {
    int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()
}

MAX_AGENT_STEPS = 6          # max tool-use rounds per question
HISTORY_TURNS = 8            # past user/assistant exchanges kept per chat
CHUNK_SIZE = 900             # characters per chunk
CHUNK_OVERLAP = 150
TOP_K = 5
MAX_FILE_BYTES = 20 * 1024 * 1024  # Telegram bot download limit
