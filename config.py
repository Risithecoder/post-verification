"""
config.py
─────────
Centralised application settings loaded from environment variables.

All secrets and tunables live here so no module ever reads os.environ directly.
"""

import os
from dotenv import load_dotenv

# Load .env file (no-op in production where env vars are injected by the platform)
load_dotenv()

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.2")
OPENAI_SEGMENT_MODEL: str = os.getenv("OPENAI_SEGMENT_MODEL", "gpt-4.1-mini")
OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

# ── Upload constraints ────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "5"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

# ── Chunking ──────────────────────────────────────────────────────────────────
# Approximate token budget per OpenAI call.  We estimate 1 token ≈ 4 characters.
CHUNK_TOKEN_LIMIT: int = int(os.getenv("CHUNK_TOKEN_LIMIT", "2000"))
CHARS_PER_TOKEN: int = 4  # conservative estimate for English text

# ── Allowed MIME type for .docx uploads ───────────────────────────────────────
ALLOWED_CONTENT_TYPE: str = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
