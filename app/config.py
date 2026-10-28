"""Settings from environment variables / .env (see .env.example)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")

INSTITUTE_NAME = os.getenv("INSTITUTE_NAME", "Vidya Coaching Academy")
CONTACT = os.getenv("CONTACT", "the front office")
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "rag.db"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "change-me")
SAMPLE_DIR = BASE_DIR / "sample_docs"

# Chunking / retrieval
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "1.0"))  # BM25 score below this = "not in the documents"
MIN_COVERAGE = float(os.getenv("MIN_COVERAGE", "0.5"))  # share of question words that must match

# LLM for answers (any OpenAI-compatible API). Empty key = offline extractive answers.
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# Optional dense embeddings (OpenAI-compatible /embeddings). Empty key = BM25 only.
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://api.openai.com/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
