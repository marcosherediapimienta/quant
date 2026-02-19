# ── Embedding / RAG ────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200
RETRIEVER_K = 5

# ── LLM defaults ──────────────────────────────────────────────────
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.4

# ── Conversation memory ───────────────────────────────────────────
MAX_CONVERSATION_MESSAGES = 20
HISTORY_WINDOW = 10
MAX_CONTEXT_MSG_LENGTH = 500

# ── Query enhancement ─────────────────────────────────────────────
MAX_QUERY_ENHANCEMENTS = 3

# ── Source formatting ─────────────────────────────────────────────
SOURCE_PREVIEW_LENGTH = 200

# ── Code indexer limits (max lines captured per node type) ────────
MAX_CLASS_LINES = 60
MAX_FUNCTION_LINES = 40
MAX_METHOD_LINES = 30
MIN_METHOD_NAME_LENGTH = 4

# ── Code indexer exclusions ───────────────────────────────────────
DEFAULT_EXCLUDE_DIRS = [
    '__pycache__', '.git', 'venv', 'env', '.pytest_cache',
    'node_modules', 'dist', 'build', '.tox', 'htmlcov',
    '.mypy_cache', '.ruff_cache', 'eggs', '.eggs'
]

DEFAULT_EXCLUDE_FILES = [
    'setup.py', 'setup.cfg', 'conftest.py', 'manage.py',
    'wsgi.py', 'asgi.py', 'migrations'
]

# ── Keyword search scoring ────────────────────────────────────────
KEYWORD_SCORES = {
    'name': 10,
    'docstring': 7,
    'content': 3,
}

