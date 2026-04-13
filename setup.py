"""
First-run setup: chunks all documents in documents/, embeds them with
all-MiniLM-L6-v2, and persists the vector index to data/chromadb.

Standalone:  python setup.py
From app.py: import setup; setup.run()
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))

from embed_and_store import embed_and_store


def run() -> None:
    """Build the ChromaDB index from documents/. Safe to call repeatedly."""
    embed_and_store(
        documents_dir=str(_ROOT / "documents"),
        db_path=str(_ROOT / "data" / "chromadb"),
    )


if __name__ == "__main__":
    run()
