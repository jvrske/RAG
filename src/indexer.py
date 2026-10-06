"""Building the search index and keeping it on disk between commands.

Each CLI command runs in its own process, so nothing survives in memory
from one to the next: index must write what search will read back. Two
things are stored side by side, and they only make sense together — the
BM25 index ranks positions, and the chunk list is what turns a position
into a file path and a character range.
"""

import json
import bm25s
from pathlib import Path
from src import corpus
from src.chunking import chunk_markdown, chunk_text
from src.models import Chunk
from src.retriever import build_index

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
PROCESSED_DIR = ROOT / "data" / "processed"


def build(max_chunk_size: int = 2000) -> tuple[list[Chunk], bm25s.BM25]:
    """Read the corpus, cut it into chunks and index them.

    Markdown files are cut along their own headings and everything else
    into fixed windows, since a section carries one topic while a window
    only carries whatever happened to fall inside it.

    Returns both the chunks and the index because neither is usable
    alone: a search result is a position into this exact list.
    """

    crpus = corpus.build()
    chunks = []

    for path, text in crpus:
        if path.endswith(".md"):
            chunks.extend(chunk_markdown(path, text, max_chunk_size))
        else:
            chunks.extend(chunk_text(path, text, max_chunk_size))

    retriever = build_index(chunks)
    return chunks, retriever


def save(chunks: list[Chunk], retriever: bm25s.BM25,
         directory: Path = PROCESSED_DIR) -> None:
    """Write the chunks and the BM25 index under directory.

    The chunks go to chunks.json in indexing order, and the index to a
    bm25 subdirectory of its own, because bm25s spreads several files
    with fixed names and they would otherwise sit loose among ours.

    The chunk text is stored along with the indices: it costs a few tens
    of megabytes and saves every later command from reopening the corpus
    to recover the passage it already found.
    """

    directory.mkdir(parents=True, exist_ok=True)
    data = [chunk.model_dump() for chunk in chunks]

    with open(directory / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
    retriever.save(str(directory / "bm25"))


def load(directory: Path = PROCESSED_DIR
         ) -> tuple[list[Chunk], bm25s.BM25]:
    """Read back what save wrote, in the order it was indexed.

    The order is the whole contract: bm25s answers with positions, and
    reordering this list would silently return the wrong source for
    every query.

    Raises:
        FileNotFoundError: if either half of the index is missing, which
            means index was never run or its output was deleted.
        ValidationError: if a stored chunk no longer satisfies the
            MinimalSource rules, raised by pydantic on reconstruction.
    """

    chunks = []
    chunks_path = directory / "chunks.json"
    index_path = directory / "bm25"

    if not chunks_path.is_file() or not index_path.is_dir():
        raise FileNotFoundError(
            f"no index found in {directory}: run 'index' first"
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i in data:
        chunks.append(Chunk(**i))

    retriever = bm25s.BM25.load(str(index_path))
    return chunks, retriever
