"""BM25 retrieval over the indexed chunks, backed by the bm25s library."""

import bm25s
from src.models import Chunk, MinimalSource


def build_index(chunks: list[Chunk]) -> bm25s.BM25:
    """Build a BM25 index over the text of every chunk.

    The index stores positions into the chunks list, not the chunks
    themselves, so the caller must keep that list in the same order to
    be able to turn a search result back into a source.

    The tokenizer keeps bm25s' defaults: English stopwords are dropped
    and underscores stay inside a token, which keeps Python identifiers
    such as trust_remote_code whole. Splitting them measurably halves
    recall on the code dataset.
    """

    tokens = bm25s.tokenize([chunk.text for chunk in chunks])
    retriever = bm25s.BM25()
    retriever.index(tokens)
    return retriever


def search(query: str, k: int, chunks: list[Chunk],
           retriever: bm25s.BM25) -> list[MinimalSource]:
    """Return the k chunks the index ranks highest for the query.

    Always returns k results, even when nothing matches: bm25s fills
    the ranking with zero-scored documents rather than returning fewer.
    """

    indices, _ = retriever.retrieve(bm25s.tokenize(query, show_progress=False),
                                    k=k, show_progress=False)
    return [chunks[int(i)] for i in indices[0]]
