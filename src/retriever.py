import bm25s


def build_index(chunks):
    tokens = bm25s.tokenize([c.text for c in chunks], stopwords=None)
    retriever = bm25s.BM25()
    retriever.index(tokens)
    return retriever

def search(query, k, chunks, retriever):
    idx, _ = retriever.retrieve(bm25s.tokenize(query, stopwords=None), k=k)
    return [chunks[i] for i in idx[0]]
