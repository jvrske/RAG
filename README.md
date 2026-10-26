*This project has been created as part of the 42 curriculum by csilva.*

# RAG Against the Machine

A Retrieval-Augmented Generation pipeline over the vLLM codebase: it
indexes the corpus, retrieves the passages most likely to answer a
question, hands them to a local Qwen3-0.6B, and measures retrieval
quality with recall@k.

## Description

A language model knows nothing about code released after its training.
Instead of retraining it, this pipeline looks the answer up at the
moment the question is asked.

```
index      data/raw/ -> chunks -> BM25 index -> data/processed/
search     question -> top-k passages
answer     question + passages -> grounded answer
evaluate   our results vs ground truth -> recall@k
```

## Instructions

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

### Setup

```bash
uv sync
```

The corpus and datasets are not tracked here. Unzip them into:

```
data/raw/          the corpus
data/datasets/     the question datasets
```

### Usage

```bash
# Build the index (~4s)
uv run python -m src index --max_chunk_size 2000

# One question
uv run python -m src search "How do I load a LoRA adapter?" --k 10
uv run python -m src answer  "How do I load a LoRA adapter?" --k 10

# A whole dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output

uv run python -m src answer_dataset \
    --student_search_results_path data/output/<results>.json \
    --save_directory data/output

# Recall@k against ground truth
uv run python -m src evaluate \
    --student_search_results_path data/output/<results>.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

### Makefile

| rule | does |
|---|---|
| `make install` | install dependencies |
| `make run` | run the main script |
| `make debug` | run under `pdb` |
| `make lint` | flake8 + mypy |
| `make clean` | remove `data/processed/` and Python caches |

`make clean` never touches `data/raw/`.

## System architecture

```
data/raw/ --> corpus.build() --> chunking --> chunks ------+
                                                |          |
                                    retriever.build_index()|
                                                |          |
                                                v          v
                                        indexer.save() -> data/processed/
                                                              |
                                              indexer.load() -+
                                                              |
         question --> retriever.search() --> sources ---------+
                                                 |
                              generator.generate()
                                                 |
                                                 v
                                            answer text
```

| module | responsibility |
|---|---|
| `models.py` | pydantic models for every JSON boundary |
| `corpus.py` | reading files off disk |
| `chunking.py` | cutting files into passages |
| `retriever.py` | BM25 index and ranking |
| `indexer.py` | assembling and persisting the index |
| `generator.py` | prompting the local model |
| `evaluation.py` | recall@k |
| `__main__.py` | CLI (Python Fire) |

Modules return, `__main__` prints. The index lives on disk because each
command is a separate process: `chunks.json` holds the chunk list in
indexing order, `bm25/` holds the BM25 index. A search returns
positions into that list, so its order is the contract.

## Chunking strategy

Two strategies, chosen by file extension.

**Markdown — by section.** Cuts along headings, so a chunk carries one
topic. Headings inside ```` ``` ```` fences are ignored (151 of 1424
candidates in this corpus were shell comments). Sections under 200
characters merge into the next one. Sections over `max_chunk_size` fall
back to the window chunker, with indices shifted back to file
coordinates.

**Everything else — fixed window.** Overlapping windows of
`max_chunk_size` characters with 200 of overlap, so an answer across a
boundary survives in one of them.

Measured on the 100 docs gold passages, median IoU between the best
chunk and the gold:

| strategy on markdown | median IoU | IoU > 0.95 |
|---|---|---|
| fixed window | 0.47 | 10/100 |
| by section | 0.89 | 42/100 |

The datasets were built by splitting documentation on its sections, so
cutting the same way recovers nearly the same passages.

## Retrieval method

**BM25** via [`bm25s`](https://github.com/xhluca/bm25s), defaults
`k1=1.5`, `b=0.75`, Lucene variant.

```
                        tf
score = Σ idf(t) × ─────────────────────────
                   tf + k1·(1 - b + b·len/avg)
```

- **idf** weights rare terms higher: `is` appears in 66% of chunks,
  `dynamically` in 0.3%.
- **k1** saturates repetition: the 10th occurrence adds little.
- **b** normalises by length, so a long chunk does not win by accident.

Defaults were kept: sweeping `k1` and `b` moved recall@5 by one or two
questions in a hundred, which is noise.

**Tokenisation** keeps the library defaults, including underscores
inside a token. Splitting `trust_remote_code` into three tokens halves
code recall (0.657 → 0.313), because questions cite identifiers
verbatim.

## Performance analysis

| recall | docs | code |
|---|---|---|
| @1 | 0.580 | 0.414 |
| @3 | 0.780 | 0.566 |
| **@5** | **0.820** | **0.636** |
| @10 | 0.890 | 0.667 |

How it got there:

| scoring | docs@10 | code@10 |
|---|---|---|
| counting shared words | 0.700 | 0.283 |
| + IDF weighting | 0.780 | 0.646 |
| full BM25 | 0.890 | 0.667 |

IDF is what rescued code: source files are dense with `self`, `return`
and `import`, which a plain word count mistakes for evidence.

| operation | time |
|---|---|
| index 1969 files → 12549 chunks | ~4 s |
| retrieve 100 questions | < 1 s |
| one answer, k=10 | ~1 min |
| answer a 199-question dataset | ~50 min |

Where the docs errors are — position of the correct passage:

```
1st 58   2nd 16   3rd 4   4th 3   5th 1   6-10th 7   11-50th 4   never 7
```

The seven never found are a vocabulary gap: 65% of query terms appear
in the correct passage for docs but only 41% for code, since questions
are English and code is Python identifiers. No lexical method closes
that; embeddings would.

## Design decisions

- **BM25 by library, after writing it by hand.** The scoring function
  was first implemented from scratch, which is where the measurements
  above come from. `bm25s` replaced it: same recall, 75x faster search.
- **Chunk text stored in the index** (~24 MB), so later commands do not
  reopen the corpus. Where it is missing — a results file keeps only
  paths and indices — the generator recovers it by slicing the file,
  exactly as the grader does.
- **Guards describe the disaster.** Every validator and command opens
  by rejecting what cannot work, then proceeds.
- **Degenerate input is answered, not crashed on.** Empty query, `k=0`,
  missing index, malformed JSON and the wrong dataset type each print
  one `Error:` line and exit.
- **`torch` is imported lazily**, inside the model loader, so `search`
  and `search_dataset` do not pay for a model they never touch.
- **Private datasets were never used for tuning.** They are the only
  honest measurement available.

## Challenges faced

- **The grading rule had to be established first.** Probing the grader
  revealed the IoU threshold, the exact file-path match, and that one
  source over 2000 characters invalidates a whole submission.
- **Measuring ceilings before optimising.** Whole files reach 32% on
  code and 66% on docs; the current chunker can represent 96% of gold
  passages. That number showed the remaining gap was ranking, not
  chunking.
- **Silent bugs cost the most.** Forgetting to add the section offset
  back produces a chunk that is valid, whose text is right, that
  pydantic and mypy accept — and whose indices point elsewhere.
- **The model was the hard constraint, not the code.** The first run
  managed 0.4 tokens/s and froze the machine: a CUDA build of torch on
  a machine without a GPU, running from swap. Fixing that and switching
  to float32 gave 4x, taking a dataset run from ~6 hours to ~50 min.
- **One gold entry in the code dataset is broken** — twelve characters
  of whitespace, 50000 away from the real answer, marked
  `is_valid: true`. Worth about a point of recall, and unanswerable.

## Resources

- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25
  and Beyond*
- [Lucene scoring
  notes](https://lucene.apache.org/core/documentation.html) — why the
  Lucene IDF adds 1 inside the logarithm
- [`bm25s`](https://github.com/xhluca/bm25s)
- [pydantic](https://docs.pydantic.dev/) ·
  [Python Fire](https://github.com/google/python-fire) ·
  [transformers](https://huggingface.co/docs/transformers)

### How AI was used

Claude was used as a tutor, not as a code generator.

| used for | not used for |
|---|---|
| explaining concepts before I implemented them (IoU, recall@k, BM25, IDF, `pathlib`, `re`, lazy imports) | writing the modules in `src/` |
| running measurements: chunking ceilings, IoU distributions, document frequencies, tokeniser comparison, parameter sweeps | choosing what to implement |
| reviewing my code against adversarial input | — |
| explaining flake8 and mypy errors | — |

Bugs found during review and fixed by me: an inverted comparison in a
validator, a module-level accumulator doubling results on a second
call, an off-by-one dropping the last chunk, a lost section offset, and
iterating a pydantic model expecting values but getting field names.

The `bm25s` call sites and the prompt template were drafted with
Claude's help and then adjusted. No code was kept that I could not
explain.
