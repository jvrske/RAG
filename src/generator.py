from pathlib import Path

from src.models import MinimalSource

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 150
NO_SOURCES = "I could not find any source that answers this question."

PROMPT = """Answer the question using only the sources below.
Cite the sources you used by their number, like [1] or [2].
If the sources do not contain the answer, say so.

{sources}

Question: {question}
Answer:"""

_TOKENIZER = None
_MODEL = None
_FILES: dict[str, str] = {}


def _load() -> tuple:
    """Return the tokenizer and the model, loading them on first use.

    A run over a whole dataset calls generate hundreds of times, so the
    pair is kept in module state rather than rebuilt on each call.

    The weights are kept in float32 rather than the bfloat16 the config
    asks for: on a CPU without native bfloat16 the smaller type is
    emulated, and measuring showed float32 roughly twice as fast here.
    """

    global _TOKENIZER, _MODEL
    if _MODEL is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float32)
    return _TOKENIZER, _MODEL


def read_source_text(source: MinimalSource) -> str:
    """Return the passage a source points at, reading it from the corpus.

    A source that came back from a search carries its text already, but
    one parsed from a results file carries only a path and two indices:
    the text is dropped when the file is written, since it is the
    corpus that holds it. Reading it back here lets both work the same.

    Files are cached because a dataset run asks for the same handful of
    files thousands of times.
    """

    text = getattr(source, "text", None)
    if text:
        return str(text)

    if source.file_path not in _FILES:
        _FILES[source.file_path] = Path(source.file_path).read_text(
            encoding="utf-8", errors="ignore")

    return _FILES[source.file_path][
        source.first_character_index:source.last_character_index]


def build_prompt(question: str, sources: list[MinimalSource]) -> str:
    """Lay the retrieved sources and the question out as one prompt.

    Sources are numbered from one and carry their file path, so the
    model has something concrete to cite and the reader can check the
    citation against the corpus.
    """

    blocks = []
    for number, source in enumerate(sources, 1):
        text = read_source_text(source)
        blocks.append(f"[{number}] {source.file_path}\n{text}")
    return PROMPT.format(sources="\n\n".join(blocks), question=question)


def generate(question: str, sources: list[MinimalSource]) -> str:
    """Answer the question from the given sources.

    With no sources there is nothing to ground an answer on, so this
    says so instead of letting the model invent one: that is both the
    honest answer and seconds of generation saved.
    """

    if not sources:
        return NO_SOURCES

    tokenizer, model = _load()
    prompt = build_prompt(question, sources)

    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([chat], return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    answer = output[0][len(inputs.input_ids[0]):]
    return str(tokenizer.decode(answer, skip_special_tokens=True)).strip()
