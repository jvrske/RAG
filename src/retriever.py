import re


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def score(query_tokens: list[str], chunk_tokens: list[str]) -> float:
    return len(set(query_tokens) & set(chunk_tokens))
