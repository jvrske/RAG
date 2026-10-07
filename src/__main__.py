import fire
import json
from pydantic import ValidationError
from pathlib import Path
from tqdm import tqdm
from src import indexer
from src.models import MinimalSearchResults, RagDataset, StudentSearchResults
from src.retriever import search as search_chunks
from src.evaluation import recall_at_k


def index(max_chunk_size: int = 2000) -> None:
    """Build the search index over data/raw/ and persist it."""

    if max_chunk_size < 1:
        print(f"Error: chunk size must be a valid "
              f"number (received {max_chunk_size})")
        return

    if max_chunk_size > 2000:
        print(f"Error: chunk size can't be greater than "
              f"2000 (received {max_chunk_size})")
        return

    chunks, retriever = indexer.build(max_chunk_size)
    indexer.save(chunks, retriever)
    print(f"Indexed {len(chunks)} chunks into {indexer.PROCESSED_DIR}")


def search(query: str, k: int = 10) -> None:
    """Print the top-k sources for a single query."""

    if k < 1:
        print(f"Error: k must have at least 1 (received {k})")
        return

    query = str(query).strip()
    if not query:
        print("Error: query is empty")
        return

    try:
        chunks, retriever_index = indexer.load()
    except (FileNotFoundError, json.JSONDecodeError) as error:
        print(f"Error: {error}")
        return

    sources = search_chunks(query, k, chunks, retriever_index)
    if not sources:
        print("No relevant sources found")
        return
    for rank, source in enumerate(sources, 1):
        print(f"{rank}. {source.file_path} "
              f"[{source.first_character_index}, "
              f"{source.last_character_index}]")


def search_dataset(dataset_path: str, save_directory: str,
                   k: int = 10) -> None:
    """Search every question of a dataset and write the results as JSON."""

    if k < 1:
        print(f"Error: k must have at least 1 (received {k})")
        return

    try:
        with open(dataset_path, encoding="utf-8") as f:
            dataset = RagDataset(**json.load(f))
        chunks, retriever = indexer.load()
    except (FileNotFoundError, json.JSONDecodeError,
            ValidationError) as error:
        print(f"Error: {error}")
        return

    results = []
    for question in tqdm(dataset.rag_questions, desc="searching"):
        sources = search_chunks(question.question, k, chunks, retriever)
        results.append(MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=sources,
        ))

    student = StudentSearchResults(k=k, search_results=results)

    directory = Path(save_directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(dataset_path).stem
    parent = Path(dataset_path).parent.name
    output_path = directory / f"{parent}_{stem}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(student.model_dump(), f, indent=2)

    print(f"Wrote {len(student.search_results)} results "
          f"(k={student.k}) to {output_path}")


def answer(query: str, k: int = 10) -> None:
    """Answer a single query from the sources retrieved for it."""

    if k < 1:
        print(f"Error: k must have at least 1 (received {k})")
        return

    query = str(query).strip()
    if not query:
        print("Error: query is empty")
        return

    print(f"[answer] query={query!r} k={k}")


def answer_dataset(student_search_results_path: str,
                   save_directory: str) -> None:
    """Generate an answer for every question of a search-results file."""

    print(f"[answer_dataset] "
          f"student_search_results_path={student_search_results_path} "
          f"save_directory={save_directory}")


def evaluate(student_search_results_path: str, dataset_path: str) -> None:
    """Report our own recall@k against a ground-truth dataset."""

    try:
        with open(student_search_results_path, encoding="utf-8") as f:
            student = StudentSearchResults(**json.load(f))

        with open(dataset_path, encoding="utf-8") as f:
            data_set = RagDataset(**json.load(f))

    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
        print(f"Error: {e}")
        return

    try:
        recall = recall_at_k(student, data_set)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"recall@{student.k} = {recall:.4f}")


fire.Fire({"index": index, "search": search,
           "search_dataset": search_dataset, "answer": answer,
           "answer_dataset": answer_dataset, "evaluate": evaluate})
