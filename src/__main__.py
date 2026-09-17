import fire


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

    print(f"{max_chunk_size}")


def search(query: str, k: int = 10) -> None:
    """Print the top-k sources for a single query."""

    if k < 1:
        print(f"Error: k must have at least 1 (received {k})")
        return

    query = str(query).strip()
    if not query:
        print("Error: query is empty")
        return

    print(f"[search] query={query!r} k={k}")


def search_dataset(dataset_path: str, save_directory: str,
                   k: int = 10) -> None:
    """Search every question of a dataset and write the results as JSON."""

    if k < 1:
        print(f"Error: k must have at least 1 (received {k})")
        return

    print(f"[search_dataset] dataset_path={dataset_path} "
          f"save_directory={save_directory}")


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

    print(f"[evaluate] "
          f"student_search_results_path={student_search_results_path} "
          f"dataset_path={dataset_path}")


fire.Fire({"index": index, "search": search,
           "search_dataset": search_dataset, "answer": answer,
           "answer_dataset": answer_dataset, "evaluate": evaluate})
