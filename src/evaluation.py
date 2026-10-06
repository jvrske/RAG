"""My own recall@k, so the pipeline can be measured without the grader.

A retrieved source counts as a hit when it names the same file as the
gold source and overlaps it by at least IOU_THRESHOLD, the same rule
the subject states. Nothing here imports or shells out to the
moulinette: the grader is the exam, this module is the ruler we use
while building.
"""

from src.models import (AnsweredQuestion, MinimalSearchResults,
                        MinimalSource, RagDataset, StudentSearchResults)

IOU_THRESHOLD = 0.05


def iou(first_a: int, last_a: int, first_b: int, last_b: int) -> float:
    """Return how much two character ranges overlap, from 0.0 to 1.0.

    Intersection over union: the shared characters divided by the
    characters covered by either range. Dividing by the union is what
    punishes a range far larger than the one it is compared against,
    so returning a whole file cannot pass as a precise answer.

    A range whose last index does not come after its first one covers
    no text at all, so it scores 0.0 instead of raising: this function
    is fed indices parsed from JSON files we do not control.
    """

    if first_a >= last_a or first_b >= last_b:
        return 0.0

    intersection = min(last_a, last_b) - max(first_a, first_b)
    union = max(last_a, last_b) - min(first_a, first_b)
    iou_value = intersection / union

    if iou_value < 0:
        iou_value = 0.0
    return iou_value


def is_hit(retrieved: MinimalSource, gold: MinimalSource) -> bool:
    """Tell whether a retrieved source counts as finding a gold one.

    Both conditions must hold: the file paths are identical, with no
    tolerance, and the character ranges overlap by at least
    IOU_THRESHOLD. The path is compared first because most pairs live
    in different files, and a string comparison is far cheaper than
    the overlap.
    """

    if retrieved.file_path != gold.file_path:
        return False

    iou_value = iou(
        retrieved.first_character_index,
        retrieved.last_character_index,
        gold.first_character_index,
        gold.last_character_index,
    )

    return iou_value >= IOU_THRESHOLD


def recall_for_question(result: MinimalSearchResults,
                        gold_sources: list[MinimalSource],
                        k: int) -> float:
    """Return the recall of one question: gold sources found over all.

    Only the first k retrieved sources are considered, since the rest
    is what the user would never see. The outer loop walks the gold
    sources rather than the retrieved ones, so every gold source
    contributes at most once however many retrieved sources land on
    it, and a gold source nobody reached is never silently skipped.

    A question with no gold sources cannot be measured, so it scores
    0.0: a score that hides a malformed file is worse than one that
    exposes it.
    """

    if not gold_sources:
        return 0.0

    count = 0
    for gold in gold_sources:
        for retrieved in result.retrieved_sources[:k]:
            if is_hit(retrieved, gold):
                count += 1
                break

    return count / len(gold_sources)


def recall_at_k(student: StudentSearchResults,
                dataset: RagDataset) -> float:
    """Average recall@k over every question of the gold dataset.

    Questions the student never answered count as zero, so the
    denominator is the size of the gold dataset and not the size of
    the student's own file. Results carrying a question_id the dataset
    does not know are ignored instead of raising, and they change
    neither the numerator nor the denominator.

    The k comes from the student file, because that file records the k
    its own search was run with, and evaluate takes no k of its own.

    Raises:
        ValueError: if the dataset holds questions without sources,
            which means an UnansweredQuestions file was passed. That
            is a wrong command rather than a bad score, so it must not
            come back as a recall of zero.
    """

    if not dataset.rag_questions:
        return 0.0

    gold_by_id = {}
    for question in dataset.rag_questions:
        if not isinstance(question, AnsweredQuestion):
            raise ValueError(
                "dataset has questions without sources: it is an "
                "UnansweredQuestions file, not an AnsweredQuestions one"
            )
        gold_by_id[question.question_id] = question.sources

    total = 0.0
    for result in student.search_results:
        gold_sources = gold_by_id.get(result.question_id)
        if gold_sources is None:
            continue
        total += recall_for_question(result, gold_sources, student.k)

    return total / len(dataset.rag_questions)
