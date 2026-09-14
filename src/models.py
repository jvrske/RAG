"""Pydantic models for every JSON boundary of the RAG pipeline.

The school's datasets are read into RagDataset, our submissions are built as
StudentSearchResults and StudentSearchResultsAndAnswer. Validating at these
boundaries means a malformed source raises where it is built, not six minutes
later inside the grader.
"""

from pydantic import BaseModel, Field, model_validator
from typing import List
import uuid


class MinimalSource(BaseModel):
    """A passage of the corpus: one file plus a character range in it.

    The indices are offsets into the file's text, so text[first:last] is
    exactly the passage. The grader invalidates the whole submission if any
    single source is longer than 2000 characters, so that limit is enforced
    here. A source that breaks it cannot be built in the first place.
    """

    file_path: str
    first_character_index: int
    last_character_index: int

    @model_validator(mode="after")
    def max_length(self) -> "MinimalSource":
        """Reject spans that are empty, backwards, or over 2000 characters."""
        if self.last_character_index <= self.first_character_index:
            raise ValueError(
                "Second index couldn't be lower than the first one."
            )
        if self.last_character_index - self.first_character_index > 2000:
            raise ValueError("Size couldn't be higher than 2000.")
        return self


class UnansweredQuestion(BaseModel):
    """A question with nothing attached, the shape of the final datasets.

    question_id is only generated when we invent a question ourselves, the
    school's datasets always carry their own.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question shipped with its ground-truth sources and answer.

    Used during development: we measure recall against these, then run the
    exact same code over the unanswered datasets.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A whole dataset file, answered or unanswered."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """What our search returns for a single question.

    retrieved_sources is ranked: position 0 is our best guess, and Recall@k
    only ever looks at the first k entries.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """A search result with the generated answer attached."""

    answer: str


class StudentSearchResults(BaseModel):
    """The file written by search_dataset: one entry per question."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """The file written by answer_dataset: one entry per question."""

    search_results: List[MinimalAnswer]
    k: int
