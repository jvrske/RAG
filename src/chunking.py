"""Cutting of corpus files into the chunks the search actually ranks.

A whole file is too coarse to be a search result: the passage that answers a
question is a few percent of it, so returning the file dilutes the IoU below
the grader's threshold — and anything over 2000 characters invalidates the
submission outright.

Every chunk keeps the invariant that its text is exactly
original_text[first_character_index:last_character_index]. The indices are
positions in the whole file, never inside the chunk, because they are handed
to the grader as-is.
"""

from src.models import Chunk

MAX_CHUNK_SIZE = 2000
MIN_FRACTION = 0.05
MIN_SECTION_SIZE = 200
OVERLAP = 200


def chunk_text(file_path: str,
               text: str,
               max_chunk_size: int = MAX_CHUNK_SIZE,
               overlap: int = OVERLAP,
               min_fraction: float = MIN_FRACTION) -> list[Chunk]:
    """Cut one file's text into overlapping fixed-size windows.

    Windows start every (max_chunk_size - overlap) characters, so consecutive
    chunks share their edges. Without that overlap a cut lands inside a word
    and the identifier a question asks about survives in neither neighbour.

    `min_fraction` drops the tail windows that carry nothing but overlap: a
    5-character leftover competes for a top-5 slot while adding nothing the
    previous chunk does not already contain. It is a fraction rather than a
    fixed count because that leftover is proportional to the window — the
    trade-off is that a small window makes the filter vanish.

    Returns the chunks in file order; the last one is truncated at the end of
    the text so no index ever points past the file.
    """

    chunks = []
    min_chunk_size = int(max_chunk_size * min_fraction)
    window = max_chunk_size - overlap
    for init in range(0, len(text), window):
        end = min(init + max_chunk_size, len(text))
        if end - init < min_chunk_size:
            continue
        text_slice = text[init:end]

        c = Chunk(file_path=file_path,
                  first_character_index=init,
                  last_character_index=end,
                  text=text_slice)
        chunks.append(c)

    return chunks


def find_headings(text: str) -> list[int]:
    """Return the character index of every markdown heading in text.

    A heading is a line starting with '#' that is not inside a fenced
    code block, where '#' opens a comment instead.
    """
    pos = 0
    result = []
    in_code_block = False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_code_block = not in_code_block
        elif line.startswith("#") and not in_code_block:
            result.append(pos)
        pos += len(line) + 1
    return result


def find_sections(text: str,
                  min_section_size: int = MIN_SECTION_SIZE
                  ) -> list[tuple[int, int]]:
    """Return the (start, end) character range of every markdown section.

    A section runs from one heading to the next, the last one ending at
    the end of the file, and any text before the first heading becomes a
    section of its own. Sections shorter than min_section_size are merged
    into the following one, so a bare heading is never a chunk by itself.
    """
    bounds = find_headings(text)
    if not bounds or bounds[0] != 0:
        bounds = [0] + bounds
    bounds.append(len(text))

    sections = []
    start = bounds[0]
    for end in bounds[1:]:
        if end - start < min_section_size and end != len(text):
            continue
        if end > start:
            sections.append((start, end))
        start = end

    return sections


def chunk_markdown(file_path: str,
                   text: str,
                   max_chunk_size: int = MAX_CHUNK_SIZE) -> list[Chunk]:
    """Cut one markdown file into chunks that follow its own sections.

    Each heading starts a new chunk, so a chunk holds a whole topic
    instead of an arbitrary window. Sections longer than max_chunk_size
    fall back to chunk_text, whose indices are shifted back into the
    file's own coordinates.
    """
    chunks = []
    for start, end in find_sections(text):
        section = text[start:end]
        if len(section) <= max_chunk_size:
            chunks.append(Chunk(file_path=file_path,
                                first_character_index=start,
                                last_character_index=end,
                                text=section))
            continue
        for c in chunk_text(file_path, section, max_chunk_size):
            c.first_character_index += start
            c.last_character_index += start
            chunks.append(c)

    return chunks
