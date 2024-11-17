import difflib

from matcher.longest_matches import find_best_matching_spans
from tokenizer.context_aware_tokenizer import SpanToken


Span = tuple[int, int]
SpanMovement = tuple[SpanToken, SpanToken]


def get_text_dif_spans(left_input: list, right_input: list) -> tuple[list[Span], list[Span]]:
    left = []
    right = []

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, left_input, right_input).get_opcodes():
        if tag == 'delete':
            left.append(tuple((i1, i2)))
        elif tag == 'insert':
            right.append(tuple((j1, j2)))
        elif tag == 'replace':
            left.append(tuple((i1, i2)))
            right.append(tuple((j1, j2)))

    left.sort(key=lambda span: span[0])
    right.sort(key=lambda span: span[0])

    return left, right


def get_text_deltas(left_tokens: list[SpanToken], right_tokens: list[SpanToken]) -> tuple[list[SpanToken], list[SpanToken], list[SpanMovement]]:
    left_token_ids = [token[2] for token in left_tokens]
    right_token_ids = [token[2] for token in right_tokens]
    left_spans, right_spans = get_text_dif_spans(left_token_ids, right_token_ids)
    left_tokens_dif = [
        left_tokens[i]
        for span in left_spans
        if span[1] - span[0] > 1
        for i in range(span[0], span[1])
    ]
    right_tokens_dif = [
        right_tokens[i]
        for span in right_spans
        if span[1] - span[0] > 1
        for i in range(span[0], span[1])
    ]
    left_token_ids_dif = [token[2] for token in left_tokens_dif]
    right_token_ids_dif = [token[2] for token in right_tokens_dif]
    matching_spans = find_best_matching_spans(left_token_ids_dif, right_token_ids_dif)
    subtractions = [
        token
        for i, token in enumerate(left_tokens_dif)
        if not any(
            span[0] <= i < span[0] + span[2]
            for span in matching_spans
        )
    ]
    additions = [
        token
        for i, token in enumerate(right_tokens_dif)
        if not any(
            span[0] <= i < span[0] + span[2]
            for span in matching_spans
        )
    ]
    movements = [
        tuple((
            tuple((
                left_tokens_dif[span[0]][0],
                left_tokens_dif[span[0] + span[2] - 1][1],
                [left_tokens_dif[i][2] for i in range(span[0], span[0] + span[2])]
            )),
            tuple((
                right_tokens_dif[span[1]][0],
                right_tokens_dif[span[1] + span[2] - 1][1],
                [right_tokens_dif[i][2] for i in range(span[1], span[1] + span[2])]
            ))
        ))
        for span in matching_spans
    ]

    return additions, subtractions, movements
