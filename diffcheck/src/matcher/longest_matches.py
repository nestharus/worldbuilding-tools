SpanMatch = tuple[int, int, int]


def find_matching_spans(
    left_elements: list,
    right_elements: list
) -> list[tuple[int, int, int]]:
    matching_spans = []

    left_index = 0
    while left_index < len(left_elements):
        element = left_elements[left_index]

        right_index = 0
        while right_index < len(right_elements):
            while right_index < len(right_elements) and right_elements[right_index] != element:
                right_index += 1

            length = 1
            while left_index + length < len(left_elements) and right_index + length < len(right_elements) and left_elements[left_index + length] == right_elements[right_index + length]:
                length += 1

            if length > 1:
                matching_spans.append(tuple((
                    left_index,
                    right_index,
                    length
                )))

            right_index += length

        left_index += 1

    return matching_spans


def find_best_matching_spans(
    left_tokens: list,
    right_tokens: list
) -> list[SpanMatch]:
    """Find by longest and closest"""
    matching_spans = []

    while left_tokens:
        all_matching_spans = find_matching_spans(left_tokens, right_tokens)

        if len(all_matching_spans) == 0:
            break

        all_matching_spans.sort(key=lambda match: (match[2], -abs(match[0] - match[1])), reverse=True)

        for span_left in all_matching_spans:
            if not any(
                (span_right[0] < span_left[0] + span_left[2]
                    and span_left[0] < span_right[0] + span_right[2]
                )
                or (span_right[1] < span_left[1] + span_left[2]
                    and span_left[1] < span_right[1] + span_right[2]
                )
                for span_right in matching_spans
            ):
                matching_spans.append(span_left)

        left_tokens = [
            token
            for i, token in enumerate(left_tokens)
            if not any(
                span[0] <= i < span[0] + span[2]
                for span in matching_spans
            )
        ]

        right_tokens = [
            token
            for i, token in enumerate(right_tokens)
            if not any(
                span[1] <= i < span[1] + span[2]
                for span in matching_spans
            )
        ]

    matching_spans.sort(key=lambda matching_span: matching_span[0])

    return matching_spans
