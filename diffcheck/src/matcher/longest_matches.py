from matcher.match import Match, find_token_matches
from tokenizer.token import Token
from tokenizer.token_sequence import TokenSequence


def find_longest_matches_from_token(
    left_index: int,
    left_tokens: list[Token],
    right_tokens: list[Token]
) -> list[Match]:
    left_token = left_tokens[left_index]

    match_positions = [
        right_index
        for right_index in find_token_matches(left_token.text, right_tokens)
    ]
    match_scores = [
        Match.compute_match_length(left_tokens, right_tokens, left_index, right_index)
        for right_index in match_positions
    ]
    right_sequence = [
        TokenSequence(right_tokens, right_index, match_score)
        for right_index, match_score in zip(match_positions, match_scores)
    ]
    right_sequences = [
        right_sequence
        for right_sequence in right_sequence
        if right_sequence.is_legal
    ]

    matches = [
        Match(
            TokenSequence(left_tokens, left_index, len(right_sequence)),
            right_sequence,
            abs(right_sequence.start_token_index - left_index)
        )
        for right_sequence in right_sequences
    ]

    return matches


def filter_remaining_tokens(
    all_matches: list[Match],
    best_matches: list[Match]
) -> tuple[list[Token], list[Token]]:
    best_left_tokens_by_range = {
        token.bounds: token
        for match in best_matches
        for token in match.left
    }

    best_right_tokens_by_range = {
        token.bounds: token
        for match in best_matches
        for token in match.right
    }

    left_tokens = [
        token
        for match in all_matches
        for token in match.left
        if token.bounds not in best_left_tokens_by_range
    ]

    right_tokens = [
        token
        for match in all_matches
        for token in match.right
        if token.bounds not in best_right_tokens_by_range
    ]

    return left_tokens, right_tokens


def find_best_matches(
    left_tokens: list[Token],
    right_tokens: list[Token]
) -> list[Match]:
    best_matches = []
    repeats = 0

    while left_tokens:
        all_matches = [
            match
            for left_index in range(len(left_tokens))
            for match in find_longest_matches_from_token(left_index, left_tokens, right_tokens)
        ]
        all_matches.sort(key=lambda match: (len(match), -match.distance), reverse=True)

        for match in all_matches:
            if all(not match.intersects(existing_match) for existing_match in best_matches):
                best_matches.append(match)

        left_tokens, right_tokens = filter_remaining_tokens(all_matches, best_matches)

        repeats += 1
        if repeats == 100:
            break

    best_matches.sort(key=lambda match: match.left.start_token.start)

    return best_matches
