from matcher.match import Match, find_token_matches
from tokenizer.token import Token


def find_longest_matches_from_token(
    left_index: int,
    left_tokens: list[Token],
    right_tokens: list[Token]
) -> list[Match]:
    left_token = left_tokens[left_index]

    matches = [
        Match(
            left_start_index = left_index,
            right_start_index = right_index,
            source_tokens_left = left_tokens,
            source_tokens_right = right_tokens,
            token_distance = abs(right_index - left_index),
            token_length = Match.compute_match_length(left_tokens, right_tokens, left_index, right_index)
        )
        for right_index in find_token_matches(left_token.text, right_tokens)
    ]
    matches = [match for match in matches if match.is_legal]

    return matches


def find_best_matches(
    left_tokens: list[Token],
    right_tokens: list[Token]
) -> list[Match]:
    all_matches = [
        match
        for left_index in range(len(left_tokens))
        for match in find_longest_matches_from_token(left_index, left_tokens, right_tokens)
    ]
    all_matches.sort(key=lambda match: (match.token_length, match.token_distance), reverse=True)

    best_matches = []

    for match in all_matches:
        if all(not match.intersects(existing_match) for existing_match in best_matches):
            best_matches.append(match)

    return best_matches
