from typing import List
from tokenizer.token import Token
from tokenizer.tokenizer import tokenize_english_text
from matcher.longest_matches import find_best_matches
from .comparison import TextComparison

def compare_text(left_text: str, right_text: str) -> TextComparison:
    # Tokenize both texts
    left_tokens = tokenize_english_text(left_text)
    right_tokens = tokenize_english_text(right_text)
    
    # Find matches between the texts
    matches = find_best_matches(left_tokens, right_tokens)

    removed_tokens = [
        token for token in left_tokens
        if not any(
            match.token_intersects_left(token)
            for match in matches
        )
    ]

    added_tokens = [
        token for token in right_tokens
        if not any(
            match.token_intersects_right(token)
            for match in matches
        )
    ]

    added_words = sum(
        1 for token in added_tokens
        if token.is_word
    )
    word_count_score = len(matches) + added_words
    
    return TextComparison(
        removed_tokens=removed_tokens,
        added_tokens=added_tokens,
        matches=matches,
        added_words=added_words,
        word_count_score=word_count_score
    )
