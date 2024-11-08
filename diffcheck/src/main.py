import json

from tokenizer.tokenizer import tokenize_english_text
from matcher.longest_matches import find_best_matches

with open('original.txt', 'r') as file:
    original = file.read()

with open('revised.txt', 'r') as file:
    revised = file.read()

original_tokens = tokenize_english_text(original)
revised_tokens = tokenize_english_text(revised)

matches = find_best_matches(original_tokens, revised_tokens)

removals = [
    token for token in original_tokens
    if not any(
        match.token_intersects_left(token)
        for match in matches
    )
]

additions = [
    token for token in revised_tokens
    if not any(
        match.token_intersects_right(token)
        for match in matches
    )
]

additional_words = sum(
    1 for token in additions
    if token.is_word
)

word_count = additional_words + len(matches)

print(len(matches))
print(additional_words)