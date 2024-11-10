import difflib
import json

from matcher.match import Match
from movement.identify_movement import identify_moved_blocks, to_moved_matches, to_unmoved_matches
from tokenizer.tokenizer import tokenize_english_text
from matcher.longest_matches import find_best_matches

# with open('original.txt', 'r') as file:
#     left_text = file.read()
#
# with open('revised.txt', 'r') as file:
#     right_text = file.read()

# left_tokens = tokenize_english_text(left_text)
# right_tokens = tokenize_english_text(right_text)
#
# matches = find_best_matches(left_tokens, right_tokens)
# left_blocks, right_blocks = Match.matches_to_blocks(matches)
# moved_blocks = identify_moved_blocks(left_blocks, right_blocks)
# moved_matches = to_moved_matches(moved_blocks, matches)
# unmoved_matches = to_unmoved_matches(moved_blocks, left_blocks, matches)

# these are individual tokens
# ignore sequences
# get matches between left and right with no unrelated tokens in between
# start left and get longest possible match on the right and keep going
# don't need to filter
# '1234567' -> '1237775674'
# '123' (same), '4567' -> '7775674'
# '4' (same), '567', -> '' ('777567' insertion)
# '567' deletion

# '1234567' -> '1237775674'
# '123' same
# '567' same
# '4' on left deleted
# '777' on right inserted
# '4' on right inserted

# don't need to tokenize. can simply look at each individual character and find sequences
# movement can only occur with sequence
# is_sequence: len(sequence.split(whitespace+) needs to be > 1. This will not work because this will match partial words. can't see will match don't saw ('t s)
# wordcount: len(addition_sequences.join().split(whitespace+))

print('あ'.isalnum())
print(' \t\n\r'.isspace())


def is_sequence(source: str, span: tuple[int, int]):
    if span[1] - span[0] <= 2:
        return False

    # can't start with punctuation
    if span[0] > 0 and not source[span[0] - 1].isspace():
        return False

    # can end with punctuation!!!
    # sometimes next can be punctuation Jess(')
    # sometimes next cannot be can('t)
    # the next character can be punctuation Jess(.)
    # but it might not be able to be punctuation U.S.A(.)
    if span[1] <= len(source) and not source[span[1] + 1].isspace():
        return False

def analyze_token_changes(left, right):
    result = []
    s = difflib.SequenceMatcher(None, left, right)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            result.append(f"=={left[i1:i2]}")
        elif tag == 'delete':
            result.append(f"-{left[i1:i2]}")
        elif tag == 'insert':
            result.append(f"+{right[j1:j2]}")
        elif tag == 'replace':
            result.append(f"-{left[i1:i2]}")
            result.append(f"+{right[j1:j2]}")
    return result

analysis = analyze_token_changes('1234567', '1237775674')
for line in analysis:
    print(line)

'''
removals = [
    token for token in left_tokens
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
'''