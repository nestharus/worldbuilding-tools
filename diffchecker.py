import nltk
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt', quiet=True)


def preprocess_text(text):
    text = ' '.join(text.split())
    tokens = word_tokenize(text)
    return [word.lower() for word in tokens]


def find_match(tokens1, tokens2, start1, start2, min_length=3):
    """Find length of matching sequence starting at given positions"""
    length = 0
    while (start1 + length < len(tokens1) and
           start2 + length < len(tokens2) and
           tokens1[start1 + length] == tokens2[start2 + length]):
        length += 1
    return length if length >= min_length else 0


def find_largest_match(tokens1, tokens2, used1, used2, min_length=3):
    """Find the largest matching sequence that hasn't been used"""
    best_match = None
    best_length = min_length - 1

    for i in range(len(tokens1)):
        if i in used1:
            continue

        for j in range(len(tokens2)):
            if j in used2:
                continue

            if tokens1[i] == tokens2[j]:
                length = find_match(tokens1, tokens2, i, j, min_length)
                if length > best_length:
                    best_length = length
                    best_match = (i, j, length)

    return best_match


def find_all_matches(tokens1, tokens2):
    """Find all matching blocks between texts"""
    matches = []
    used1 = set()
    used2 = set()

    while True:
        match = find_largest_match(tokens1, tokens2, used1, used2)
        if not match:
            break

        start1, start2, length = match
        matches.append({
            'start1': start1,
            'start2': start2,
            'length': length,
            'text': tokens2[start2:start2 + length]
        })

        used1.update(range(start1, start1 + length))
        used2.update(range(start2, start2 + length))

    return matches


def combine_related_movements(moved_blocks, tokens2):
    """Combine blocks that are part of the same movement"""
    if len(moved_blocks) <= 1:
        return moved_blocks

    combined = []
    used = set()

    for i, block1 in enumerate(moved_blocks):
        if i in used:
            continue

        related_blocks = [block1]
        start_pos1 = block1['start1']
        start_pos2 = block1['start2']

        # Look for blocks that moved in a complementary way
        for j, block2 in enumerate(moved_blocks):
            if j == i or j in used:
                continue

            # Check if this is a related movement
            # Two blocks are related if they swapped positions
            if (abs(block1['start1'] - block2['start2']) < 5 or
                    abs(block1['start2'] - block2['start1']) < 5):
                related_blocks.append(block2)
                used.add(j)

        if len(related_blocks) > 1:
            # These blocks are part of the same movement
            combined.append(block1)
            used.add(i)

    return combined if combined else moved_blocks


def analyze_texts(original_text, revised_text):
    # Preprocess texts
    tokens1 = preprocess_text(original_text)
    tokens2 = preprocess_text(revised_text)

    # Find all matching blocks
    matches = find_all_matches(tokens1, tokens2)

    # Track matched positions in both texts
    matched1 = set()
    matched2 = set()
    for match in matches:
        matched1.update(range(match['start1'], match['start1'] + match['length']))
        matched2.update(range(match['start2'], match['start2'] + match['length']))

    # Find moved blocks (matching sequences in different positions)
    moved_blocks = []
    for match in matches:
        if match['start1'] != match['start2']:
            # Make sure this isn't a subset of a larger moved block
            is_subset = False
            for other in matches:
                if other != match and \
                        other['start1'] <= match['start1'] and \
                        other['start1'] + other['length'] >= match['start1'] + match['length'] and \
                        other['start2'] <= match['start2'] and \
                        other['start2'] + other['length'] >= match['start2'] + match['length']:
                    is_subset = True
                    break
            if not is_subset:
                moved_blocks.append(match)

    # Combine related movements
    moved_blocks = combine_related_movements(moved_blocks, tokens2)

    # Find added words (tokens in text2 that aren't matched)
    added_words = []
    for i, token in enumerate(tokens2):
        if i not in matched2 and token not in string.punctuation:
            added_words.append(token)

    # Print results
    print(f"Total added words from modifications and new content: {len(added_words)}")
    print(f"Total movements (largest matching text fragments): {len(moved_blocks)}")

    print("Added words:", added_words)
    if moved_blocks:
        print("Moved blocks:")
        for block in moved_blocks:
            print(' '.join(block['text']))

# Example original and revised texts
with open("diffcheck/src/original.txt", "r") as file:
    original_text = file.read()

with open("diffcheck/src/revised.txt", "r") as file:
    revised_text = file.read()

# Split texts into word lists for word-by-word comparison
original_sentences = original_text.split(". ")
new_sentences = revised_text.split(". ")

analyze_texts(original_text, revised_text)