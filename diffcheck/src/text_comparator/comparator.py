# need to figure out movement
# text moves in a particular direction
# a movement consists of text moving left and text moving right
# all text associated with a movement is classified by moving left or right
# a movement will have N blocks in one direction and excatly 1 block in the other
# the entire movement will count as 1 word

from tokenizer.tokenizer import tokenize_english_text
from matcher.longest_matches import find_best_matches
from matcher.match import Match
from text_comparator.comparison import TextComparison, MatchClassification

def classify_matches(matches: list[Match]) -> list[MatchClassification]:
    """
    Classifies matches as either moved or unchanged based on their positions.
    A match is considered moved if its relative position in the right text
    differs from its position in the left text.
    """
    # First, identify the largest non-overlapping matches
    non_overlapping = []
    sorted_matches = sorted(matches, key=lambda m: (m.token_length, -m.token_distance), reverse=True)
    
    for match in sorted_matches:
        if not any(m.intersects(match) for m in non_overlapping):
            non_overlapping.append(match)
    
    # Sort by left position to establish baseline order
    sorted_matches = sorted(non_overlapping, key=lambda m: m.left_start)
    
    # Track relative positions to detect moves
    classifications = []
    expected_right_pos = 0
    
    for match in sorted_matches:
        # A match is moved if its position relative to previous matches has changed significantly
        position_shift = abs(match.right_start - expected_right_pos)
        is_moved = position_shift > len(match.source_tokens_left) // 2
        
        classifications.append(MatchClassification(match, is_moved))
        expected_right_pos = match.right_start + len(match.source_tokens_left)
        
    return classifications

def compare_text(left_text: str, right_text: str) -> TextComparison:
    # Tokenize both texts
    left_tokens = tokenize_english_text(left_text)
    right_tokens = tokenize_english_text(right_text)
    
    # Find matches between the texts
    matches = find_best_matches(left_tokens, right_tokens)
    
    # Debug print
    print("\nDebug - Matches found:")
    for match in matches:
        left_text = ' '.join(t.text for t in match.source_tokens_left)
        print(f"Match - Left text: {left_text}")
        print(f"Left start: {match.left_start}, Right start: {match.right_start}")

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
    
    # Classify matches as moved or unchanged
    match_classifications = classify_matches(matches)
    
    return TextComparison(
        left_tokens=left_tokens,
        right_tokens=right_tokens,
        matches=matches,
        match_classifications=match_classifications,
        added_words=added_words,
        word_count_score=word_count_score
    )
