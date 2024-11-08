from typing import List
from tokenizer.token import Token
from tokenizer.tokenizer import tokenize_english_text
from matcher.longest_matches import find_best_matches
from .comparison import TextComparison

def compare_text(original_text: str, revised_text: str) -> TextComparison:
    """
    Compare two texts and return a TextComparison object containing the differences
    """
    # Tokenize both texts
    original_tokens = tokenize_english_text(original_text)
    revised_tokens = tokenize_english_text(revised_text)
    
    # Find matches between the texts
    matches = find_best_matches(original_tokens, revised_tokens)
    
    # Track which tokens are part of matches
    matched_original_indices = set()
    matched_revised_indices = set()
    
    for match in matches:
        for i in range(len(match.source_tokens_left)):
            matched_original_indices.add(match.left_start + i)
        for i in range(len(match.source_tokens_right)):
            matched_revised_indices.add(match.right_start + i)
    
    # Find removed and added tokens
    removed_tokens = [
        token for i, token in enumerate(original_tokens) 
        if i not in matched_original_indices and not token.text.isspace()
    ]
    
    added_tokens = [
        token for i, token in enumerate(revised_tokens)
        if i not in matched_revised_indices and not token.text.isspace()
    ]
    
    # Calculate added words and word count score
    added_words = len(added_tokens)
    word_count_score = len(matches) + added_words
    
    return TextComparison(
        removed_tokens=removed_tokens,
        added_tokens=added_tokens,
        matches=matches,
        added_words=added_words,
        word_count_score=word_count_score
    )
