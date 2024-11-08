from dataclasses import dataclass
from typing import List
from tokenizer.token import Token
from matcher.match import Match

@dataclass
class TextComparison:
    """Represents the comparison between two texts"""
    removed_tokens: List[Token]
    added_tokens: List[Token]
    matches: List[Match]
    added_words: int
    word_count_score: int
