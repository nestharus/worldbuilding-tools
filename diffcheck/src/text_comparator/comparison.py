from dataclasses import dataclass
from tokenizer.token import Token
from matcher.match import Match
from typing import NamedTuple

class MatchClassification(NamedTuple):
    match: Match
    is_moved: bool

@dataclass
class TextComparison:
    """Represents the comparison between two texts"""
    left_tokens: list[Token]
    right_tokens: list[Token]
    matches: list[Match]
    match_classifications: list[MatchClassification]
    added_words: int
    word_count_score: int
