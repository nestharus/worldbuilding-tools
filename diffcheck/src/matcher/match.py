from itertools import islice
from typing import Generator

from tokenizer.token import Token
from tokenizer.token_sequence import TokenSequence


def find_token_matches(text, tokens) -> list[int]:
    return [
        index for index, token in enumerate(tokens) if token.text == text
    ]


class Match:
    left: TokenSequence
    right: TokenSequence
    distance: int

    def __init__(
        self,
        left: TokenSequence,
        right: TokenSequence,
        distance: int
    ):
        self.left = left
        self.right = right
        self.distance = distance

    def __repr__(self) -> str:
        return f"Match({self.left}, {self.right}, {self.distance})"

    def __lt__(self, other: 'Match') -> bool:
        if len(self.left) != len(other.left):
            return len(self.left) < len(other.left)
        return self.distance < other.distance

    def __eq__(self, other: 'Match') -> bool:
        return len(self.left) == len(other.left) and self.distance == other.distance

    def __ne__(self, other: 'Match') -> bool:
        return len(self.left) != len(other.left) or self.distance != other.distance

    def __gt__(self, other: 'Match') -> bool:
        if len(self.left) != len(other.left):
            return len(self.left) > len(other.left)
        return self.distance > other.distance

    def intersects(self, other: 'Match') -> bool:
        return self.left.intersects(other.left) or self.right.intersects(other.right)

    def left_intersects_token(self, other: Token) -> bool:
        return self.left.intersects_token(other)

    def right_intersects_token(self, other: Token) -> bool:
        return self.right.intersects_token(other)

    def __iter__(self):
        return iter(self.left)

    def __len__(self):
        return len(self.left)

    @staticmethod
    def matches_to_blocks(matches: list['Match']) -> tuple[list[object], list[object]]:
        left_blocks = [
            object()
            for _ in matches
        ]
        right_blocks = [
            (match.right.start_token_index, left_blocks[index])
            for index, match in enumerate(matches)
        ]
        right_blocks.sort(key=lambda x: x[0])
        right_blocks = [
            right_block[1]
            for right_block in right_blocks
        ]
        return left_blocks, right_blocks

    @staticmethod
    def blocks_to_indices(blocks: list[tuple[int, int]]) -> Generator[int, int, None]:
        return (
            index
            for block in blocks
            for index in range(block[0], block[1] + 1)
        )

    @staticmethod
    def compute_match_length(
        left_tokens: list[Token],
        right_tokens: list[Token],
        left_start: int,
        right_start: int
    ) -> int:
        match_length = 0

        for left, right in zip(islice(left_tokens, left_start, None), islice(right_tokens, right_start, None)):
            if left.text != right.text:
                break

            match_length += 1

        return match_length
