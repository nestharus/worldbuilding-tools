from tokenizer.token import Token


def find_token_matches(text, tokens) -> list[int]:
    return [
        index for index, token in enumerate(tokens) if token.text == text
    ]


class Match:
    source_tokens_left: list[Token]
    source_tokens_right: list[Token]
    token_distance: int
    token_length: int
    left_region: range
    right_region: range

    def __init__(
        self,
        left_start_index: int,
        right_start_index: int,
        source_tokens_left: list[Token],
        source_tokens_right: list[Token],
        token_distance: int,
        token_length: int
    ):
        self.source_tokens_left = source_tokens_left
        self.source_tokens_right = source_tokens_right
        self.left_region = range(left_start_index, left_start_index + token_length)
        self.right_region = range(right_start_index, right_start_index + token_length)
        self.token_distance = token_distance
        self.token_length = token_length

    def __repr__(self) -> str:
        return f"Match({self.left_start}, {self.right_start}, {self.token_distance}, {self.token_length})"

    def __lt__(self, other) -> bool:
        if self.token_length != other.token_length:
            return self.token_length < other.token_length
        return self.token_distance < other.token_distance

    def __eq__(self, other) -> bool:
        return self.token_length == other.token_length and self.token_distance == other.token_distance

    def __ne__(self, other) -> bool:
        return self.token_length != other.token_length or self.token_distance != other.token_distance

    def __gt__(self, other) -> bool:
        if self.token_length != other.token_length:
            return self.token_length > other.token_length
        return self.token_distance > other.token_distance

    def intersects_left(self, other) -> bool:
        return self.left_region.start < other.left_region.stop and other.left_region.start < self.left_region.stop

    def intersects_right(self, other) -> bool:
        return self.right_region.start < other.right_region.stop and other.right_region.start < self.right_region.stop

    def intersects(self, other) -> bool:
        return self.intersects_left(other) or self.intersects_right(other)

    def token_intersects_left(self, token: Token) -> bool:
        return token.end >= self.left_start_token.start and token.start <= self.left_end_token.end

    def token_intersects_right(self, token: Token) -> bool:
        return token.end >= self.right_start_token.start and token.start <= self.right_end_token.end

    @property
    def left_start(self) -> int:
        return self.left_region.start

    @property
    def right_start(self) -> int:
        return self.right_region.start

    @property
    def left_start_token(self) -> Token:
        return self.source_tokens_left[self.left_start]

    @property
    def right_start_token(self) -> Token:
        return self.source_tokens_right[self.right_start]

    @property
    def left_end_token(self) -> Token:
        return self.source_tokens_left[self.left_region.stop - 1]

    @property
    def right_end_token(self) -> Token:
        return self.source_tokens_right[self.right_region.stop - 1]

    @property
    def text_length(self) -> int:
        return self.left_end_token.end - self.left_start_token.start + 1

    @property
    def word_count(self) -> int:
        return sum(
            1
            for i in self.left_region
            if self.source_tokens_left[i].is_word
        )

    @property
    def is_legal(self):
        return self.word_count >= 2

    @property
    def left_text(self):
        return "".join(self.source_tokens_left[token_index].text for token_index in self.left_region)

    @property
    def right_text(self):
        return "".join(self.source_tokens_right[token_index].text for token_index in self.right_region)

    def __iter__(self):
        for token_index in self.left_region:
            yield self.source_tokens_left[token_index]

    def __len__(self):
        return self.token_length

    @staticmethod
    def compute_match_length(
        left_tokens: list[Token],
        right_tokens: list[Token],
        left_start: int,
        right_start: int
    ) -> int:
        left_len = len(left_tokens)
        right_len = len(right_tokens)
        match_length = 0

        while (left_start + match_length < left_len and
               right_start + match_length < right_len and
               left_tokens[left_start + match_length].text == right_tokens[right_start + match_length].text):
            match_length += 1

        return match_length
