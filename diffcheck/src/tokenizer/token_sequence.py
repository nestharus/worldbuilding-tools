from tokenizer.token import Token


class TokenSequence:
    source_tokens: list[Token]
    token_length: int
    bounds: tuple[int, int]

    def __init__(self, source_tokens: list[Token], start_index: int, token_length: int):
        self.source_tokens = source_tokens
        self.token_length = token_length
        self.bounds = (start_index, start_index + token_length - 1)

    def intersects(self, other: 'TokenSequence') -> bool:
        return self.start_token.start <= other.end_token.end and other.start_token.start <= self.end_token.end

    def intersects_token(self, other: Token) -> bool:
        return self.start_token.start <= other.end and other.start <= self.end_token.end

    @property
    def word_count(self) -> int:
        return sum(
            1
            for i in range(self.bounds[0], self.bounds[1] + 1)
            if self.source_tokens[i].is_word
        )

    @property
    def is_legal(self):
        return self.word_count >= 2

    @property
    def text(self):
        return "".join(self.source_tokens[token_index].text for token_index in range(self.bounds[0], self.bounds[1] + 1))

    @property
    def start_token(self) -> Token:
        return self.source_tokens[self.bounds[0]]

    @property
    def end_token(self) -> Token:
        return self.source_tokens[self.bounds[1]]

    @property
    def start_token_index(self) -> int:
        return self.bounds[0]

    @property
    def end_token_index(self) -> int:
        return self.bounds[1]

    @property
    def start_text_index(self) -> int:
        return self.start_token.start

    def __iter__(self):
        for token_index in range(self.bounds[0], self.bounds[1] + 1):
            yield self.source_tokens[token_index]

    def __len__(self):
        return self.token_length

    @property
    def text_length(self) -> int:
        return self.end_token.end - self.start_token.start + 1

    def __repr__(self):
        return f"TokenSequence({self.start_token.start}, {self.end_token.end}, {self.bounds}, {self.token_length})"
