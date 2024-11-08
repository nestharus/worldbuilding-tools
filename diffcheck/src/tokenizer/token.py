class Token:
    text: str
    bounds: tuple[int, int]

    def __init__(self, text: str, start: int):
        self.text = text
        self.bounds = (start, start + len(text) - 1)

    @property
    def is_word(self) -> bool:
        return self.text.isalnum()

    @property
    def is_punctuation(self) -> bool:
        return not self.is_word

    @property
    def start(self) -> int:
        return self.bounds[0]

    @property
    def end(self) -> int:
        return self.bounds[1]

    def __repr__(self):
        return f"Token(text='{self.text}', start={self.start}, end={self.end})"

    def __eq__(self, other):
        return self.text == other.text

    def __ne__(self, other):
        return self.text != other.text

    def __hash__(self):
        return hash((self.text, self.bounds))

    def __len__(self):
        return self.bounds[1] - self.bounds[0] + 1

    def __getitem__(self, key):
        return self.text[key]

    def __contains__(self, item):
        return item in self.text

    def __add__(self, other):
        return self.text + other

    def __radd__(self, other):
        return other + self.text

    def __mul__(self, other):
        return self.text * other

    def __rmul__(self, other):
        return other * self.text

    def __iter__(self):
        return iter(self.text)

    def __reversed__(self):
        return reversed(self.text)

    def __bool__(self):
        return bool(self.text)

    def __format__(self, format_spec):
        return self.text.__format__(format_spec)

    def __str__(self):
        return self.text
