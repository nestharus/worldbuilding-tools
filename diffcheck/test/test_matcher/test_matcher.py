import pytest

from text_comparator.get_text_diff import get_text_deltas
from tokenizer.context_aware_tokenizer import ContextAwareTokenizer
from tokenizer.deberta_tokenizer import DebertaTokenizer
from tokenizer.spacy_tokenizer import spacy_tokenizer


@pytest.fixture
def tokenizer():
    deberta = DebertaTokenizer()
    spacy = spacy_tokenizer()
    tokenizer = ContextAwareTokenizer(deberta, spacy)

    return tokenizer


@pytest.fixture
def text_tokens(tokenizer):
    return lambda tokens: [
        tokenizer.deberta_tokenizer.tokenizer.convert_ids_to_tokens(token[2])
        for token in tokens
    ]


@pytest.fixture
def tokenize(tokenizer):
    return lambda text: tokenizer.tokenize(text)


@pytest.mark.unit
def test_addition(tokenize, text_tokens):
    left = 'This is a test'
    right = 'This is a test with an addition'
    expected_additions = ['with', 'addition']
    expected_subtractions = []
    expected_movements = []

    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    additions, subtractions, movements = get_text_deltas(left_tokens, right_tokens)

    assert text_tokens(additions) == expected_additions
    assert text_tokens(subtractions) == expected_subtractions
    assert text_tokens([movement[0] for movement in movements]) == expected_movements


@pytest.mark.unit
def test_subtraction(tokenize, text_tokens):
    left = 'This is a test with subtraction'
    right = 'This is a test'
    expected_additions = []
    expected_subtractions = ['with', 'subtraction']
    expected_movements = []

    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    additions, subtractions, movements = get_text_deltas(left_tokens, right_tokens)

    assert text_tokens(additions) == expected_additions
    assert text_tokens(subtractions) == expected_subtractions
    assert text_tokens([movement[0] for movement in movements]) == expected_movements


@pytest.mark.unit
def test_multi(tokenize, text_tokens):
    left = 'This is a test with a lot of stuff in it This is a test.'
    right = 'This is a test This is a test with a lot of stuff in it.'
    expected_additions = []
    expected_subtractions = []
    expected_movements = [['this', 'is', 'test']]

    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    additions, subtractions, movements = get_text_deltas(left_tokens, right_tokens)

    assert text_tokens(additions) == expected_additions
    assert text_tokens(subtractions) == expected_subtractions
    assert text_tokens([movement[0] for movement in movements]) == expected_movements


@pytest.mark.unit
def test_multi2(tokenize, text_tokens):
    left = 'First block. Middle block that should stay unmoved. Third block.'
    right = 'First block. New prepended text. Middle block that should stay unmoved. Third block.'
    expected_additions = ['new', 'prepended', 'text']
    expected_subtractions = []
    expected_movements = []

    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    additions, subtractions, movements = get_text_deltas(left_tokens, right_tokens)

    assert text_tokens(additions) == expected_additions
    assert text_tokens(subtractions) == expected_subtractions
    assert text_tokens([movement[0] for movement in movements]) == expected_movements


@pytest.mark.unit
def test_token_match(tokenize, text_tokens):
    left = '1 2 3 4 9 5 6 7 8'
    right = '5 6 9 7 8 3 4 1 2'
    expected_additions = ['9']
    expected_subtractions = ['9']
    expected_movements = [['3', '4'], ['5', '6'], ['7', '8']]

    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    additions, subtractions, movements = get_text_deltas(left_tokens, right_tokens)

    assert text_tokens(additions) == expected_additions
    assert text_tokens(subtractions) == expected_subtractions
    assert text_tokens([movement[0] for movement in movements]) == expected_movements
