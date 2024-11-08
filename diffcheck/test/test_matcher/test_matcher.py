import json
import unittest

from matcher.longest_matches import find_best_matches
from matcher.match import Match
from tokenizer.token import Token
from tokenizer.token_sequence import TokenSequence
from tokenizer.tokenizer import tokenize_english_text


class TestLongestContiguousMatches(unittest.TestCase):

    def test_addition(self):
        left = 'This is a test'
        right = 'This is a test with an addition'
        expected = ['This is a test']

        left_tokens = tokenize_english_text(left)
        right_tokens = tokenize_english_text(right)
        matches = find_best_matches(left_tokens, right_tokens)

        self.assertEqual(len(matches), len(expected))
        for match, expected in zip(matches, expected):
            self.assertEqual(expected, match.left.text)
            self.assertEqual(expected, match.right.text)

    def test_subtraction(self):
        left = 'This is a test with subtraction'
        right = 'This is a test'
        expected = ['This is a test']

        left_tokens = tokenize_english_text(left)
        right_tokens = tokenize_english_text(right)
        matches = find_best_matches(left_tokens, right_tokens)

        self.assertEqual(len(matches), len(expected))
        for match, expected in zip(matches, expected):
            self.assertEqual(expected, match.left.text)
            self.assertEqual(expected, match.right.text)

    def test_offset_start_left(self):
        left = ' This is a test'
        right = 'This is a test'
        expected = ['This is a test']

        left_tokens = tokenize_english_text(left)
        right_tokens = tokenize_english_text(right)
        matches = find_best_matches(left_tokens, right_tokens)

        self.assertEqual(len(matches), len(expected))
        for match, expected in zip(matches, expected):
            self.assertEqual(expected, match.left.text)
            self.assertEqual(expected, match.right.text)

    def test_offset_start_right(self):
        left = 'This is a test'
        right = ' This is a test'
        expected = ['This is a test']

        left_tokens = tokenize_english_text(left)
        right_tokens = tokenize_english_text(right)
        matches = find_best_matches(left_tokens, right_tokens)

        self.assertEqual(len(matches), len(expected))
        for match, expected in zip(matches, expected):
            self.assertEqual(expected, match.left.text)
            self.assertEqual(expected, match.right.text)

    def test_multi(self):
        left = 'This is a test with a lot of stuff in it This is a test.'
        right = 'This is a test This is a test with a lot of stuff in it.'
        expected = [
            'This is a test with a lot of stuff in it',
            'This is a test'
        ]

        left_tokens = tokenize_english_text(left)
        right_tokens = tokenize_english_text(right)
        matches = find_best_matches(left_tokens, right_tokens)

        self.assertEqual(len(expected), len(matches))
        for match, expected in zip(matches, expected):
            self.assertEqual(expected, match.left.text)
            self.assertEqual(expected, match.right.text)

    def test_multi2(self):
        left = 'First block. Middle block that should stay unmoved. Third block.'
        right = 'First block. New prepended text. Middle block that should stay unmoved. Third block.'
        expected = [
            'First block',
            '. Middle block that should stay unmoved. Third block.'
        ]

        left_tokens = tokenize_english_text(left)
        right_tokens = tokenize_english_text(right)
        matches = find_best_matches(left_tokens, right_tokens)
        self.assertEqual(len(expected), len(matches))
        for match, expected in zip(matches, expected):
            self.assertEqual(expected, match.left.text)
            self.assertEqual(expected, match.right.text)

    def test_intersect(self):
        tokens = [
            Token("hello", 0),
            Token(" ", 5),
            Token("world", 6),
            Token("!", 11)
        ]
        # Create TokenSequences for left and right sides
        left_seq = TokenSequence(tokens, 1, 2)  # Start at index 1, length 2
        right_seq = TokenSequence(tokens, 1, 2)  # Start at index 1, length 2
        match = Match(left_seq, right_seq, 0)  # distance 0 since same position

        self.assertFalse(match.left_intersects_token(Token("hello", 0)))
        self.assertTrue(match.left_intersects_token(Token(" ", 5)))
        self.assertTrue(match.left_intersects_token(Token("world", 6)))
        self.assertFalse(match.left_intersects_token(Token("!", 11)))

    def test_token_match(self):
        left_text = 'a b c d m e f g h'
        right_text = 'e f m g h c d a b'
        expected = ['a b', ' c d ', 'e f', ' g h']

        left_tokens = tokenize_english_text(left_text)
        right_tokens = tokenize_english_text(right_text)

        matches = find_best_matches(left_tokens, right_tokens)
        match_text = [match.left.text for match in matches]
        self.assertEqual(expected, match_text)

if __name__ == '__main__':
    unittest.main()
