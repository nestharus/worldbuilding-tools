import json
import unittest

from matcher.longest_matches import find_best_matches
from matcher.match import Match
from tokenizer.token import Token
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
            self.assertEqual(expected, match.left_text)
            self.assertEqual(expected, match.right_text)

    def test_subtraction(self):
        left = 'This is a test with subtraction'
        right = 'This is a test'
        expected = ['This is a test']

        left_tokens = tokenize_english_text(left)
        right_tokens = tokenize_english_text(right)
        matches = find_best_matches(left_tokens, right_tokens)

        self.assertEqual(len(matches), len(expected))
        for match, expected in zip(matches, expected):
            self.assertEqual(expected, match.left_text)
            self.assertEqual(expected, match.right_text)

    def test_offset_start_left(self):
        left = ' This is a test'
        right = 'This is a test'
        expected = ['This is a test']

        left_tokens = tokenize_english_text(left)
        right_tokens = tokenize_english_text(right)
        matches = find_best_matches(left_tokens, right_tokens)

        self.assertEqual(len(matches), len(expected))
        for match, expected in zip(matches, expected):
            self.assertEqual(expected, match.left_text)
            self.assertEqual(expected, match.right_text)

    def test_offset_start_right(self):
        left = 'This is a test'
        right = ' This is a test'
        expected = ['This is a test']

        left_tokens = tokenize_english_text(left)
        right_tokens = tokenize_english_text(right)
        matches = find_best_matches(left_tokens, right_tokens)

        self.assertEqual(len(matches), len(expected))
        for match, expected in zip(matches, expected):
            self.assertEqual(expected, match.left_text)
            self.assertEqual(expected, match.right_text)

    def test_multi(self):
        left = 'This is a test with a lot of stuff in it. This is a test.'
        right = 'This is a test. This is a test with a lot of stuff in it.'
        expected = [
            'This is a test with a lot of stuff in it.',
            'This is a test.'
        ]

        left_tokens = tokenize_english_text(left)
        right_tokens = tokenize_english_text(right)
        matches = find_best_matches(left_tokens, right_tokens)
        print(json.dumps([{'left': match.left_text, 'right': match.right_text} for match in matches], indent=2))

        self.assertEqual(len(matches), len(expected))
        for match, expected in zip(matches, expected):
            self.assertEqual(expected, match.left_text)
            self.assertEqual(expected, match.right_text)

    def test_intersect(self):
        tokens = [
            Token("hello", 0),
            Token(" ", 5),
            Token("world", 6),
            Token("!", 11)
        ]
        match = Match(1, 1, tokens, tokens, 1, 2)

        self.assertFalse(match.token_intersects_left(Token("hello", 0)))
        self.assertTrue(match.token_intersects_left(Token(" ", 5)))
        self.assertTrue(match.token_intersects_left(Token("world", 6)))
        self.assertFalse(match.token_intersects_left(Token("!", 11)))

if __name__ == '__main__':
    unittest.main()