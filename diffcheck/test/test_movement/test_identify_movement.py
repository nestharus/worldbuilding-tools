import json
import unittest

from matcher.longest_matches import find_best_matches
from matcher.match import Match
from movement.identify_movement import identify_moved_blocks, to_moved_matches, to_unmoved_matches
from tokenizer.tokenizer import tokenize_english_text


class TestIdentifyMovedBlocks(unittest.TestCase):
    def test_no_moved_blocks(self):
        left = [1, 2, 3, 4, 5, 6]
        right = [1, 2, 3, 4, 5, 6]
        expected = []

        movements = identify_moved_blocks(left, right)
        moved_blocks = [
            (
                left[value]
                for value in range(block[0], block[1] + 1)
            )
            for block in movements
        ]

        self.assertEqual(expected, moved_blocks)

    def test_single_moved_block(self):
        left = [1, 2, 3, 4, 5, 6]
        right = [1, 2, 6, 4, 5, 3]
        expected = [[3], [6]]

        movements = identify_moved_blocks(left, right)
        moved_blocks = [
            [
                left[value]
                for value in range(block[0], block[1] + 1)
            ]
            for block in movements
        ]
        for block in moved_blocks:
            if len(block) == 2 and block[0] == block[1]:
                block.pop()

        self.assertEqual(expected, moved_blocks)

    def test_multiple_moved_blocks(self):
        left = [1, 2, 3, 4, 5, 6, 7, 8]
        right = [3, 7, 6, 4, 5, 8, 1, 2]
        expected = [[1, 2], [6], [7]]

        movements = identify_moved_blocks(left, right)
        moved_blocks = [
            [
                left[value]
                for value in range(block[0], block[1] + 1)
            ]
            for block in movements
        ]
        for block in moved_blocks:
            if len(block) == 2 and block[0] == block[1]:
                block.pop()

        self.assertEqual(expected, moved_blocks)

    def test_another_single_moved_block(self):
        left = [1, 2, 3, 4, 5, 6]
        right = [6, 3, 5, 1, 4, 2]
        expected = [[1], [2], [3], [4], [5]]

        movements = identify_moved_blocks(left, right)
        moved_blocks = [
            [
                left[value]
                for value in range(block[0], block[1] + 1)
            ]
            for block in movements
        ]
        for block in moved_blocks:
            if len(block) == 2 and block[0] == block[1]:
                block.pop()

        self.assertEqual(expected, moved_blocks)

    def test_continuous_move(self):
        left = [1, 2, 3, 4, 5, 6, 7, 8]
        right = [5, 6, 7, 8, 3, 4, 1, 2]
        expected = [[1, 2], [3, 4]]

        movements = identify_moved_blocks(left, right)
        moved_blocks = [
            [
                left[value]
                for value in range(block[0], block[1] + 1)
            ]
            for block in movements
        ]
        for block in moved_blocks:
            if len(block) == 2 and block[0] == block[1]:
                block.pop()

        self.assertEqual(expected, moved_blocks)

    def test_token_match(self):
        left_text = 'a b c d m e f g h'
        right_text = 'e f m g h c d a b'
        expected_moved_blocks = ['a b', ' c d ']
        expected_unmoved_blocks = ['e f', ' g h']

        left_tokens = tokenize_english_text(left_text)
        right_tokens = tokenize_english_text(right_text)
        matches = find_best_matches(left_tokens, right_tokens)
        left_blocks, right_blocks = Match.matches_to_blocks(matches)

        moved_blocks = identify_moved_blocks(left_blocks, right_blocks)
        moved_matches = to_moved_matches(moved_blocks, matches)
        unmoved_matches = to_unmoved_matches(moved_blocks, left_blocks, matches)
        moved_text_blocks = [
            match.left.text
            for match in moved_matches
        ]
        unmoved_text_blocks = [
            match.left.text
            for match in unmoved_matches
        ]

        self.assertEqual(expected_moved_blocks, moved_text_blocks)
        self.assertEqual(expected_unmoved_blocks, unmoved_text_blocks)

if __name__ == '__main__':
    unittest.main()