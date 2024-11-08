import unittest

from tokenizer.token import Token
from tokenizer.token_sequence import TokenSequence

class TestTokenSequence(unittest.TestCase):
    def test_intersects_1_3_with_2_4(self):
        source = [
            Token(str(i), i)
            for i in range(10)
        ]
        ts1 = TokenSequence(source, 0, 2)  # 1,3
        ts2 = TokenSequence(source, 1, 3)  # 2,4
        self.assertTrue(ts1.intersects(ts2))

    def test_intersects_1_3_with_3_5(self):
        source = [
            Token('a', i)
            for i in range(10)
        ]
        ts1 = TokenSequence(source, 0, 3)  # 1,3
        ts3 = TokenSequence(source, 2, 3)  # 3,5
        self.assertTrue(ts1.intersects(ts3))

    def test_not_intersects_1_3_with_4_6(self):
        source = [
            Token('a', i)
            for i in range(10)
        ]
        ts1 = TokenSequence(source, 0, 3)  # 1,3
        ts4 = TokenSequence(source, 3, 3)  # 4,6
        self.assertFalse(ts1.intersects(ts4))

    def test_intersects_2_4_with_1_3(self):
        source = [
            Token('a', i)
            for i in range(10)
        ]
        ts2 = TokenSequence(source, 1, 3)  # 2,4
        ts1 = TokenSequence(source, 0, 3)  # 1,3
        self.assertTrue(ts2.intersects(ts1))

    def test_intersects_3_5_with_1_3(self):
        source = [
            Token('a', i)
            for i in range(10)
        ]
        ts3 = TokenSequence(source, 2, 3)  # 3,5
        ts1 = TokenSequence(source, 0, 3)  # 1,3
        self.assertTrue(ts3.intersects(ts1))

    def test_not_intersects_4_6_with_1_3(self):
        source = [
            Token('a', i)
            for i in range(10)
        ]
        ts4 = TokenSequence(source, 3, 3)  # 4,6
        ts1 = TokenSequence(source, 0, 3)  # 1,3
        self.assertFalse(ts4.intersects(ts1))

if __name__ == '__main__':
    unittest.main()