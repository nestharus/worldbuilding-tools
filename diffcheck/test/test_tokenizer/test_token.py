import unittest

from tokenizer.token import Token


class TestToken(unittest.TestCase):

    def test_token_initialization(self):
        token = Token("hello", 0)
        self.assertEqual(token.text, "hello")
        self.assertEqual(token.start, 0)
        self.assertEqual(token.end, 4)
        self.assertEqual(token.region, range(0, 5))

    def test_is_word(self):
        token = Token("hello", 0)
        self.assertTrue(token.is_word)
        token = Token("!", 0)
        self.assertFalse(token.is_word)

    def test_is_punctuation(self):
        token = Token("hello", 0)
        self.assertFalse(token.is_punctuation)
        token = Token("!", 0)
        self.assertTrue(token.is_punctuation)

    def test_token_equality(self):
        token1 = Token("hello", 0)
        token2 = Token("hello", 5)
        token3 = Token("world", 0)
        self.assertEqual(token1, token2)
        self.assertNotEqual(token1, token3)

    def test_token_length(self):
        token = Token("hello", 0)
        self.assertEqual(len(token), 5)

    def test_token_repr(self):
        token = Token("hello", 0)
        self.assertEqual(repr(token), "Token(text='hello', start=0, end=4)")

    def test_token_addition(self):
        token = Token("hello", 0)
        self.assertEqual(token + " world", "hello world")
        self.assertEqual("Say " + token, "Say hello")

    def test_token_multiplication(self):
        token = Token("hello", 0)
        self.assertEqual(token * 3, "hellohellohello")
        self.assertEqual(3 * token, "hellohellohello")

    def test_token_iteration(self):
        token = Token("hello", 0)
        self.assertEqual(list(iter(token)), list("hello"))

    def test_token_reversed(self):
        token = Token("hello", 0)
        self.assertEqual(list(reversed(token)), list("olleh"))

    def test_token_bool(self):
        token = Token("hello", 0)
        self.assertTrue(token)
        token = Token("", 0)
        self.assertFalse(token)

    def test_token_format(self):
        token = Token("hello", 0)
        self.assertEqual(format(token, ""), "hello")

if __name__ == '__main__':
    unittest.main()
