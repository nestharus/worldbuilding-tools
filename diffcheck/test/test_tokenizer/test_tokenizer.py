import unittest
from tokenizer.tokenizer import tokenize_english_text
from tokenizer.token import Token

class TestTokenizer(unittest.TestCase):

    def test_tokenize_english_text(self):
        txt = 'His cloak and glove remain camouflaged on Däor'
        expected = [
            Token("His", 0),
            Token(" ", 3),
            Token("cloak", 4),
            Token(" ", 9),
            Token("and", 10),
            Token(" ", 13),
            Token("glove", 14),
            Token(" ", 19),
            Token("remain", 20),
            Token(" ", 26),
            Token("camouflaged", 27),
            Token(" ", 38),
            Token("on", 39),
            Token(" ", 41),
            Token("Däor", 42)
        ]

        result = tokenize_english_text(txt)
        print(result)
        self.assertEqual(len(expected), len(result))
        for res_token, exp_token in zip(result, expected):
            self.assertEqual(exp_token.text, res_token.text)
            self.assertEqual(exp_token.start, res_token.start)

if __name__ == '__main__':
    unittest.main()
