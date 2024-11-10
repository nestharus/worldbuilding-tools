import re
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
        expected_tokens = [t.text for t in expected]
        result_tokens = [t.text for t in result]
        self.assertEqual(expected_tokens, result_tokens)

    def test_tokenize_contractions(self):
        test_cases = [
            ("can't", [Token("can't", 0)]),
            ("won't", [Token("won't", 0)]),
            ("I'm", [Token("I'm", 0)]),
            ("they're", [Token("they're", 0)]),
            ("we've", [Token("we've", 0)]),
            ("they'll", [Token("they'll", 0)]),
            ("he'd", [Token("he'd", 0)])
        ]

        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = tokenize_english_text(input_text)
                expected_tokens = [t.text for t in expected]
                result_tokens = [t.text for t in result]
                self.assertEqual(expected_tokens, result_tokens)

    def test_tokenize_possessives(self):
        test_cases = [
            ("jess'", [Token("jess'", 0)]),
            ("James's", [Token("James's", 0)]),
            ("the cats'", [Token("the", 0), Token(" ", 3), Token("cats'", 4)])
        ]

        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = tokenize_english_text(input_text)
                expected_tokens = [t.text for t in expected]
                result_tokens = [t.text for t in result]
                self.assertEqual(expected_tokens, result_tokens)

    def test_with_period(self):
        txt = 'Hi.'
        expected = [
            Token("Hi", 0),
            Token(".", 2)
        ]

        result = tokenize_english_text(txt)
        expected_tokens = [t.text for t in expected]
        result_tokens = [t.text for t in result]
        self.assertEqual(expected_tokens, result_tokens)

    def test_with_periods(self):
        txt = 'U.S.A..'
        expected = [
            Token("U.S.A.", 0),
            Token(".", 6)
        ]

        result = tokenize_english_text(txt)
        expected_tokens = [t.text for t in expected]
        result_tokens = [t.text for t in result]
        self.assertEqual(expected_tokens, result_tokens)

    def test_with_periods2(self):
        txt = 'U.S.A ..'
        expected = [
            Token("U.S.A", 0),
            Token(" ", 5),
            Token(".", 6),
            Token(".", 7)
        ]

        result = tokenize_english_text(txt)
        expected_tokens = [t.text for t in expected]
        result_tokens = [t.text for t in result]
        self.assertEqual(expected_tokens, result_tokens)

    def test_with_periods3(self):
        txt = 'U.S.A. .'
        expected = [
            Token("U.S.A.", 0),
            Token(" ", 6),
            Token(".", 7)
        ]

        result = tokenize_english_text(txt)
        expected_tokens = [t.text for t in expected]
        result_tokens = [t.text for t in result]
        self.assertEqual(expected_tokens, result_tokens)

    def test_with_more_periods(self):
        txt = 'Ah..'
        expected = [
            Token("Ah", 0),
            Token(".", 2),
            Token(".", 3)
        ]

        result = tokenize_english_text(txt)
        expected_tokens = [t.text for t in expected]
        result_tokens = [t.text for t in result]
        self.assertEqual(expected_tokens, result_tokens)

    def test_with_names(self):
        txt = 'Ron, B. is'
        expected = [
            Token("Ron", 0),
            Token(",", 3),
            Token(" ", 4),
            Token("B", 5),
            Token(".", 6),
            Token(" ", 7),
            Token("is", 8)
        ]

        result = tokenize_english_text(txt)
        expected_tokens = [t.text for t in expected]
        result_tokens = [t.text for t in result]
        self.assertEqual(expected_tokens, result_tokens)

    def test_with_i_name(self):
        txt = 'Ron, I.'
        expected = [
            Token("Ron", 0),
            Token(",", 3),
            Token(" ", 4),
            Token("I", 5),
            Token(".", 6)
        ]

        result = tokenize_english_text(txt)
        expected_tokens = [t.text for t in expected]
        result_tokens = [t.text for t in result]
        self.assertEqual(expected_tokens, result_tokens)

    def test_with_i_period(self):
        txt = 'Oh. I.'
        expected = [
            Token("Oh", 0),
            Token(".", 2),
            Token(" ", 3),
            Token("I", 4),
            Token(".", 5)
        ]

        result = tokenize_english_text(txt)
        expected_tokens = [t.text for t in expected]
        result_tokens = [t.text for t in result]
        self.assertEqual(expected_tokens, result_tokens)

    def test_with_names2(self):
        txt = 'Ron, Bar. is'
        expected = [
            Token("Ron", 0),
            Token(",", 3),
            Token(" ", 4),
            Token("Bar", 5),
            Token(".", 8),
            Token(" ", 9),
            Token("is", 10)
        ]

        result = tokenize_english_text(txt)
        expected_tokens = [t.text for t in expected]
        result_tokens = [t.text for t in result]
        self.assertEqual(expected_tokens, result_tokens)

    def test_with_exclamation(self):
        txt = 'Ah?!'
        expected = [
            Token("Ah", 0),
            Token("?", 2),
            Token("!", 3)
        ]

        result = tokenize_english_text(txt)
        expected_tokens = [t.text for t in expected]
        result_tokens = [t.text for t in result]
        self.assertEqual(expected_tokens, result_tokens)

    def test_with_dash(self):
        txt = 'do-good'
        expected = [
            Token("do-good", 0)
        ]

        result = tokenize_english_text(txt)
        expected_tokens = [t.text for t in expected]
        result_tokens = [t.text for t in result]
        self.assertEqual(expected_tokens, result_tokens)

    def test_with_dash3(self):
        txt = 'do-good-with'
        expected = [
            Token("do-good-with", 0)
        ]

        result = tokenize_english_text(txt)
        expected_tokens = [t.text for t in expected]
        result_tokens = [t.text for t in result]
        self.assertEqual(expected_tokens, result_tokens)

    def test_with_underscore(self):
        txt = 'many_thanks'
        expected = [
            Token("many_thanks", 0)
        ]

        result = tokenize_english_text(txt)
        expected_tokens = [t.text for t in expected]
        result_tokens = [t.text for t in result]
        self.assertEqual(expected_tokens, result_tokens)

    def test_with_numbers(self):
        txt = 'hello69'
        expected = [
            Token("hello69", 0)
        ]

        result = tokenize_english_text(txt)
        expected_tokens = [t.text for t in expected]
        result_tokens = [t.text for t in result]
        self.assertEqual(expected_tokens, result_tokens)

    def test_ellipsis(self):
        """Test handling of ellipsis"""
        txt = "Something..."
        expected = [
            Token("Something", 0),
            Token("...", 9)
        ]
        result = tokenize_english_text(txt)
        self.assertEqual([t.text for t in expected], [t.text for t in result])

    def test_time_formats(self):
        """Test handling of time formats"""
        txt = "9 a.m. to 5 p.m."
        expected = [
            Token("9", 0),
            Token(" ", 1),
            Token("a.m.", 2),
            Token(" ", 6),
            Token("to", 7),
            Token(" ", 9),
            Token("5", 10),
            Token(" ", 11),
            Token("p.m.", 12)
        ]
        result = tokenize_english_text(txt)
        self.assertEqual([t.text for t in expected], [t.text for t in result])

    def test_academic_titles(self):
        """Test handling of academic and professional titles"""
        txt = "Dr. Smith, Ph.D."
        expected = [
            Token("Dr.", 0),
            Token(" ", 3),
            Token("Smith", 4),
            Token(",", 9),
            Token(" ", 10),
            Token("Ph.D.", 11)
        ]
        result = tokenize_english_text(txt)
        self.assertEqual([t.text for t in expected], [t.text for t in result])

    def test_mixed_punctuation(self):
        """Test handling of mixed punctuation cases"""
        txt = "Hello?! What's this...?"
        expected = [
            Token("Hello", 0),
            Token("?", 5),
            Token("!", 6),
            Token(" ", 7),
            Token("What's", 8),
            Token(" ", 14),
            Token("this", 15),
            Token("...", 19),
            Token("?", 22)
        ]
        result = tokenize_english_text(txt)
        self.assertEqual([t.text for t in expected], [t.text for t in result])

if __name__ == '__main__':
    unittest.main()
