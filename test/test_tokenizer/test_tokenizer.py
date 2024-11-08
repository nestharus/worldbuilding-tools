import unittest
from diffcheck.src.tokenizer.tokenizer import tokenize_english_text, process_abbreviations
from diffcheck.src.tokenizer.token import Token

class TestTokenizer(unittest.TestCase):
    def test_ellipsis(self):
        """Test handling of ellipsis"""
        # Test three dots
        tokens = tokenize_english_text("Something...")
        self.assertEqual(['Something', '...'], [t.text for t in tokens])
        
        # Test ellipsis character
        tokens = tokenize_english_text("Something…")
        self.assertEqual(['Something', '…'], [t.text for t in tokens])

    def test_contractions(self):
        """Test handling of contractions"""
        tokens = tokenize_english_text("can't do that")
        self.assertEqual(["can't", " ", "do", " ", "that"], [t.text for t in tokens])
        
        tokens = tokenize_english_text("jess' book")
        self.assertEqual(["jess'", " ", "book"], [t.text for t in tokens])

    def test_hyphenated_words(self):
        """Test handling of hyphenated compounds"""
        tokens = tokenize_english_text("do-good person")
        self.assertEqual(['do-good', ' ', 'person'], [t.text for t in tokens])
        
        tokens = tokenize_english_text("do-good-with others")
        self.assertEqual(['do-good-with', ' ', 'others'], [t.text for t in tokens])

    def test_initials_and_dots(self):
        """Test handling of initials and multiple dots"""
        tokens = tokenize_english_text("Ron, I.")
        self.assertEqual(['Ron', ',', ' ', 'I', '.'], [t.text for t in tokens])
        
        tokens = tokenize_english_text("Oh. I.")
        self.assertEqual(['Oh', '.', ' ', 'I', '.'], [t.text for t in tokens])
        
        tokens = tokenize_english_text("Ah..")
        self.assertEqual(['Ah', '.', '.'], [t.text for t in tokens])

    def test_complex_abbreviations(self):
        """Test handling of complex abbreviations"""
        # Test basic abbreviation with one period
        tokens = tokenize_english_text("U.S.A.")
        self.assertEqual(['U.S.A.'], [t.text for t in tokens])
        
        # Test abbreviation with extra periods
        tokens = tokenize_english_text("U.S.A..")
        self.assertEqual(['U.S.A.', '.'], [t.text for t in tokens])
        
        # Test direct abbreviation processing
        from diffcheck.src.tokenizer.tokenizer import process_abbreviations
        input_tokens = [
            Token("U.S.A", 0),
            Token(".", 5),
            Token(".", 6)
        ]
        expected = [
            Token("U.S.A.", 0),
            Token(".", 6)
        ]
        result = process_abbreviations(input_tokens)
        self.assertEqual([t.text for t in expected], [t.text for t in result])

    def test_titles(self):
        """Test handling of titles with periods"""
        tokens = tokenize_english_text("Mr. Smith and Dr. Jones")
        self.assertEqual(['Mr.', ' ', 'Smith', ' ', 'and', ' ', 'Dr.', ' ', 'Jones'], 
                        [t.text for t in tokens])

    def test_mixed_case_abbreviations(self):
        """Test handling of mixed-case abbreviations"""
        tokens = tokenize_english_text("PhD. in CS from MIT.")
        self.assertEqual(['PhD.', ' ', 'in', ' ', 'CS', ' ', 'from', ' ', 'MIT.'], 
                        [t.text for t in tokens])

    def test_with_periods(self):
        """Test handling of abbreviations with extra periods"""
        input_tokens = [
            Token("U.S.A", 0),
            Token(".", 5),
            Token(".", 6)
        ]
        result = process_abbreviations(input_tokens)
        self.assertEqual(["U.S.A.", "."], [t.text for t in result])

    def test_ordinals(self):
        """Test handling of ordinal numbers"""
        tokens = tokenize_english_text("1st. 2nd. and 3rd.")
        self.assertEqual(['1st.', ' ', '2nd.', ' ', 'and', ' ', '3rd.'], 
                        [t.text for t in tokens])

    def test_time_formats(self):
        """Test handling of time formats"""
        tokens = tokenize_english_text("9 a.m. to 5 p.m.")
        self.assertEqual(['9', ' ', 'a.m.', ' ', 'to', ' ', '5', ' ', 'p.m.'], 
                        [t.text for t in tokens])

    def test_academic_titles(self):
        """Test handling of academic and professional titles"""
        text = "John Smith, M.D., and Jane Doe, Ph.D., work at Corp."
        tokens = tokenize_english_text(text)
        self.assertEqual(
            ['John', ' ', 'Smith', ',', ' ', 'M.D.', ',', ' ', 'and', ' ', 
             'Jane', ' ', 'Doe', ',', ' ', 'Ph.D.', ',', ' ', 'work', ' ', 'at', ' ', 'Corp.'],
            [t.text for t in tokens]
        )

    def test_whitespace_handling(self):
        """Test basic whitespace handling"""
        tokens = tokenize_english_text("Hello  World")
        self.assertEqual(['Hello', '  ', 'World'], [t.text for t in tokens])

        tokens = tokenize_english_text("\tHello\nWorld")
        self.assertEqual(['\t', 'Hello', '\n', 'World'], [t.text for t in tokens])

    def test_numbers_and_punctuation(self):
        """Test handling of numbers and basic punctuation"""
        tokens = tokenize_english_text("The price is $19.99!")
        self.assertEqual(['The', ' ', 'price', ' ', 'is', ' ', '$', '19.99', '!'], 
                        [t.text for t in tokens])

    def test_mixed_punctuation(self):
        """Test handling of mixed punctuation cases"""
        tokens = tokenize_english_text("Hello?! What's this...?")
        self.assertEqual(['Hello', '?', '!', ' ', "What's", ' ', 'this', '...', '?'], 
                        [t.text for t in tokens])

    def test_empty_and_spaces(self):
        """Test handling of empty string and multiple spaces"""
        tokens = tokenize_english_text("")
        self.assertEqual([], [t.text for t in tokens])

        tokens = tokenize_english_text("   ")
        self.assertEqual(['   '], [t.text for t in tokens])

    def test_special_characters(self):
        """Test handling of special characters"""
        tokens = tokenize_english_text("©2023 — Example™")
        self.assertEqual(['©', '2023', ' ', '—', ' ', 'Example', '™'], 
                        [t.text for t in tokens])

if __name__ == '__main__':
    unittest.main()
