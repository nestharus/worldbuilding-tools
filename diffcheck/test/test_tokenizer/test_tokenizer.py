import pytest

from tokenizer.context_aware_tokenizer import ContextAwareTokenizer
from tokenizer.deberta_tokenizer import DebertaTokenizer
from tokenizer.spacy_tokenizer import spacy_tokenizer


@pytest.fixture
def tokenize():
    deberta = DebertaTokenizer()
    spacy = spacy_tokenizer()
    tokenizer = ContextAwareTokenizer(deberta, spacy)

    return lambda text: [tokenizer.deberta_tokenizer.tokenizer.convert_ids_to_tokens(token[2]) for token in tokenizer.tokenize(text)]


@pytest.mark.unit
def test_contraction(tokenize):
    """Test handling of contractions"""
    assert tokenize('can''t') == ['can''t']


@pytest.mark.unit
def test_acronym(tokenize):
    """Test handling of acronyms with periods"""
    assert tokenize('U.S.A.') == ['u.s.a.']


@pytest.mark.unit
def test_pronoun_i_with_period(tokenize):
    assert tokenize('I want it') == ['i', 'want', 'it']


@pytest.mark.unit
def test_pronoun_i_with_period(tokenize):
    assert tokenize('Was I.') == ['was', 'i']


@pytest.mark.unit
def test_underscore(tokenize):
    assert tokenize('▁▁Was') == ['▁▁was']


@pytest.mark.unit
def test_underscore2(tokenize):
    assert tokenize('__Was') == ['__was']


@pytest.mark.unit
def test_underscore3(tokenize):
    assert tokenize('_it_is_ok') == ['_it_is_ok']


@pytest.mark.unit
def test_underscore4(tokenize):
    assert tokenize('it__is__ok') == ['it__is__ok']


@pytest.mark.unit
def test_underscore5(tokenize):
    assert tokenize('_____') == []


@pytest.mark.unit
def test_initial_i_with_period(tokenize):
    """Test handling of initial 'I' with period"""
    assert tokenize('Bob, I.') == ['bob', 'i.']


@pytest.mark.unit
def test_initial_i_with_ellipsis(tokenize):
    """Test handling of initial 'I' with ellipsis"""
    assert tokenize('Bob, I...') == ['bob', 'i']


@pytest.mark.unit
def test_fake_initial_i(tokenize):
    """Test handling of initial 'I' with ellipsis"""
    assert tokenize('Well Bob, I.') == ['well', 'bob', 'i']


@pytest.mark.unit
def test_roman_numeral(tokenize):
    assert tokenize('I.') == []


@pytest.mark.unit
def test_ambiguous_i(tokenize):
    assert tokenize('Hey Bob, I.') == ['hey', 'bob', 'i']


@pytest.mark.unit
def test_initial_i_after_name(tokenize):
    """Test handling of initial 'I' after name with contraction"""
    assert tokenize('Their name is Bob, I.') == ['their', 'name', 'is', 'bob', 'i.']


@pytest.mark.unit
def test_multiple_initials(tokenize):
    """Test handling of multiple initials"""
    assert tokenize('Dr. J. Smith') == ['dr.', 'j.', 'smith']


@pytest.mark.unit
def test_pronoun_i_with_verb(tokenize):
    """Test handling of pronoun 'I' with thinking verb"""
    assert tokenize('I think I.') == ['i', 'think', 'i']


@pytest.mark.unit
def test_full_initials(tokenize):
    """Test handling of full name with initials"""
    assert tokenize('A. B. Smith') == ['a.', 'b.', 'smith']


@pytest.mark.unit
def test_ellipsis_after_word(tokenize):
    """Test handling of ellipsis after word"""
    assert tokenize('And then...') == ['and', 'then']


@pytest.mark.unit
def test_ellipsis_mid_sentence(tokenize):
    """Test handling of ellipsis in middle of sentence"""
    assert tokenize('He said... well') == ['he', 'said', 'well']


@pytest.mark.unit
def test_ellipsis_after_title(tokenize):
    """Test handling of ellipsis after title"""
    assert tokenize('Mr. Smith...') == ['mr.', 'smith']


@pytest.mark.unit
def test_dash_word(tokenize):
    """Test handling of ellipsis after title"""
    assert tokenize('merry-go-round') == ['merry', 'go', 'round']


@pytest.mark.unit
def test_dash_word2(tokenize):
    """Test handling of ellipsis after title"""
    # 0.3671875
    assert tokenize('well-to-do') == ['well', 'to', 'do']


@pytest.mark.unit
def test_fictional_world(tokenize):
    """Test handling of ellipsis after title"""
    assert tokenize('Däor') == ['däor']


@pytest.mark.unit
def test_range(tokenize):
    """Test handling of ellipsis after title"""
    # .4948
    assert tokenize('one-seven') == ['one', 'seven']


@pytest.mark.unit
def test_range2(tokenize):
    """Test handling of ellipsis after title"""
    # .53
    assert tokenize('jan-feb') == ['jan', 'feb']


@pytest.mark.unit
def test_range3(tokenize):
    """Test handling of ellipsis after title"""
    # .41
    assert tokenize('january-february') == ['january', 'february']


@pytest.mark.unit
def test_range4(tokenize):
    """Test handling of ellipsis after title"""
    assert tokenize('january-11') == ['january-11']


@pytest.mark.unit
def test_empty_string(tokenize):
    """Test handling of empty string"""
    assert tokenize("") == []


@pytest.mark.unit
def test_single_period(tokenize):
    """Test handling of single period"""
    assert tokenize('.') == []


@pytest.mark.unit
def test_multiple_spaces(tokenize):
    """Test handling of multiple spaces"""
    assert tokenize('Hello   world') == ['hello', 'world']


@pytest.mark.edge_cases
class TestEdgeCases:
    @pytest.mark.unit
    def test_fictional_places(self, tokenize):
        """Test handling of fictional place names"""
        assert tokenize('Visiting Na''vi on Pandora') == ['visiting', 'na''vi', 'on', 'pandora']
        assert tokenize('From Qa''pla, Qo''noS') == ['from', 'qa''pla', 'qo''nos']

    @pytest.mark.unit
    def test_mixed_case_words(self, tokenize):
        """Test handling of mixed case words"""
        assert tokenize('iPhone and MacBook') == ['iphone', 'macbook']
        assert tokenize('PostgreSQL database') == ['postgresql', 'database']

    @pytest.mark.unit
    def test_special_characters(self, tokenize):
        """Test handling of special characters"""
        assert tokenize('Hello@World.com') == ['hello@world.com']
        assert tokenize('C++ and C#') == ['c++', 'c#']

    @pytest.mark.unit
    def test_unicode_characters(self, tokenize):
        """Test handling of Unicode characters"""
        assert tokenize('café and résumé') == ['café', 'résumé']
        assert tokenize('über schön') == ['über', 'schön']

    @pytest.mark.unit
    def test_emoji_handling(self, tokenize):
        """Test handling of emoji characters"""
        assert tokenize('Hello 👋 World 🌍') == ['hello', '👋', 'world', '🌍']

    @pytest.mark.unit
    def test_multiple_punctuation(self, tokenize):
        """Test handling of multiple punctuation marks"""
        assert tokenize('Really?!?!') == ['really']
        assert tokenize('Wait...???') == ['wait']

    @pytest.mark.unit
    def test_repeated_characters(self, tokenize):
        """Test handling of repeated characters"""
        assert tokenize('Heeeello!!!') == ['heeeello']
        assert tokenize('Nooooo....') == ['nooooo']

    @pytest.mark.unit
    def test_alphanumeric_combinations(self, tokenize):
        """Test handling of alphanumeric combinations"""
        assert tokenize('3M and 7-Eleven') == ['3m', '7', 'eleven']
        assert tokenize('Win32 and API') == ['win32', 'api']


# @pytest.mark.error_handling
# class TestErrorHandling:
#     @pytest.mark.unit
#     def test_none_input(self, tokenize):
#         """Test handling of None input"""
#         with pytest.raises(TokenizerError) as exc_info:
#             tokenize(None)
#         assert "Input must be string" in str(exc_info.value)
#
#     @pytest.mark.unit
#     def test_invalid_type_input(self, tokenize):
#         """Test handling of invalid input types"""
#         with pytest.raises(TokenizerError) as exc_info:
#             tokenize(123)
#         assert "Input must be string" in str(exc_info.value)
#
#     @pytest.mark.unit
#     def test_invalid_unicode(self, tokenize):
#         """Test handling of invalid Unicode sequences"""
#         invalid_unicode = "Hello \ud800 World"  # Invalid surrogate pair
#         result = tokenize(invalid_unicode)
#         assert result == ['hello', 'world']

    # @pytest.mark.unit
    # def test_extremely_long_input(self, tokenize):
    #     """Test handling of extremely long input"""
    #     long_input = "word " * 10000
    #     result = tokenize(long_input)
    #     assert len(result) > 0
    #     assert all(token == "word" for token in result)

    # @pytest.mark.unit
    # def test_zero_width_spaces(self, tokenize):
    #     """Test handling of zero-width spaces and other invisible characters"""
    #     text_with_zero_width = "hello\u200bworld"
    #     assert tokenize(text_with_zero_width) == ['hello', 'world']
    #
    # @pytest.mark.unit
    # def test_control_characters(self, tokenize):
    #     """Test handling of control characters"""
    #     text_with_control = "hello\x00world\x01"
    #     assert tokenize(text_with_control) == ['hello', 'world']

    # @pytest.mark.unit
    # @pytest.mark.xfail(strict=True)
    # def test_memory_error_simulation(self, tokenize):
    #     """Test handling of potential memory errors with massive input"""
    #     massive_input = "a" * (10 ** 8)  # Might raise MemoryError
    #     tokenize(massive_input)


# @pytest.mark.performance
# class TestPerformance:
#     @pytest.mark.unit
#     def test_many_periods(self, tokenize):
#         """Test performance with many periods"""
#         text = "." * 1000
#         result = tokenize(text)
#         assert len(result) == 0
#
#     @pytest.mark.unit
#     def test_many_spaces(self, tokenize):
#         """Test performance with many spaces"""
#         text = " " * 1000 + "hello" + " " * 1000
#         assert tokenize(text) == ['hello']
#
#     @pytest.mark.unit
#     def test_many_newlines(self, tokenize):
#         """Test performance with many newlines"""
#         text = "\n" * 1000 + "hello" + "\n" * 1000
#         assert tokenize(text) == ['hello']


if __name__ == "__main__":
    pytest.main(['-v', __file__])
