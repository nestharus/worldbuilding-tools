import numpy as np
import pytest
import logging
from typing import List
import spacy
from spacy.tokens import Doc

from tokenizer.compound_classifier import CompoundClassifier
from tokenizer.spacy_tokenizer import spacy_tokenizer


# Test fixtures and data
# @pytest.fixture(scope="module")
# def nlp():
#     """
#     Fixture to provide spaCy model for all tests.
#     Uses the provided spacy_tokenizer function.
#     """
#
#     return spacy_tokenizer()
#
#
# @pytest.fixture(scope="module")
# def classifier(nlp):
#     """
#     Fixture to provide compound classifier instance.
#     """
#     return CompoundClassifier(nlp)
#
#
# @pytest.fixture
# def get_compound_tokens():
#     """Helper fixture to get tokens from text."""
#     def _get_tokens(nlp: spacy.language.Language, text: str) -> List[spacy.tokens.Token]:
#         doc = nlp(text)
#         return [token for token in doc]
#     return _get_tokens


# Test data with examples we know should work with the model
LEXICAL_COMPOUNDS = [
    "merry-go-round",
    "x-ray",
    "ice-cream"
]

COMPOSITIONAL_COMPOUNDS = [
    "cost-benefit",
    "parent-child",
    "client-server"
]


@pytest.mark.unit
class TestCompoundClassifier:
    pass
    # @pytest.mark.parametrize("compound_text", [
    #     "merry-go-round",  # Clearly lexical - carnival ride
    #     "x-ray",  # Lexical - medical term
    #     "ice-cream"  # Lexical - food item
    # ])
    # def test_lexical_compounds(self, classifier, nlp, get_compound_tokens, compound_text):
    #     """Test that known lexical compounds are classified correctly"""
    #     tokens = get_compound_tokens(nlp, compound_text)
    #     classification, confidence, scores = classifier.classify_compound(tokens)
    #
    #     assert classification == 'lexical', f"Expected '{compound_text}' to be lexical, got {classification}"
    #     assert round(confidence, 1) >= round(0.6, 1), f"Expected high confidence for clear lexical compound '{compound_text}'"
    #
    # @pytest.mark.parametrize("compound_text", [
    #     "cost-benefit",  # Compositional - describes relationship
    #     "north-south",  # Compositional - directional
    #     "buyer-seller",  # Compositional - relationship
    #     "input-output"  # Compositional - process relationship
    # ])
    # def test_compositional_compounds(self, classifier, nlp, get_compound_tokens, compound_text):
    #     """Test that known compositional compounds are classified correctly"""
    #     tokens = get_compound_tokens(nlp, compound_text)
    #     classification, confidence, scores = classifier.classify_compound(tokens)
    #
    #     assert classification == 'compositional', f"Expected '{compound_text}' to be compositional, got {classification}"
    #     assert round(confidence, 1) >= round(0.6, 1), f"Expected high confidence for clear compositional compound '{compound_text}'"
    #
    # def test_borderline_cases(self, classifier, nlp, get_compound_tokens):
    #     """Test compounds that could reasonably be either classification"""
    #     borderline_cases = [
    #         ("well-being", 0.4),  # Could be either, but tends lexical
    #         ("non-stop", 0.4),  # Could be either, but tends lexical
    #         ("end-user", 0.5)  # Truly ambiguous
    #     ]
    #
    #     for compound, threshold in borderline_cases:
    #         tokens = get_compound_tokens(nlp, compound)
    #         _, confidence, _ = classifier.classify_compound(tokens)
    #
    #         assert confidence <= threshold, \
    #             f"Expected low confidence for ambiguous compound '{compound}', got {confidence}"


if __name__ == "__main__":
    pytest.main([__file__])
