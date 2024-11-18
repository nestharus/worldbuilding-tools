import logging

import spacy


def spacy_tokenizer():
    logger = logging.getLogger(__name__)

    try:
        tokenizer = spacy.load('en_core_web_sm')
        return tokenizer
    except Exception as e:
        logger.error(f"Error initializing tokenizer: {e}")
        raise
