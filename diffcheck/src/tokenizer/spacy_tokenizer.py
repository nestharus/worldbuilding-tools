import logging

import spacy
from spacy.tokens import Token

from system_check import SystemResources


def spacy_tokenizer():
    logger = logging.getLogger(__name__)
    sys_check = SystemResources()
    recommended_models = sys_check.get_recommended_models()
    model = recommended_models['spacy']
    try:
        tokenizer = spacy.load(model)
        if not Token.has_extension("trf_data"):
            Token.set_extension("trf_data", default=None)
        logger.info(f"Initialized with spaCy={model}")
        return tokenizer
    except Exception as e:
        logger.error(f"Error initializing tokenizer: {e}")
        raise
