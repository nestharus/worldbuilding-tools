import logging

from rich.logging import RichHandler


def setup_rich_logging():
    """Configure enhanced logging with Rich handler."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(rich_tracebacks=True, markup=True),
            logging.FileHandler("tokenizer_setup.log")
        ]
    )

    # Set logging levels for specific modules
    logging.getLogger("huggingface_hub").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.INFO)
    logging.getLogger("spacy").setLevel(logging.INFO)

    return logging.getLogger(__name__)
