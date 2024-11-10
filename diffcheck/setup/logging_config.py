import logging
import logging.handlers
from pathlib import Path


def setup_logging():
    """Configure logging with rotation and structured format."""
    log_dir = Path.home() / '.cache' / 'tokenizer_models' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'tokenizer.log'

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    console.setFormatter(console_formatter)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Add handlers
    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger
