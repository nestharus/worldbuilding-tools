import logging
import psutil
import subprocess
from pathlib import Path


class SystemResources:
    """Analyzes and reports system capabilities for model deployment."""
    
    logger: logging.Logger
    
    def __init__(self):
        """Initialize SystemResources with logger and cached results."""
        self.logger = logging.getLogger(__name__)
        self._cached_resources = None

    def check_system(self) -> dict[str, any]:
        """Check available system resources and return capabilities."""
        if self._cached_resources is not None:
            return self._cached_resources

        # Initialize with basic system info
        self._cached_resources = {
            'cpu_count': psutil.cpu_count(logical=False),
            'ram_gb': psutil.virtual_memory().total / (1024 ** 3),
            'free_disk_gb': psutil.disk_usage('/').free / (1024 ** 3)
        }

        self.logger.info("System resources:")
        self.logger.info(f"CPU cores: {self._cached_resources['cpu_count']}")
        self.logger.info(f"RAM: {self._cached_resources['ram_gb']:.1f} GB")
        self.logger.info(f"Free disk space: {self._cached_resources['free_disk_gb']:.1f} GB")
        return self._cached_resources

    def get_recommended_models(self) -> dict[str, str]:
        """Determine which models can be supported by the system."""
        resources = self.check_system()

        # Model requirements (in GB)
        SPACY_LG_RAM = 4
        SPACY_MD_RAM = 1

        models = {
            'spacy': 'en_core_web_md',  # Default to medium
            'hf': 'microsoft/deberta-v3-small'  # Default to small DeBERTa
        }

        # SpaCy model selection based on available RAM
        if resources['ram_gb'] >= SPACY_LG_RAM:
            models['spacy'] = 'en_core_web_lg'
        elif resources['ram_gb'] >= SPACY_MD_RAM:
            models['spacy'] = 'en_core_web_md'
        else:
            models['spacy'] = 'en_core_web_sm'

        self.logger.info(f"Recommended models: {models}")
        return models
