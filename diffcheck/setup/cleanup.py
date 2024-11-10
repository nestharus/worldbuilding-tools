import json
import logging
import shutil
from pathlib import Path
from typing import List


class ModelCleaner:
    """Manages cleanup of downloaded models and verification of model integrity."""
    
    logger: logging.Logger
    cache_dir: Path
    versions_file: Path
    hf_cache_dir: Path
    torch_cache_dir: Path
    
    def __init__(self):
        """Initialize the ModelCleaner with logger and paths."""
        self.logger = logging.getLogger(__name__)
        # Our custom cache directory
        self.cache_dir = Path.home() / '.cache' / 'tokenizer_models'
        self.versions_file = self.cache_dir / 'versions.json'
        # HuggingFace cache directory
        self.hf_cache_dir = Path.home() / '.cache' / 'huggingface'
        # PyTorch cache directory
        self.torch_cache_dir = Path.home() / '.cache' / 'torch'

    def print_cache_locations(self) -> None:
        """Print the full paths of all model cache directories."""
        self.logger.info("Model cache locations:")
        self.logger.info(f"Custom tokenizer models: {self.cache_dir.absolute()}")
        self.logger.info(f"HuggingFace cache: {self.hf_cache_dir.absolute()}")
        self.logger.info(f"PyTorch cache: {self.torch_cache_dir.absolute()}")

    def get_directory_size(self, directory: Path) -> float:
        """Calculate total size of a directory in GB."""
        try:
            total = sum(f.stat().st_size for f in directory.glob('**/*') if f.is_file())
            return total / (1024 ** 3)  # Convert to GB
        except Exception as e:
            self.logger.error(f"Error calculating size for {directory}: {e}")
            return 0.0

    def clean_unused_models(self) -> None:
        """
        Remove unused model files to free up space.
        
        Cleans up files from:
        - Custom tokenizer_models cache
        - HuggingFace cache
        - PyTorch cache
        """
        try:
            # Track initial sizes
            initial_sizes = {
                'tokenizer': self.get_directory_size(self.cache_dir),
                'huggingface': self.get_directory_size(self.hf_cache_dir),
                'torch': self.get_directory_size(self.torch_cache_dir)
            }
            
            total_initial = sum(initial_sizes.values())
            self.logger.info(f"Initial cache sizes:")
            self.logger.info(f"- Tokenizer models: {initial_sizes['tokenizer']:.2f} GB")
            self.logger.info(f"- HuggingFace cache: {initial_sizes['huggingface']:.2f} GB")
            self.logger.info(f"- PyTorch cache: {initial_sizes['torch']:.2f} GB")
            self.logger.info(f"Total initial size: {total_initial:.2f} GB")

            # Get active models from versions file
            active_models = set()
            if self.versions_file.exists():
                with open(self.versions_file, 'r') as f:
                    active_models = set(json.load(f).keys())
            
            # Clean tokenizer models cache
            if self.cache_dir.exists():
                for file_path in self.cache_dir.glob('**/*'):
                    if file_path.is_file() and not any(model in str(file_path) for model in active_models):
                        try:
                            file_path.unlink()
                        except Exception as e:
                            self.logger.error(f"Error deleting {file_path}: {e}")
                    elif file_path.is_dir() and not any(model in str(file_path) for model in active_models):
                        try:
                            shutil.rmtree(file_path)
                        except Exception as e:
                            self.logger.error(f"Error removing directory {file_path}: {e}")

            # Clean HuggingFace cache
            if self.hf_cache_dir.exists():
                try:
                    shutil.rmtree(self.hf_cache_dir)
                    self.logger.info("Cleaned HuggingFace cache")
                except Exception as e:
                    self.logger.error(f"Error cleaning HuggingFace cache: {e}")

            # Clean PyTorch cache
            if self.torch_cache_dir.exists():
                try:
                    shutil.rmtree(self.torch_cache_dir)
                    self.logger.info("Cleaned PyTorch cache")
                except Exception as e:
                    self.logger.error(f"Error cleaning PyTorch cache: {e}")

            # Calculate space saved
            final_sizes = {
                'tokenizer': self.get_directory_size(self.cache_dir),
                'huggingface': self.get_directory_size(self.hf_cache_dir),
                'torch': self.get_directory_size(self.torch_cache_dir)
            }
            
            total_final = sum(final_sizes.values())
            total_saved = total_initial - total_final

            self.logger.info(f"\nSpace cleaned:")
            self.logger.info(f"- Tokenizer models: {initial_sizes['tokenizer'] - final_sizes['tokenizer']:.2f} GB")
            self.logger.info(f"- HuggingFace cache: {initial_sizes['huggingface'] - final_sizes['huggingface']:.2f} GB")
            self.logger.info(f"- PyTorch cache: {initial_sizes['torch'] - final_sizes['torch']:.2f} GB")
            self.logger.info(f"Total space saved: {total_saved:.2f} GB")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def verify_models(self) -> None:
        """
        Verify integrity of installed models.
        
        Attempts to load each installed model to verify it's working correctly.
        Handles both spaCy and Hugging Face models differently based on their type.
        Logs success or failure for each model verification attempt.
        """
        from transformers import DebertaV2Tokenizer
        import spacy

        try:
            with open(self.versions_file, 'r') as f:
                installed_models = json.load(f)

            for model_key, version in installed_models.items():
                try:
                    if model_key.startswith('spacy_'):
                        model_name = model_key.replace('spacy_', '')
                        nlp = spacy.load(model_name)
                        self.logger.info(f"Verified spaCy model: {model_name}")
                    elif model_key.startswith('hf_'):
                        model_name = model_key.replace('hf_', '')
                        # Use consistent path sanitization
                        model_dir = self.cache_dir / model_name.replace('/', '-')
                        tokenizer = DebertaV2Tokenizer.from_pretrained(
                            str(model_dir)
                        )
                        self.logger.info(f"Verified Hugging Face model: {model_name}")
                except Exception as e:
                    self.logger.error(f"Model verification failed for {model_key}: {e}")

        except Exception as e:
            self.logger.error(f"Error during model verification: {e}")
