import json
import logging

from pathlib import Path
from huggingface_hub import HfApi
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from download_manager import ModelDownloadManager
from progress import ProgressManager
from setup_logging import setup_rich_logging


class ModelInstaller:
    """Manages installation and verification of language models."""
    
    progress_manager: ProgressManager
    download_manager: ModelDownloadManager
    logger: logging.Logger
    recommended_models: dict[str, str]
    cache_dir: Path
    versions_file: Path
    versions: dict[str, str]
    progress: Progress
    setup_task: int

    def __init__(self):
        self.progress_manager = ProgressManager()
        self.download_manager = ModelDownloadManager()
        self.logger = logging.getLogger(__name__)
        setup_rich_logging()

        # Initialize cache directory
        self.cache_dir = Path.home() / '.cache' / 'tokenizer_models'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.cache_dir / 'versions.json'
        self.versions = self._load_versions()

        with Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                console=Console()
        ) as self.progress:
            self.setup_task = self.progress.add_task("Setup", total=100)

    def _load_versions(self):
        """Load or create versions tracking file."""
        if self.versions_file.exists():
            return json.loads(self.versions_file.read_text())
        return {}

    def install_all(self):
        """Install all recommended models with progress tracking."""
        models = [
            'hf-microsoft/deberta-v3-xsmall'
        ]
        total_steps = len(models)
        completed_steps = 0

        self.logger.info("[bold blue]Starting model installation...[/]")

        for model in models:
            self.progress.update(
                self.setup_task,
                description=f'Installing {model}',
                completed=int((completed_steps / total_steps) * 100)
            )
            success = False
            if model.startswith('hf-'):
                model = model[3:]
                success = self.install_hf_model(model)

            completed_steps += 1
            if not success:
                self.logger.error(f'Failed to install {model}')

            with self.versions_file.open('w') as f:
                f.write(json.dumps(self.versions))

        self.install_hf_model('microsoft/deberta-v3-xsmall')

        self.progress.update(
            self.setup_task,
            description="Installation complete",
            completed=100
        )
        self.logger.info("[bold green]Installation complete![/]")

    def install_hf_model(self, model_id: str) -> bool:
        """Install a Hugging Face model."""
        api = HfApi()
        model = api.model_info(model_id)
        files = {
            "config.json",
            "tokenizer_config.json",
            "spm.model",
        }

        model_dir = self.cache_dir / model.id.replace('/', '-')
        model_dir.mkdir(parents=True, exist_ok=True)

        current_hash = self.versions.get(f"hf_{model.id}")
        if current_hash is None:
            self.logger.info(f'No version information found for {model.id}')

        if current_hash != model.sha:
            invalid_files = files
            self.logger.info(f'Downloading {model.id}')
        else:
            invalid_files = [
                filename
                for filename in files
                if not (model_dir / filename).exists()
            ]
            if len(invalid_files) == 0:
                self.logger.info(f'{model.id} is up to date')
            else:
                self.logger.info(f'Downloading missing files for {model.id}')


        for filename in invalid_files:
            self.logger.info(f'Downloading {model.id}/{filename}')

            file_url = f'https://huggingface.co/{model.id}/resolve/main/{filename}'
            dest_path = model_dir / filename
            self.download_manager.download_with_progress(file_url, dest_path)

        self.versions[f"hf_{model.id}"] = model.sha

        return True
