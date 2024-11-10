from tqdm import tqdm
from rich.console import Console
import logging
from typing import Optional
from dataclasses import dataclass


@dataclass
class DownloadProgress:
    """Track download progress for a specific model."""
    model_name: str
    total_size: int
    downloaded: int = 0
    status: str = "pending"
    progress_bar: Optional[tqdm] = None


class ProgressManager:
    """Manages progress bars and download status tracking for model downloads."""
    
    console: Console
    downloads: dict[str, DownloadProgress]
    logger: logging.Logger
    
    def __init__(self):
        """Initialize ProgressManager with console, downloads tracking, and logger."""
        self.console = Console()
        self.downloads: dict[str, DownloadProgress] = {}
        self.logger = logging.getLogger(__name__)

    def create_progress_bar(self, model_name: str, total_size: int) -> tqdm:
        """Create a progress bar for a model download."""
        progress_bar = tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc=f"Downloading {model_name}",
            leave=True
        )
        self.downloads[model_name] = DownloadProgress(
            model_name=model_name,
            total_size=total_size,
            progress_bar=progress_bar
        )
        return progress_bar

    def update_progress(self, model_name: str, bytes_downloaded: int):
        """Update progress for a specific model."""
        if model_name in self.downloads:
            download = self.downloads[model_name]
            download.downloaded += bytes_downloaded
            if download.progress_bar:
                download.progress_bar.update(bytes_downloaded)

    def close_progress(self, model_name: str, status: str = "completed"):
        """Close progress bar for a model."""
        if model_name in self.downloads:
            download = self.downloads[model_name]
            if download.progress_bar:
                download.progress_bar.close()
            download.status = status
            self.logger.info(f"Download {status} for {model_name}")
