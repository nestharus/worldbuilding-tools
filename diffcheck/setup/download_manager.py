import logging
from pathlib import Path
import requests
from progress import ProgressManager


class ModelDownloadManager:
    """Manages downloading of model files with progress tracking."""
    
    progress_manager: ProgressManager
    logger: logging.Logger
    
    def __init__(self):
        """Initialize ModelDownloadManager with progress manager and logger."""
        self.progress_manager = ProgressManager()
        self.logger = logging.getLogger(__name__)

    def download_with_progress(self, url: str, dest_path: Path) -> bool:
        """Download a file with progress tracking."""
        progress_bar = None
        model_name = dest_path.name  # Initialize model_name at the start
        try:
            # Use a session to handle redirects properly
            session = requests.Session()
            response = session.get(url, stream=True)
            response.raise_for_status()  # Raise error for bad status codes
            
            # Ensure proper encoding for text files
            content_type = response.headers.get('content-type', '')
            is_text = 'text' in content_type or 'json' in content_type
            
            total_size = int(response.headers.get('content-length', 0))
            # Initialize progress bar before any operations
            progress_bar = None if total_size <= 0 else self.progress_manager.create_progress_bar(
                model_name, total_size
            )
            
            # Initialize error tracking
            download_error = None
            
            try:
                # Use text mode for JSON/text files, binary for others
                mode = 'w' if is_text else 'wb'
                encoding = 'utf-8' if is_text else None
                
                with open(dest_path, mode, encoding=encoding) as f:
                    if is_text:
                        # For text files, download completely then write
                        content = response.text
                        f.write(content)
                        self.progress_manager.update_progress(model_name, len(content.encode('utf-8')))
                    else:
                        # For binary files, stream the download
                        for data in response.iter_content(chunk_size=1024):
                            size = f.write(data)
                            self.progress_manager.update_progress(model_name, size)

                return True

            except Exception as e:
                download_error = e
                raise e

            finally:
                if progress_bar:
                    self.progress_manager.close_progress(
                        model_name, 
                        "completed" if download_error is None else "failed"
                    )

        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error downloading {url}: {e}")
            if progress_bar:
                self.progress_manager.close_progress(model_name, "failed")
            return False
        except Exception as e:
            self.logger.error(f"Download failed for {url}: {e}")
            if 'model_name' in locals() and progress_bar:
                self.progress_manager.close_progress(model_name, "failed")
            return False
