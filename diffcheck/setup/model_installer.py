import json
import logging
import requests
import spacy
import pkg_resources
from pathlib import Path
from huggingface_hub import HfApi
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from transformers import AutoTokenizer

from download_manager import ModelDownloadManager
from progress import ProgressManager
from setup_logging import setup_rich_logging
from system_check import SystemResources


class ModelInstaller:
    """Manages installation and verification of language models."""
    
    progress_manager: ProgressManager
    download_manager: ModelDownloadManager
    logger: logging.Logger
    sys_check: SystemResources
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
            
        # Initialize system resources checker and cache results
        self.sys_check = SystemResources()
        self.sys_check.check_system()  # Cache the results
        # Initialize recommended models right away
        self.recommended_models = self.sys_check.get_recommended_models()

    def _configure_model_loading(self):
        """Configure model loading settings and suppress known warnings."""
        import warnings
        import torch
        
        # Suppress known FutureWarning about pickle usage in torch.load
        warnings.filterwarnings('ignore', category=FutureWarning, 
                              module='thinc.shims.pytorch',
                              message='You are using `torch.load` with `weights_only=False`')
        
        # Basic PyTorch configuration
        torch.set_grad_enabled(False)
        torch.set_default_tensor_type(torch.FloatTensor)


    def _verify_model_files(self, model_dir: Path) -> bool:
        """
        Verify model files exist and have correct formats.
        Returns True if model files are valid.
        """
        try:
            self.logger.info("\n=== Model File Verification ===")
            
            # Check required configuration files
            config_files = ["config.json", "tokenizer_config.json"]
            for config_file in config_files:
                config_path = model_dir / config_file
                if not config_path.exists():
                    self.logger.error(f"Missing required config file: {config_file}")
                    return False
                try:
                    with open(config_path, 'r') as f:
                        json.load(f)  # Verify JSON is valid
                    self.logger.info(f"✓ Verified {config_file}")
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON in {config_file}")
                    return False
            
            # Check for safetensors model file only
            safetensors_path = model_dir / "model.safetensors"
            if not safetensors_path.exists():
                self.logger.error("Required model.safetensors file not found")
                return False
                
            size_gb = safetensors_path.stat().st_size / (1024**3)
            if size_gb < 0.1:  # 100MB minimum
                self.logger.error(f"model.safetensors is too small: {size_gb:.2f} GB")
                return False
                
            self.logger.info(f"✓ Using model.safetensors: {size_gb:.2f} GB")
                
            self.logger.info("All model files verified successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Model verification failed: {e}")
            return False

    def _load_versions(self):
        """Load or create versions tracking file."""
        if self.versions_file.exists():
            return json.loads(self.versions_file.read_text())
        return {}

    def install_all(self, force_update: bool = False):
        """Install all recommended models with progress tracking."""
        total_steps = len(self.recommended_models) * 2  # Download and verify for each
        completed_steps = 0

        self.logger.info("[bold blue]Starting model installation...[/]")
        self.logger.info(f"Force update mode: {'enabled' if force_update else 'disabled'}")

        # Log selected models
        self.logger.info(f"Installing recommended models: {self.recommended_models}")

        for model_type, model_name in self.recommended_models.items():
            self.progress.update(
                self.setup_task,
                description=f"Installing {model_name}",
                completed=int((completed_steps / total_steps) * 100)
            )

            success = False
            if model_type == 'spacy':
                success = self.install_spacy_model(model_name, force_update)
                if not success:
                    # Try fallback spaCy models
                    for fallback in ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']:
                        self.logger.info(f"Trying fallback spaCy model: {fallback}")
                        if self.install_spacy_model(fallback, force_update):
                            success = True
                            break
            else:  # DeBERTa model
                success = self.install_hf_model(model_name, force_update)
                if not success and model_name != 'microsoft/deberta-v3-base':
                    # Try fallback DeBERTa model
                    self.logger.info("Trying fallback DeBERTa model: deberta-v3-base")
                    success = self.install_hf_model('microsoft/deberta-v3-base', force_update)

            completed_steps += 1
            if not success:
                self.logger.error(f"Failed to install {model_name} and its fallbacks")
                continue

            self.progress.update(
                self.setup_task,
                description=f"Verifying {model_name}",
                completed=int((completed_steps / total_steps) * 100)
            )

            completed_steps += 1

        self.progress.update(
            self.setup_task,
            description="Installation complete",
            completed=100
        )
        self.logger.info("[bold green]Installation complete![/]")

        # Log final status
        self.logger.info("Final installation status:")
        for model_type, model_name in self.recommended_models.items():
            installed_version = self.versions.get(f"{model_type}_{model_name}")
            status = "[green]Installed[/]" if installed_version else "[red]Failed[/]"
            self.logger.info(f"{model_name}: {status} (version: {installed_version})")

    def _check_hf_model_update(self, model_name: str) -> bool:
        """Check if a HuggingFace model has updates available."""
        try:
            api = HfApi()
            model_info = api.model_info(model_name)
            
            # Get model directory
            model_dir = self.cache_dir / model_name.replace('/', '-')
            
            # Check if required files exist
            required_files = ["config.json", "tokenizer_config.json", "spm.model"]
            model_files = ["model.safetensors", "pytorch_model.bin"]
            
            # Check required config files
            if not all((model_dir / file).exists() for file in required_files):
                self.logger.info("One or more required config files missing")
                return True
                
            # Check model files - need either safetensors or pytorch
            model_found = False
            for model_file in model_files:
                file_path = model_dir / model_file
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    if file_size >= 100_000_000:  # 100MB minimum
                        model_found = True
                        break
            
            if not model_found:
                self.logger.info("No valid model file found")
                return True
                
            # Get current version info
            current_hash = self.versions.get(f"hf_{model_name}")
            if not current_hash:
                self.logger.info("No version information found")
                return True
                
            # Check version tag if available
            if model_info.tags:
                current_version = self.versions.get(f"hf_{model_name}_version")
                latest_version = next((tag for tag in model_info.tags if tag.startswith('v')), None)
                if latest_version and current_version != latest_version:
                    self.logger.info(f"Version update available: {current_version} -> {latest_version}")
                    return True
            
            # Compare hashes
            latest_hash = model_info.sha
            if current_hash != latest_hash and current_hash != "latest":
                self.logger.info(f"Hash update available: {current_hash} -> {latest_hash}")
                return True
            
            self.logger.info("Model is up to date")
            return False
        except Exception as e:
            self.logger.warning(f"Failed to check updates for {model_name}: {e}")
            return False

    def install_hf_model(self, model_name: str, force_update: bool = False) -> bool:
        """Install a Hugging Face model."""
        try:
            # Start with requested model, then try fallbacks if it fails
            model_options = [model_name]  # Try requested model first
            
            # Only add fallbacks if the requested model isn't already one
            if model_name not in ["microsoft/deberta-v3-xsmall", "microsoft/deberta-v3-small"]:
                model_options.extend([
                    "microsoft/deberta-v3-base",  # Try base model first
                    "microsoft/deberta-v3-small",
                    "microsoft/deberta-v3-xsmall"  # Smallest as last resort
                ])
            
            for current_model in model_options:
                try:
                    needs_update = force_update
                    if not force_update and f"hf_{current_model}" in self.versions:
                        model_dir = self.cache_dir / current_model.replace('/', '-')
                        
                        # Check for either safetensors or pytorch format
                        model_files = ["model.safetensors", "pytorch_model.bin"]
                        model_found = False
                        
                        for model_file in model_files:
                            model_path = model_dir / model_file
                            if model_path.exists():
                                size_gb = model_path.stat().st_size / (1024**3)
                                if size_gb >= 0.1:  # 100MB minimum
                                    self.logger.info(f"Found existing {model_file}: {size_gb:.2f} GB")
                                    needs_update = self._check_hf_model_update(current_model)
                                    model_found = True
                                    break
                                    
                        if model_found and not needs_update:
                            self.logger.info(f"Model {current_model} is up to date")
                            return True
                            
                        if model_found:
                            self.logger.info(f"Updates available for {current_model}")

                    self.logger.info(f"Downloading Hugging Face model: {current_model}")
                    
                    # DeBERTa models required files
                    files_to_download = [
                        "config.json",
                        "tokenizer_config.json", 
                        "spm.model",
                    ]
                    
                    # Check for model formats
                    safetensors_url = f"https://huggingface.co/{current_model}/resolve/main/model.safetensors"
                    pytorch_url = f"https://huggingface.co/{current_model}/resolve/main/pytorch_model.bin"
                    
                    # Try both formats
                    safetensors_response = requests.head(safetensors_url)
                    pytorch_response = requests.head(pytorch_url)
                    
                    if safetensors_response.status_code == 200:
                        self.logger.info("Safetensors model available")
                        files_to_download.append("model.safetensors")
                    elif pytorch_response.status_code == 200:
                        self.logger.info("PyTorch model available")
                        files_to_download.append("pytorch_model.bin")
                    else:
                        self.logger.warning(f"No compatible model format found for {current_model}, trying next option")
                        continue  # Try next model in the list

                except requests.exceptions.RequestException as e:
                    self.logger.error(f"Error checking model availability: {e}")
                    if "large" in model_name.lower():
                        base_model = model_name.replace("large", "base")
                        self.logger.info(f"Connection error, trying fallback model: {base_model}")
                        return self.install_hf_model(base_model, force_update)
                return False
            
            # Create a sanitized directory name without extra slashes
            model_dir = self.cache_dir / model_name.replace('/', '-')
            model_dir.mkdir(parents=True, exist_ok=True)

            # First verify if all files exist and are valid
            all_files_valid = True
            missing_or_invalid = []
            
            # Check if we already have the latest version
            if not force_update and f"hf_{model_name}" in self.versions:
                api = HfApi()
                model_info = api.model_info(model_name)
                latest_hash = model_info.sha
                current_hash = self.versions.get(f"hf_{model_name}")
                
                # Check model last modified timestamp
                if model_info.lastModified:
                    model_timestamp = model_info.lastModified.timestamp()
                    pytorch_file = model_dir / "pytorch_model.bin"
                    if pytorch_file.exists():
                        local_timestamp = pytorch_file.stat().st_mtime
                        if local_timestamp < model_timestamp:
                            self.logger.info(f"PyTorch model is outdated - update needed")
                            self.logger.info(f"Local timestamp: {local_timestamp}")
                            self.logger.info(f"Remote timestamp: {model_timestamp}")
                            all_files_valid = False
                            missing_or_invalid.append("pytorch_model.bin")
                
                if current_hash == latest_hash and all_files_valid:
                    # Try to verify the model by loading it
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            str(model_dir),
                            local_files_only=True,
                            trust_remote_code=True,
                            use_fast=False,
                            tokenizer_type='slow',
                            slow_tokenizer_class='DebertaV2Tokenizer'
                        )
                        # Test tokenizer with a simple input
                        test_result = tokenizer.encode("Test sentence", add_special_tokens=True)
                        if test_result and len(test_result) > 0:
                            self.logger.info(f"Existing model installation is valid and up to date - skipping download")
                            return True
                    except Exception as e:
                        self.logger.info(f"Existing model needs verification despite matching hash: {e}")
                        all_files_valid = False
                else:
                    self.logger.info(f"Model update available: {current_hash} -> {latest_hash}")
                    all_files_valid = False
            else:
                self.logger.info("No version information found - need to verify files")
                all_files_valid = False

            # If model loading failed, check individual files
            if not all_files_valid:
                for filename in files_to_download:
                    dest_path = model_dir / filename
                    if not dest_path.exists():
                        self.logger.info(f"File {filename} missing - will download")
                        missing_or_invalid.append(filename)
                        continue
                    
                    # Verify JSON files are valid
                    if filename.endswith('.json'):
                        try:
                            with open(dest_path, 'r', encoding='utf-8') as f:
                                json.load(f)
                            self.logger.info(f"Existing {filename} is valid")
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            self.logger.warning(f"Existing {filename} is corrupt - will redownload")
                            missing_or_invalid.append(filename)
                            dest_path.unlink(missing_ok=True)
                    else:
                        # For non-JSON files, check if they're empty or too small
                        file_size = dest_path.stat().st_size
                        if filename == "pytorch_model.bin":
                            if file_size < 100_000_000:  # 100MB
                                self.logger.warning(f"Existing {filename} appears incomplete ({file_size / 1_000_000:.1f} MB) - will redownload")
                                missing_or_invalid.append(filename)
                                dest_path.unlink(missing_ok=True)
                            else:
                                self.logger.info(f"Existing model file {filename} is valid and up-to-date ({file_size / 1_000_000:.1f} MB)")
                        elif filename == "spm.model":
                            if file_size < 1000:  # 1KB
                                self.logger.warning(f"Existing {filename} appears corrupt - will redownload")
                                missing_or_invalid.append(filename)
                                dest_path.unlink(missing_ok=True)
                            else:
                                self.logger.info(f"Existing {filename} is valid ({file_size / 1_000:.1f} KB)")
                        else:
                            self.logger.info(f"Existing {filename} appears valid ({file_size} bytes)")

            # Only download files that are missing or invalid
            for filename in missing_or_invalid:
                file_url = f"https://huggingface.co/{model_name}/resolve/main/{filename}"
                dest_path = model_dir / filename
                
                # Try download up to 3 times
                for attempt in range(3):
                    try:
                        success = self.download_manager.download_with_progress(file_url, dest_path)
                        if not success:
                            if filename == "model.safetensors":
                                self.logger.error("Failed to download safetensors model")
                                # Try to get detailed error info
                                try:
                                    response = requests.get(file_url, stream=True)
                                    self.logger.error(f"Download failed with status {response.status_code}")
                                    if response.status_code == 403:
                                        self.logger.error("Access denied - this may be a rate limiting issue")
                                    elif response.status_code == 404:
                                        self.logger.error("File not found - model may not have safetensors format")
                                    else:
                                        self.logger.error(f"Server response: {response.text[:500]}")
                                except Exception as e:
                                    self.logger.error(f"Error checking model status: {e}")
                                return False
                            continue

                        # Verify file size after download for pytorch_model.bin
                        if filename == "pytorch_model.bin":
                            file_size = dest_path.stat().st_size
                            if file_size < 100_000_000:  # 100MB minimum size
                                self.logger.warning(f"Downloaded {filename} is too small ({file_size / 1_000_000:.1f} MB)")
                                dest_path.unlink(missing_ok=True)
                                if attempt < 2:  # Only retry if we have attempts left
                                    self.logger.info(f"Retrying download of {filename} (attempt {attempt + 2}/3)")
                                    continue
                                else:
                                    raise ValueError(f"Failed to download valid {filename} after 3 attempts")
                            else:
                                self.logger.info(f"Successfully downloaded {filename} ({file_size / 1_000_000:.1f} MB)")
                                self.logger.info(f"PyTorch model file is up to date")

                        # Verify JSON files immediately after download
                        if filename.endswith('.json'):
                            try:
                                with open(dest_path, 'r', encoding='utf-8') as f:
                                    json.load(f)
                            except json.JSONDecodeError:
                                self.logger.warning(f"Corrupted JSON download for {filename}, retrying...")
                                if attempt < 2:  # Only delete and retry if we have attempts left
                                    dest_path.unlink(missing_ok=True)
                                    continue
                                else:
                                    raise ValueError(f"Failed to download valid {filename} after 3 attempts")
                        break  # Successful download and verification
                        
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 404:
                            if filename == "pytorch_model.bin":
                                self.logger.error(f"Required file {filename} not found on server")
                                return False
                            self.logger.info(f"Optional file {filename} not found on server - skipping")
                        else:
                            self.logger.error(f"HTTP error downloading {filename}: {str(e)}")
                            if filename == "pytorch_model.bin":
                                return False
                    except Exception as e:
                        self.logger.error(f"Error downloading {filename}: {str(e)}")
                        if filename == "pytorch_model.bin":
                            return False
            
            try:
                # Verify all required files exist and are valid
                required_files = ["config.json", "tokenizer_config.json", "spm.model"]
                model_files = ["model.safetensors", "pytorch_model.bin"]
                
                self.logger.info(f"Verifying required files for {model_name}")
                
                # Check required config files
                for file in required_files:
                    file_path = model_dir / file
                    if not file_path.exists():
                        self.logger.error(f"Required file {file} is missing")
                        raise FileNotFoundError(f"Required file {file} is missing")
                    if file.endswith('.json'):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                json.load(f)
                            self.logger.info(f"Verified JSON file: {file}")
                        except json.JSONDecodeError as je:
                            self.logger.error(f"Invalid JSON in {file}: {str(je)}")
                            raise ValueError(f"Invalid JSON in {file}: {str(je)}")
                    else:
                        self.logger.info(f"Verified file exists: {file}")
                
                # Check model files - try safetensors first, fall back to PyTorch
                safetensors_path = model_dir / "model.safetensors"
                pytorch_path = model_dir / "pytorch_model.bin"
                
                if safetensors_path.exists():
                    file_size = safetensors_path.stat().st_size
                    if file_size < 100_000_000:  # 100MB minimum size
                        raise ValueError(f"Safetensors file too small: {file_size / 1_000_000:.1f} MB")
                    self.logger.info(f"Verified model.safetensors size: {file_size / 1_000_000:.1f} MB")
                elif pytorch_path.exists():
                    file_size = pytorch_path.stat().st_size
                    if file_size < 100_000_000:  # 100MB minimum size
                        raise ValueError(f"PyTorch model file too small: {file_size / 1_000_000:.1f} MB")
                    self.logger.info(f"Verified pytorch_model.bin size: {file_size / 1_000_000:.1f} MB")
                else:
                    raise FileNotFoundError("No valid model file found (tried safetensors and PyTorch formats)")

                # Attempt to load the tokenizer, but only from local files since we just downloaded them
                self.logger.info(f"Verifying tokenizer for {model_name}")
                try:
                    try:
                        import sentencepiece
                    except ImportError:
                        self.logger.error("sentencepiece library is required for DeBERTa tokenizer but not installed")
                        raise ImportError("Please install sentencepiece: pip install sentencepiece")
                        
                    tokenizer = AutoTokenizer.from_pretrained(
                        str(model_dir),
                        local_files_only=True,  # Force local files only
                        trust_remote_code=True,
                        use_fast=False,  # Use slow tokenizer to avoid conversion issues
                        tokenizer_type='deberta-v2',
                        model_max_length=512  # Add explicit max length
                    )
                except Exception as e:
                    self.logger.error(f"Failed to load tokenizer from local files: {e}")
                    raise
                
                # Test tokenizer with multiple inputs
                self.logger.info(f"Testing tokenizer for {model_name}")
                test_texts = [
                    "Test sentence",
                    "A longer test sentence with more words to encode",
                    "Special characters: !@#$%^&*()"
                ]
                for test_text in test_texts:
                    test_result = tokenizer.encode(test_text, add_special_tokens=True)
                    if not test_result or len(test_result) == 0:
                        self.logger.error(f"Tokenizer verification failed for: {test_text}")
                        raise ValueError("Tokenizer verification failed - empty output")
                    self.logger.info(f"Tokenizer test successful for: {test_text}")
                    self.logger.debug(f"Encoded tokens: {test_result[:10]}...")

                # Get version info for tracking
                api = HfApi()
                model_info = api.model_info(model_name)
                current_hash = model_info.sha
                
                # Extract version from tags if available
                version_tag = next((tag for tag in model_info.tags if tag.startswith('v')), None)
                
                # If we get here, everything worked
                self.logger.info(f"All verification steps passed for {model_name}")
                self.versions[f"hf_{model_name}"] = current_hash
                if version_tag:
                    self.versions[f"hf_{model_name}_version"] = version_tag
                    self.logger.info(f"Model version: {version_tag}")
                self.versions_file.parent.mkdir(parents=True, exist_ok=True)
                self.versions_file.write_text(
                    json.dumps(self.versions, indent=2, ensure_ascii=False),
                    encoding='utf-8'
                )
                self.logger.info(f"Successfully verified {model_name}")
                return True

            except Exception as e:
                self.logger.error(f"Model verification failed for {model_name}: {str(e)}")
                self.logger.info(f"Cleaning up failed installation for {model_name}")
                # Remove any partial version info
                self.versions.pop(f"hf_{model_name}", None)
                self.versions.pop(f"hf_{model_name}_version", None)
                # Clean up failed download
                if model_dir.exists():
                    import shutil
                    shutil.rmtree(model_dir)
                return False

        except Exception as e:
            self.logger.error(f"Failed to install {model_name}: {str(e)}")
            return False

    def _check_spacy_model_update(self, model_name: str) -> bool:
        """Check if a spaCy model has updates available."""
        try:
            from packaging import version
            
            # Get current version from installed model
            current_info = spacy.info(model_name)
            current_version = version.parse(current_info['version'])
            
            # Get latest version from PyPI
            package_name = model_name.replace('-', '_')
            import subprocess
            import re
            result = subprocess.run(['pip', 'index', 'versions', package_name], 
                                 capture_output=True, text=True)
            # Look for version numbers in the output
            version_pattern = r'\b\d+\.\d+\.\d+\b'
            versions = re.findall(version_pattern, result.stdout)
            if versions:
                # First match should be the latest version
                latest_version = version.parse(versions[0])
                
                self.logger.info(f"SpaCy model versions - Current: {current_version}, Latest: {latest_version}")
                return latest_version > current_version
            return False
        except Exception as e:
            self.logger.warning(f"Failed to check updates for {model_name}: {e}")
            return False

    def install_spacy_model(self, model_name: str, force_update: bool = False) -> bool:
        """Install a spaCy model."""
        try:
            needs_update = force_update
            if not force_update and f"spacy_{model_name}" in self.versions:
                needs_update = self._check_spacy_model_update(model_name)
                if not needs_update:
                    self.logger.info(f"Model {model_name} is up to date")
                    return True
                self.logger.info(f"Updates available for {model_name}")

            self.logger.info(f"Installing spaCy model: {model_name}")
            
            try:
                # First try to load the model if it exists
                try:
                    # Configure model loading
                    self._configure_model_loading()
                
                    nlp = spacy.load(model_name)
                    # Test the model with a simple sentence
                    test_doc = nlp("Test sentence")
                    if test_doc is not None:
                        self.logger.info(f"Existing spaCy model {model_name} is valid - skipping download")
                        self.versions[f"spacy_{model_name}"] = "latest"
                        self.versions_file.write_text(
                            json.dumps(self.versions, indent=2, ensure_ascii=False),
                            encoding='utf-8'
                        )
                        return True
                except Exception as e:
                    self.logger.info(f"Need to download spaCy model {model_name}: {e}")
                
                # If loading failed, download the model
                if not spacy.util.is_package(model_name):
                    self.logger.info(f"Downloading spaCy model {model_name}")
                    spacy.cli.download(model_name)
                
                # Verify the downloaded model with secure loading
                self.logger.info("Verifying model format and loading capabilities...")
                
                # Configure PyTorch for secure model loading
                import torch
                import torch.serialization
                torch.set_grad_enabled(False)
                
                # Configure default device and dtype
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                torch.set_default_device(device)
                torch.set_default_dtype(torch.float32)
                
                # Add safe globals for numpy arrays
                safe_modules = ['numpy', 'numpy.core.multiarray', 'numpy.core.numeric', 'numpy.core.fromnumeric']
                for module in safe_modules:
                    torch.serialization.add_safe_globals(module)
                
                nlp = spacy.load(model_name)
                test_doc = nlp("Test sentence")
                if test_doc is not None:
                    self.versions[f"spacy_{model_name}"] = "latest"
                    self.versions_file.write_text(
                        json.dumps(self.versions, indent=2, ensure_ascii=False),
                        encoding='utf-8'
                    )
                    return True
                return False
                
            except Exception as e:
                self.logger.error(f"SpaCy model {model_name} installation failed: {str(e)}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to install spaCy model {model_name}: {str(e)}")
            return False
    def initialize(self):
        """Initialize the model installer and verify system requirements."""
        # Check system and get recommendations
        self.recommended_models = self.sys_check.get_recommended_models()

