import logging
import psutil
import subprocess
import torch


class SystemResources:
    """Analyzes and reports system capabilities for model deployment."""
    
    logger: logging.Logger
    
    def __init__(self):
        """Initialize SystemResources with logger."""
        self.logger = logging.getLogger(__name__)

    def __init__(self):
        """Initialize SystemResources with logger and cached results."""
        self.logger = logging.getLogger(__name__)
        self._cached_resources = None

    def check_system(self) -> dict[str, any]:
        """Check available system resources and return capabilities."""
        if self._cached_resources is not None:
            return self._cached_resources

        # Check both PyTorch CUDA and nvidia-smi
        pytorch_cuda = torch.cuda.is_available()
        nvidia_smi_available = False
        cuda_version = None
        cudnn_version = None
        
        try:
            nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if nvidia_smi.returncode == 0:
                nvidia_smi_available = True
                # Extract CUDA version from nvidia-smi output
                import re
                cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', nvidia_smi.stdout)
                if cuda_match:
                    cuda_version = cuda_match.group(1)
        except Exception:
            pass

        # GPU is available if either PyTorch CUDA or nvidia-smi works
        cuda_available = pytorch_cuda or nvidia_smi_available
        
        if pytorch_cuda:
            cuda_version = cuda_version or torch.version.cuda
            if torch.backends.cudnn.is_available():
                cudnn_version = torch.backends.cudnn.version()
        
        # Initialize with basic system info
        self._cached_resources = {
            'gpu_available': cuda_available,
            'gpu_memory': 0,  # Will be updated later if GPU is available
            'cpu_count': psutil.cpu_count(logical=False),
            'ram_gb': psutil.virtual_memory().total / (1024 ** 3),
            'free_disk_gb': psutil.disk_usage('/').free / (1024 ** 3),
            'cuda_version': cuda_version,
            'torch_version': torch.__version__,
            'cuda_arch_list': None,
            'cuda_toolkit': None,
            'cudnn_version': cudnn_version
        }

        # Get CUDA capabilities for all GPUs if available
        if self._cached_resources['gpu_available']:
            total_gpu_memory = 0
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_memory = props.total_memory / (1024 ** 3)  # Convert to GB
                total_gpu_memory += gpu_memory
                self.logger.info(f"Found GPU {i}: {props.name} ({gpu_memory:.1f} GB)")
                
                # Store compute capability for the first GPU
                if i == 0:
                    self._cached_resources['cuda_arch_list'] = f"{props.major}.{props.minor}"
                    
            self._cached_resources['gpu_memory'] = total_gpu_memory
            
            # Try to get CUDA toolkit version from nvcc
            try:
                nvcc_output = subprocess.check_output(['nvcc', '--version'], text=True)
                self._cached_resources['cuda_toolkit'] = nvcc_output.strip()
                # Update CUDA version if not already set
                if not self._cached_resources['cuda_version']:
                    import re
                    cuda_match = re.search(r'release (\d+\.\d+)', nvcc_output)
                    if cuda_match:
                        self._cached_resources['cuda_version'] = cuda_match.group(1)
                
                # Get GPU info from nvidia-smi even if PyTorch is CPU-only
                nvidia_smi = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', '--format=csv,noheader'], 
                                         capture_output=True, text=True)
                if nvidia_smi.returncode == 0:
                    gpu_info = nvidia_smi.stdout.strip().split(',')
                    if len(gpu_info) >= 3:
                        self.logger.info(f"GPU detected: {gpu_info[0].strip()}")
                        # Convert memory from MiB to GB
                        memory_mib = float(gpu_info[1].strip().split()[0])
                        self._cached_resources['gpu_memory'] = memory_mib / 1024
                        self._cached_resources['cuda_arch_list'] = gpu_info[2].strip()
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

        # Check NVIDIA driver and tools regardless of PyTorch CUDA status
        try:
            # Check NVIDIA driver
            nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if nvidia_smi.returncode == 0:
                self.logger.info("NVIDIA driver found:")
                self.logger.info(nvidia_smi.stdout.strip())
                # Parse driver version from nvidia-smi output
                import re
                driver_match = re.search(r'Driver Version: (\d+\.\d+\.\d+)', nvidia_smi.stdout)
                if driver_match:
                    self.logger.info(f"NVIDIA driver version: {driver_match.group(1)}")
            
            # Check CUDA toolkit
            nvcc = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if nvcc.returncode == 0:
                cuda_toolkit = nvcc.stdout.strip()
                self._cached_resources['cuda_toolkit'] = cuda_toolkit
                self.logger.info(f"CUDA toolkit found:\n{cuda_toolkit}")
                # Parse CUDA version from nvcc output
                cuda_match = re.search(r'release (\d+\.\d+)', nvcc.stdout)
                if cuda_match:
                    self.logger.info(f"CUDA toolkit version: {cuda_match.group(1)}")
        except FileNotFoundError:
            self.logger.warning("NVIDIA tools not found in PATH")
        except Exception as e:
            self.logger.warning(f"Error checking CUDA tools: {e}")
        if torch.cuda.is_available():
            self._cached_resources['gpu_available'] = True
            self._cached_resources['cuda_version'] = torch.version.cuda
            self._cached_resources['cudnn_version'] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None

            # Check each GPU and accumulate total memory
            self._cached_resources['gpus'] = []
            total_gpu_memory = 0
                
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_memory_gb = props.total_memory / (1024 ** 3)
                total_gpu_memory += gpu_memory_gb
                    
                gpu_info = {
                    'name': props.name,
                    'memory_gb': gpu_memory_gb,
                    'compute_capability': f"{props.major}.{props.minor}",
                    'max_threads_per_block': props.max_threads_per_block,
                    'max_shared_memory_per_block': props.max_shared_memory_per_block,
                    'multi_processor_count': props.multi_processor_count
                }
                self._cached_resources['gpus'].append(gpu_info)
                self.logger.info(f"Found GPU {i}: {props.name}")
                self.logger.info(f"  Memory: {gpu_memory_gb:.1f} GB")
                self.logger.info(f"  Compute capability: {gpu_info['compute_capability']}")
                self.logger.info(f"  CUDA cores: {props.multi_processor_count * 64}")  # Approximate
                
            # Set total GPU memory
            self._cached_resources['gpu_memory'] = total_gpu_memory
            self.logger.info(f"Total GPU memory: {total_gpu_memory:.1f} GB")

        self.logger.debug(f"System resources: {self._cached_resources}")  # Change to debug level for internal use
        return self._cached_resources

    def get_recommended_models(self) -> dict[str, str]:
        """Determine which models can be supported by the system."""
        resources = self.check_system()

        # Model requirements (in GB)
        SPACY_TRF_RAM = 16  # Increased for CPU-only systems
        SPACY_LG_RAM = 4
        DEBERTA_RAM = 12    # Increased for CPU-only systems

        models = {
            'spacy': 'en_core_web_lg',  # Default to large
            'hf': 'microsoft/deberta-v3-base'  # Default DeBERTa model
        }

        # SpaCy model selection - prefer non-transformer models for CPU-only systems
        if resources['gpu_available'] and resources['ram_gb'] >= SPACY_TRF_RAM:
            models['spacy'] = 'en_core_web_trf'
        elif resources['ram_gb'] >= SPACY_LG_RAM:
            models['spacy'] = 'en_core_web_lg'
        else:
            models['spacy'] = 'en_core_web_md'

        # DeBERTa model selection - more conservative for CPU-only systems
        if resources['gpu_available'] and resources['gpu_memory'] >= 16:
            models['hf'] = 'microsoft/deberta-v3-large'
        elif resources['gpu_available'] and resources['ram_gb'] >= DEBERTA_RAM:
            models['hf'] = 'microsoft/deberta-v3-base'
        else:
            models['hf'] = 'microsoft/deberta-v3-small'

        self.logger.info(f"Recommended models: {models}")
        return models
