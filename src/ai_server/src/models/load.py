import time
import torch
import onnxruntime as ort

from enum import Enum
from pathlib import Path
from cachetools import TTLCache
from dataclasses import dataclass
from threading import Thread, Lock
from typing import Optional, Any, Union, List
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelType(Enum):
    TRANSFORMERS = "transformers"
    ONNX = "onnx"


@dataclass
class CachedModel:
    tokenizer: Optional[Any]  # May be None for ONNX models
    model: Any
    last_used: float
    model_type: ModelType


class ModelCacheManager:
    """
    A cache manager for transformer and ONNX models that automatically offloads models
    from memory when they haven't been used for a specified duration.
    """

    def __init__(
        self, cache_size: int = 10, ttl_seconds: int = 600, cleanup_interval: int = 60, force_cpu: bool = False
    ):
        """
        Initialize the model cache manager.

        Args:
            cache_size: Maximum number of models to keep in cache
            ttl_seconds: Time to live in seconds before a model is considered for cleanup
            cleanup_interval: How often to run the cleanup process in seconds
            force_cpu: If True, will use CPU even if GPU is available
        """
        self.cache: TTLCache = TTLCache(maxsize=cache_size, ttl=ttl_seconds)
        self.cleanup_interval = cleanup_interval
        self.force_cpu = force_cpu
        self.lock = Lock()
        self.device = self._setup_device()
        self.onnx_providers = self._setup_onnx_providers()

        print(f"Using device: {self.device}")
        print(f"ONNX providers: {self.onnx_providers}")

        self._start_cleanup_thread()

    def _setup_device(self) -> torch.device:
        """Set up the PyTorch device based on availability."""
        if self.force_cpu:
            print("Forcing CPU usage...")
            return torch.device("cpu")

        if torch.cuda.is_available():
            print("CUDA is available. Using GPU...")
            return torch.device("cuda")
        try:
            # Try to use MPS for Mac M* series
            if torch.backends.mps.is_available():
                print("MPS is available. Using MPS...")
                return torch.device("mps")
        except:
            pass

        print("CUDA is not available. Using CPU...")
        return torch.device("cpu")

    def _setup_onnx_providers(self) -> List[str]:
        """Set up ONNX Runtime providers based on available hardware."""
        available_providers = ort.get_available_providers()
        selected_providers = []

        if not self.force_cpu:
            # Try CUDA first
            if "CUDAExecutionProvider" in available_providers:
                selected_providers.append("CUDAExecutionProvider")
            # Try DirectML (Windows)
            elif "DmlExecutionProvider" in available_providers:
                selected_providers.append("DmlExecutionProvider")
            # Try TensorRT
            elif "TensorrtExecutionProvider" in available_providers:
                selected_providers.append("TensorrtExecutionProvider")

        # Use CPU as fallback
        selected_providers.append("CPUExecutionProvider")
        return selected_providers

    def load_model(
        self, model_path: Union[str, Path], model_type: ModelType, tokenizer_path: Optional[str] = None
    ) -> tuple[Optional[Any], Any]:
        """
        Load a model and its tokenizer (if applicable) from the cache or from disk.

        Args:
            model_path: Path to the model (HuggingFace model name or path to ONNX file)
            model_type: Type of model to load (TRANSFORMERS or ONNX)
            tokenizer_path: Optional separate tokenizer path/name for ONNX models

        Returns:
            A tuple of (tokenizer, model). Tokenizer may be None for ONNX models
            if no tokenizer_path is provided
        """
        model_key = str(model_path)

        with self.lock:
            cached = self._get_from_cache(model_key)
            if cached:
                return cached.tokenizer, cached.model

            return self._load_and_cache_model(model_path, model_type, tokenizer_path)

    def _get_from_cache(self, model_key: str) -> Optional[CachedModel]:
        """Get a model from cache and update its last used timestamp."""
        if model_key in self.cache:
            cached_model = self.cache[model_key]
            cached_model.last_used = time.time()
            return cached_model
        return None

    def _load_and_cache_model(
        self, model_path: Union[str, Path], model_type: ModelType, tokenizer_path: Optional[str] = None
    ) -> tuple[Optional[Any], Any]:
        """Load a model from disk and add it to the cache."""
        print(f"Loading {model_type.value} model from {model_path}...")

        tokenizer = None
        model = None

        if model_type == ModelType.TRANSFORMERS:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForCausalLM.from_pretrained(str(model_path)).to(self.device)
        else:  # ONNX
            # Load tokenizer if provided
            if tokenizer_path:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            # Create ONNX Runtime session with optimizations
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Enable parallel execution
            session_options.enable_cpu_mem_arena = True
            session_options.enable_mem_pattern = True
            session_options.intra_op_num_threads = 0  # Let ONNX Runtime decide

            model = ort.InferenceSession(
                str(model_path),
                providers=self.onnx_providers,
                provider_options=[{} for _ in self.onnx_providers],
                sess_options=session_options,
            )

        cached_model = CachedModel(tokenizer=tokenizer, model=model, last_used=time.time(), model_type=model_type)
        self.cache[str(model_path)] = cached_model

        return tokenizer, model

    def _cleanup_cache(self) -> None:
        """Remove expired models from the cache."""
        while True:
            with self.lock:
                now = time.time()
                keys_to_delete = [key for key, value in self.cache.items() if now - value.last_used > self.cache.ttl]

                for key in keys_to_delete:
                    model_data = self.cache[key]
                    print(f"Removing model {key} from cache...")

                    # Proper cleanup for CUDA tensors
                    if model_data.model_type == ModelType.TRANSFORMERS:
                        model_data.model.cpu()
                        if hasattr(model_data.model, "cuda_graphs"):
                            del model_data.model.cuda_graphs

                    del self.cache[key]

                    if self.device.type == "cuda":
                        # Force CUDA memory cleanup
                        torch.cuda.empty_cache()

            time.sleep(self.cleanup_interval)

    def _start_cleanup_thread(self) -> None:
        """Start the background cleanup thread."""
        cleanup_thread = Thread(target=self._cleanup_cache, daemon=True, name="ModelCacheCleanup")
        cleanup_thread.start()
