import time
import torch
from vllm import LLM
from typing import Optional
from cachetools import TTLCache
from dataclasses import dataclass
from threading import Thread, Lock


@dataclass
class CachedVLLMModel:
    model: LLM
    last_used: float


class VLLMCacheManager:
    def __init__(
        self,
        cache_size: int = 2,
        ttl_seconds: int = 600,
        cleanup_interval: int = 60,
        gpu_memory_utilization: float = 0.75,
        max_parallel_models: int = 1,
    ):
        self.cache: TTLCache = TTLCache(maxsize=cache_size, ttl=ttl_seconds)
        self.cleanup_interval = cleanup_interval
        self.lock = Lock()
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_parallel_models = max_parallel_models

        self.total_gpu_memory = torch.cuda.get_device_properties(0).total_memory

        print(f"Total GPU memory: {self.total_gpu_memory / 1024**3:.2f} GB")
        print(f"GPU memory utilization: {self.gpu_memory_utilization * 100}%")

        self._start_cleanup_thread()

    def get_model(self, model_path: str, tensor_parallel_size: int = 1) -> LLM:
        with self.lock:
            cached = self._get_from_cache(model_path)
            if cached:
                return cached.model

            # Always ensure we have room for the new model
            if len(self.cache) >= self.max_parallel_models:
                # Remove all existing models to free up memory
                print("Clearing cache to make room for new model...")
                self._clear_cache()

            return self._load_and_cache_model(model_path, tensor_parallel_size)

    def _get_from_cache(self, model_key: str) -> Optional[CachedVLLMModel]:
        if model_key in self.cache:
            cached_model = self.cache[model_key]
            cached_model.last_used = time.time()
            return cached_model
        return None

    def _load_and_cache_model(self, model_path: str, tensor_parallel_size: int) -> LLM:
        print(f"Loading vLLM model from {model_path}...")

        try:
            model = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=2048,
                enforce_eager=True,
                max_num_batched_tokens=4096,
                max_num_seqs=256,
            )

            cached_model = CachedVLLMModel(model=model, last_used=time.time())
            self.cache[model_path] = cached_model

            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # Ensure memory is cleared even if model loading fails
            torch.cuda.empty_cache()
            raise

    def _clear_cache(self) -> None:
        """Remove all models from cache and clean up GPU memory."""
        print("Clearing entire model cache...")
        self.cache.clear()
        torch.cuda.empty_cache()

    def _cleanup_cache(self) -> None:
        while True:
            with self.lock:
                now = time.time()
                keys_to_delete = [key for key, value in self.cache.items() if now - value.last_used > self.cache.ttl]

                if keys_to_delete:
                    print(f"Cleaning up {len(keys_to_delete)} expired models...")
                    self._clear_cache()

            time.sleep(self.cleanup_interval)

    def _start_cleanup_thread(self) -> None:
        cleanup_thread = Thread(target=self._cleanup_cache, daemon=True, name="VLLMCacheCleanup")
        cleanup_thread.start()
