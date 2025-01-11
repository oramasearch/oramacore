import torch
import logging
import threading
from vllm import SamplingParams, LLM
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ModelsManager:
    def __init__(self, config):
        self.config = config
        # Convert enum names to lowercase to match protobuf enum values
        self.model_configs = {
            key.lower(): value
            for key, value in vars(self.config.LLMs).items()
            if key not in ["__dict__", "__weakref__", "__doc__"]
        }
        self._models: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        logger.info(f"Available models: {list(self.model_configs.keys())}")

        logger.info("Pre-loading all models...")
        for model_key in self.model_configs:
            self._preload_model(model_key)

    def _preload_model(self, model_key: str) -> None:
        """Preload a model during initialization."""
        if model_key not in self.model_configs:
            logger.info(f"Model key {model_key} not found in configs")
            return

        model_config = self.model_configs[model_key]
        if model_config is None:
            logger.info(f"Configuration for model {model_key} is None")
            return

        try:
            logger.info(f"Loading model {model_config.id}...")

            model = LLM(
                model=model_config.id,
                tensor_parallel_size=model_config.tensor_parallel_size,
                gpu_memory_utilization=0.85,
                max_model_len=2048,
                enforce_eager=True,
                max_num_batched_tokens=4096,
                max_num_seqs=256,
                trust_remote_code=True,
            )

            sampling_params = SamplingParams(
                temperature=model_config.sampling_params.temperature,
                top_p=model_config.sampling_params.top_p,
                max_tokens=model_config.sampling_params.max_tokens,
            )

            model_data = {"model": model, "config": {"sampling_params": sampling_params}}

            # Run a warmup prompt
            warmup_prompt = "test"
            _ = model.generate([warmup_prompt], sampling_params)

            self._models[model_key] = model_data
            logger.info(f"Successfully loaded model {model_config.id}")

        except Exception as e:
            logger.info(f"Error preloading model {model_key}: {e}")
            if model_key in self._models:
                del self._models[model_key]

    def get_model(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get a preloaded model."""
        model_key = model_key.lower()  # Convert to lowercase for consistency
        with self._lock:
            if model_key not in self._models:
                logger.info(f"Requested model {model_key} not found. Available models: {list(self._models.keys())}")
                raise RuntimeError(f"Model {model_key} is not loaded")
            return self._models[model_key]

    def generate_text(self, model_key: str, prompt: str) -> str:
        """Generate text using a preloaded model."""
        try:
            model_data = self.get_model(model_key)
            if not model_data:
                raise RuntimeError(f"Model {model_key} is not available")

            model = model_data["model"]
            sampling_params = model_data["config"]["sampling_params"]

            outputs = model.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating text with {model_key}: {e}")
            logger.error(f"Currently loaded models: {list(self._models.keys())}")
            raise

    def process_batch(self, model_key: str, prompts: list[str]) -> list[str]:
        """Process a batch using a preloaded model."""
        model_data = self.get_model(model_key)
        if not model_data:
            raise RuntimeError(f"Model {model_key} is not available")

        model = model_data["model"]
        sampling_params = model_data["config"]["sampling_params"]

        try:
            outputs = model.generate(prompts, sampling_params)
            return [output.outputs[0].text.strip() for output in outputs]
        except Exception as e:
            logger.error(f"Error processing batch with {model_key}: {e}")
            raise

    def cleanup(self):
        """Clean up models and free GPU memory."""
        with self._lock:
            self._models.clear()
            torch.cuda.empty_cache()
