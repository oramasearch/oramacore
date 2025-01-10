from vllm import SamplingParams
from typing import Dict, Any, Optional

from src.models.cache import VLLMCacheManager


class ModelsManager:
    def __init__(self, config):
        self.config = config
        self.current_model_key = None

        self.cache = VLLMCacheManager(
            cache_size=1,
            ttl_seconds=600,
            gpu_memory_utilization=0.85,
            max_parallel_models=1,
        )

        self.model_configs = {
            key: value
            for key, value in vars(self.config.LLMs).items()
            if key not in ["__dict__", "__weakref__", "__doc__"]
        }

        self.models = {}
        self.sampling_params = {}

    def _load_model(self, model_key: str):
        """Load a vLLM model."""
        if model_key not in self.model_configs:
            raise ValueError(f"Invalid model key: {model_key}")

        model_config = self.model_configs[model_key]
        if model_config is None:
            raise ValueError(f"Configuration for model {model_key} is None")

        # If we already have this model loaded, return it
        if model_key == self.current_model_key and model_key in self.models:
            return self.models[model_key]

        try:
            model = self.cache.get_model(
                model_path=model_config.id, tensor_parallel_size=model_config.tensor_parallel_size
            )

            sampling_params = SamplingParams(
                temperature=model_config.sampling_params.temperature,
                top_p=model_config.sampling_params.top_p,
                max_tokens=model_config.sampling_params.max_tokens,
            )

            model_data = {"model": model, "config": {"sampling_params": sampling_params}}
            self.models[model_key] = model_data
            self.current_model_key = model_key
            return model_data

        except Exception as e:
            print(f"Error loading model {model_key}: {e}")
            return None

    def get_model(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get or load a model by its key."""
        # Check if we already have this model and it's the current one
        if model_key == self.current_model_key and model_key in self.models:
            return self.models[model_key]

        # Otherwise, load or reload the model
        return self._load_model(model_key)

    def generate_text(self, model_key: str, prompt: str) -> str:
        """Generate text using the specified model."""
        model_data = self.get_model(model_key)
        if not model_data:
            raise RuntimeError(f"Model {model_key} not loaded")

        model = model_data["model"]
        sampling_params = model_data["config"]["sampling_params"]

        try:
            outputs = model.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            print(f"Error generating text with {model_key}: {e}")
            raise

    def process_batch(self, model_key: str, prompts: list[str]) -> list[str]:
        """
        Process a batch of prompts using the specified model.
        """
        model_data = self.get_model(model_key)
        if not model_data:
            raise RuntimeError(f"Model {model_key} not loaded")

        model = model_data["model"]
        sampling_params = model_data["config"]["sampling_params"]

        try:
            outputs = model.generate(prompts, sampling_params)
            return [output.outputs[0].text.strip() for output in outputs]
        except Exception as e:
            print(f"Error processing batch with {model_key}: {e}")
            raise

    def cleanup(self):
        """
        Clean up all loaded models and free resources.
        """
        self.models.clear()
