import torch
import logging
import threading
from json_repair import repair_json
from vllm import SamplingParams, LLM
from typing import Dict, Any, Optional, Set

from src.models.prompts import PROMPT_TEMPLATES

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
        self._models: Dict[str, Dict[str, Any]] = {}  # Stores unique model instances by model ID
        self._model_refs: Dict[str, str] = {}  # Maps model_key to model_id
        self._lock = threading.Lock()

        logger.info(f"Available models: {list(self.model_configs.keys())}")

        # Get unique model IDs
        unique_models = self._get_unique_model_ids()
        logger.info(f"Unique models to load: {unique_models}")

        # Preload unique models
        for model_id in unique_models:
            self._preload_unique_model(model_id)

        # Set up model references and sampling params
        self._setup_model_references()

    def _get_unique_model_ids(self) -> Set[str]:
        """Get set of unique model IDs from configurations."""
        return {config.id for config in self.model_configs.values() if config is not None}

    def _preload_unique_model(self, model_id: str) -> None:
        """Preload a unique model instance."""
        try:
            logger.info(f"Loading model {model_id}...")

            model = LLM(
                model=model_id,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,
                max_model_len=2048,
                enforce_eager=True,
                max_num_batched_tokens=4096,
                max_num_seqs=256,
                trust_remote_code=True,
            )

            # Run a warmup prompt
            warmup_prompt = "hello"
            _ = model.generate([warmup_prompt], SamplingParams(temperature=0.2, top_p=0.95, max_tokens=5))

            self._models[model_id] = {"model": model, "configs": {}}
            logger.info(f"Successfully loaded model {model_id}")

        except Exception as e:
            logger.error(f"Error preloading model {model_id}: {e}")
            if model_id in self._models:
                del self._models[model_id]
            raise

    def _setup_model_references(self) -> None:
        """Set up model references and sampling parameters for each model key."""
        for model_key, config in self.model_configs.items():
            if config is None:
                continue

            model_id = config.id
            if model_id not in self._models:
                logger.warning(f"Model {model_id} not loaded, skipping {model_key} setup")
                continue

            # Store the reference
            self._model_refs[model_key] = model_id

            # Store the sampling parameters for this configuration
            sampling_params = SamplingParams(
                temperature=config.sampling_params.temperature,
                top_p=config.sampling_params.top_p,
                max_tokens=config.sampling_params.max_tokens,
            )
            self._models[model_id]["configs"][model_key] = {"sampling_params": sampling_params}

    def get_model(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get a model and its configuration by model_key."""
        model_key = model_key.lower()
        with self._lock:
            if model_key not in self._model_refs:
                logger.error(f"Model key {model_key} not found in references")
                return None

            model_id = self._model_refs[model_key]
            model_data = self._models.get(model_id)

            if not model_data:
                logger.error(f"Model {model_id} not found in loaded models")
                return None

            return {"model": model_data["model"], "config": model_data["configs"][model_key]}

    def generate_text(self, model_key: str, prompt: str) -> str:
        """Generate text using a model."""
        try:
            model_data = self.get_model(model_key)
            if not model_data:
                raise RuntimeError(f"Model {model_key} is not available")

            model = model_data["model"]
            sampling_params = model_data["config"]["sampling_params"]

            conversation_history = [
                {"role": "system", "content": PROMPT_TEMPLATES[f"{model_key}:system"]},
                {"role": "user", "content": PROMPT_TEMPLATES[f"{model_key}:user"](prompt)},
            ]

            outputs = model.chat(conversation_history, sampling_params=sampling_params)

            return repair_json(outputs[0].outputs[0].text.strip())

        except Exception as e:
            logger.error(f"Error generating text with {model_key}: {e}")
            logger.error(f"Currently loaded models: {list(self._models.keys())}")
            raise

    def generate_text_stream(self, model_key: str, prompt: str):
        """Generate text using a model with streaming output."""
        try:
            model_data = self.get_model(model_key)
            if not model_data:
                raise RuntimeError(f"Model {model_key} is not available")

            model = model_data["model"]
            sampling_params = model_data["config"]["sampling_params"]

            conversation_history = [
                {"role": "system", "content": PROMPT_TEMPLATES[f"{model_key}:system"]},
                {"role": "user", "content": PROMPT_TEMPLATES[f"{model_key}:user"](prompt)},
            ]

            # Format chat messages into prompt using Assistant/User format
            formatted_prompt = ""
            for msg in conversation_history:
                role_prefix = "Assistant: " if msg["role"] == "system" else "User: "
                formatted_prompt += f"{role_prefix}{msg['content']}\nAssistant: "

            # Use the generate method which supports streaming
            outputs = model.generate(prompts=[formatted_prompt], sampling_params=sampling_params, use_tqdm=False)

            # Stream the output tokens as they're generated
            request_output = outputs[0]
            generated_text = ""

            for output in request_output.outputs:
                new_text = output.text[len(generated_text) :].strip()
                if new_text:
                    generated_text = output.text
                    yield repair_json(new_text)

        except Exception as e:
            logger.error(f"Error generating streaming text with {model_key}: {e}")
            logger.error(f"Currently loaded models: {list(self._models.keys())}")
            raise

    def process_batch(self, model_key: str, prompts: list[str]) -> list[str]:
        """Process a batch of prompts."""
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
            self._model_refs.clear()
            torch.cuda.empty_cache()
