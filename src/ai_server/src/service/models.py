import torch
import logging
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from typing import Dict, Any, Set, List, Optional

from src.prompts.main import PROMPT_TEMPLATES
from src.utils import OramaAIConfig

logger = logging.getLogger(__name__)


class ModelsManager:
    def __init__(self, config: OramaAIConfig):
        self.config = config
        self.model_configs = {
            key.lower(): value
            for key, value in vars(self.config.LLMs).items()
            if key not in ["__dict__", "__weakref__", "__doc__"]
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models: Dict[str, Dict[str, Any]] = {}
        self._model_refs: Dict[str, str] = {}
        self._lock = threading.Lock()

        for config_name, config in self.model_configs.items():
            if config is not None:
                self._model_refs[config_name] = config.id

        unique_models = self._get_unique_model_ids()
        logger.info(f"Unique models to load: {unique_models}")

        for model_id in unique_models:
            self._preload_unique_model(model_id)

    def _get_unique_model_ids(self) -> Set[str]:
        return {config.id for config in self.model_configs.values() if config is not None}

    def _preload_unique_model(self, model_id: str) -> None:
        try:
            logger.info(f"Loading model {model_id}...")

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,  # @todo: make this configurable and False by default
            )

            model = model.to(self.device)

            self._models[model_id] = {
                "model": model,
                "tokenizer": AutoTokenizer.from_pretrained(model_id),
                "configs": {},
            }

            logger.info(f"Successfully loaded model {model_id}")

        except Exception as e:
            logger.error(f"Error preloading model {model_id}: {e}")
            if model_id in self._models:
                del self._models[model_id]
            raise

    def chat(self, model_id: str, history: List[Any], prompt: str, context: Optional[str] = None, stream: bool = True):
        actual_model_id = self._model_refs.get(model_id)
        if actual_model_id is None:
            raise ValueError(f"Unknown model configuration: {model_id}")

        model_config = self._models.get(actual_model_id)
        if model_config is None:
            raise ValueError(f"Model not loaded: {actual_model_id}")

        model = model_config["model"]
        tokenizer = model_config["tokenizer"]

        history.insert(0, {"role": "system", "content": PROMPT_TEMPLATES[f"{model_id}:system"]})
        history.append({"role": "user", "content": PROMPT_TEMPLATES[f"{model_id}:user"](prompt, context)})

        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)

        model_inputs = tokenizer([text], return_tensors="pt")
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=20,
            streamer=streamer,
            do_sample=True,
            temperature=0.3,
            top_p=0.90,
        )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response_text = generated_text[len(text) :].strip()

        return response_text
