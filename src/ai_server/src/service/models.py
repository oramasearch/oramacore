import torch
import logging
import threading
from json_repair import repair_json
from typing import Dict, Any, Set, List, Optional, Iterator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.utils import OramaAIConfig
from src.prompts.main import PROMPT_TEMPLATES
from src.prompts.party_planner_actions import DEFAULT_PARTY_PLANNER_ACTIONS_DATA, RETURN_TYPE_JSON, decode_action_result

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

            # Configuration specific for CPU usage. Not as efficient as GPU.
            if self.device == "cpu":
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                )

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

    def action(self, action: str, input: str, description: str, history: List[Any]) -> Iterator[str] | str:
        actual_model_id = self._model_refs.get("action")
        model_config = self._models.get(actual_model_id)
        model = model_config["model"]
        tokenizer = model_config["tokenizer"]

        action_data = DEFAULT_PARTY_PLANNER_ACTIONS_DATA[action]

        history.insert(0, {"role": "system", "content": action_data["prompt:system"]})
        history.append({"role": "user", "content": action_data["prompt:user"](input, description)})

        formatted_chat = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
        decoded_output = tokenizer.decode(outputs[0][inputs["input_ids"].size(1) :], skip_special_tokens=True)

        if action_data["returns"] == RETURN_TYPE_JSON:
            return decode_action_result(action=action, result=repair_json(decoded_output))

        return decoded_output

    def action_stream(self, action: str, input: str, description: str, history: List[Any]) -> Iterator[str]:
        actual_model_id = self._model_refs.get("action")
        model_config = self._models.get(actual_model_id)
        model = model_config["model"]
        tokenizer = model_config["tokenizer"]

        action_data = DEFAULT_PARTY_PLANNER_ACTIONS_DATA[action]

        history.insert(0, {"role": "system", "content": action_data["prompt:system"]})
        history.append({"role": "user", "content": action_data["prompt:user"](input, description)})

        formatted_chat = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens": True})

        generation_kwargs = dict(
            **inputs, max_new_tokens=512, temperature=0.1, do_sample=True, use_cache=True, streamer=streamer
        )
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        p = None

        for new_text in streamer:
            old_text = p
            p = new_text

            if old_text:
                yield old_text

        yield p.replace("<|im_end|>", "")  # @todo: this sucks. Fix it at transformer level.

    def chat(self, model_id: str, history: List[Any], prompt: str, context: Optional[str] = None) -> str:
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

        formatted_chat = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)

        decoded_output = tokenizer.decode(outputs[0][inputs["input_ids"].size(1) :], skip_special_tokens=True)

        return decoded_output

    def chat_stream(
        self, model_id: str, history: List[Any], prompt: str, context: Optional[str] = None
    ) -> Iterator[str]:
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

        formatted_chat = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens": True})

        generation_kwargs = dict(
            **inputs, max_new_tokens=512, temperature=0.1, do_sample=True, use_cache=True, streamer=streamer
        )
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        p = None

        for new_text in streamer:
            old_text = p
            p = new_text

            if old_text:
                yield old_text

        yield p.replace("<|im_end|>", "")  # @todo: this sucks. Fix it at transformer level.
