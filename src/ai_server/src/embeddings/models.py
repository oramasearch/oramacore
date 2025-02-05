import os
import logging
import threading
from typing import List
from fastembed import TextEmbedding

from src.utils import OramaAIConfig
from src.embeddings.embeddings import embed_alternative, ModelGroups, OramaModelInfo

logger = logging.getLogger(__name__)


class EmbeddingsModels:
    def __init__(self, config: OramaAIConfig, selected_models: List[OramaModelInfo]):
        logger.info("Initializing EmbeddingsModels...")
        self.config = config
        self.selected_models = selected_models
        self.selected_model_names = [item.name for item in selected_models]

        logger.info(f"Creating cache directory: {config.models_cache_dir}")
        os.makedirs(config.models_cache_dir, exist_ok=True)

        logger.info("Setting FastEmbed cache directory...")
        os.environ["FASTEMBED_CACHE_DIR"] = os.path.abspath(config.models_cache_dir)

        logger.info("Loading models...")
        self.loaded_models = self.load_models()
        logger.info("Models loaded successfully")

        self.model_loading_lock = threading.RLock()
        self.model_last_used = {}

        self.models_with_intent_prefix = [
            s.name for s in ModelGroups.multilingual.value
        ]
        self.models_with_intent_prefix.append(
            OramaModelInfo.MultilingualE5LargeRaw.name
        )

    def load_models(self):
        loaded_models = {}

        if not getattr(self.config, "dynamically_load_models", False):
            loaded_models = {
                item.name: TextEmbedding(
                    model_name=item.value["model_name"],
                    providers=self.config.embeddings.execution_providers,
                )
                for item in self.selected_models
            }

        return loaded_models

    def calculate_embeddings(self, input, intent, model_name) -> List[List[float]]:
        input_array = [input] if isinstance(input, str) else input
        if model_name not in self.selected_model_names:
            raise ValueError(
                f"Model {model_name} is not supported:\n Supported models {', '.join(self.selected_model_names)}"
            )

        input_strings = (
            [f"{intent}: {s}" for s in input_array]
            if (model_name in self.models_with_intent_prefix and intent)
            else input_array
        )

        if model_name in self.loaded_models:
            return list(
                embed_alternative(self.loaded_models[model_name], input_strings)
            )

        if self.config.embeddings.dynamically_load_models:
            with self.model_loading_lock:
                if model_name not in self.loaded_models:
                    self.loaded_models[model_name] = TextEmbedding(
                        model_name=OramaModelInfo[model_name].value["model_name"],
                        providers=self.config.embeddings.execution_providers,
                    )

                return list(
                    embed_alternative(self.loaded_models[model_name], input_strings)
                )
        else:
            raise ValueError(f"Model {model_name} is not loaded")
