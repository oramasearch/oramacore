import os
import logging
import threading
from typing import List, Optional
from fastembed import TextEmbedding
from pydantic import BaseModel, Field

from src.embeddings.embeddings import embed_alternative, ModelGroups, OramaModelInfo

logger = logging.getLogger(__name__)

BGE_QUERY_INSTRUCTIONS = "Represent this sentence for searching relevant passages: "
BGE_PASSAGE_INSTRUCTIONS = ""
E5_QUERY_INSTRUCTIONS = "query: "
E5_PASSAGE_INSTRUCTIONS = "passage: "
MULTILINGUALMINILML12V2_QUERY_INSTRUCTIONS = ""
MULTILINGUALMINILML12V2_PASSAGE_INSTRUCTIONS = ""
JINAEMBEDDINGSV2BASECODE_QUERY_INSTRUCTIONS = ""
JINAEMBEDDINGSV2BASECODE_PASSAGE_INSTRUCTIONS = ""

MODEL_QUERY_INSTRUCTIONS_MAP = {
    "BGESmall": BGE_QUERY_INSTRUCTIONS,
    "BGEBase": BGE_QUERY_INSTRUCTIONS,
    "BGELarge": BGE_QUERY_INSTRUCTIONS,
    "MultilingualE5Small": E5_QUERY_INSTRUCTIONS,
    "MultilingualE5Base": E5_QUERY_INSTRUCTIONS,
    "MultilingualE5Large": E5_QUERY_INSTRUCTIONS,
    "MultilingualMiniLML12V2": MULTILINGUALMINILML12V2_QUERY_INSTRUCTIONS,
    "JinaEmbeddingsV2BaseCode": JINAEMBEDDINGSV2BASECODE_QUERY_INSTRUCTIONS,
}

MODEL_PASSAGE_INSTRUCTIONS_MAP = {
    "BGESmall": BGE_PASSAGE_INSTRUCTIONS,
    "BGEBase": BGE_PASSAGE_INSTRUCTIONS,
    "BGELarge": BGE_PASSAGE_INSTRUCTIONS,
    "MultilingualE5Small": E5_PASSAGE_INSTRUCTIONS,
    "MultilingualE5Base": E5_PASSAGE_INSTRUCTIONS,
    "MultilingualE5Large": E5_PASSAGE_INSTRUCTIONS,
    "MultilingualMiniLML12V2": MULTILINGUALMINILML12V2_PASSAGE_INSTRUCTIONS,
    "JinaEmbeddingsV2BaseCode": JINAEMBEDDINGSV2BASECODE_PASSAGE_INSTRUCTIONS,
}

class EmbeddingsModelsConfiguration(BaseModel):
    """EmbeddingsModels configuration."""
    execution_providers: List[str] = Field(description="execution_providers for the model")
    dynamically_load_models: bool = Field(description="Whether to dynamically load models on demand")
    default_model_group: str = Field(default="all", description="Default model group to use if no models are specified")


class EmbeddingsModels:
    def __init__(self, config: EmbeddingsModelsConfiguration):
        logger.info("Initializing EmbeddingsModels: %s", config)
        self.config = config

        self.selected_models = ModelGroups[self.config.default_model_group].value
        self.selected_model_names = [item.name for item in self.selected_models]

        logger.info(f"Configured execution providers: {self.config.execution_providers}")

        logger.info(f"Creating cache directory: /tmp/fastembed_cache")
        os.makedirs("/tmp/fastembed_cache", exist_ok=True)

        logger.info("Setting FastEmbed cache directory...")
        os.environ["FASTEMBED_CACHE_DIR"] = os.path.abspath("/tmp/fastembed_cache")

        logger.info("Loading models...")
        self.loaded_models = self.load_models()
        logger.info("Models loaded successfully")

        self.model_loading_lock = threading.RLock()
        self.model_last_used = {}

    def load_models(self):
        loaded_models = {}

        if not getattr(self.config, "dynamically_load_models", False):
            logger.info(f"Pre-loading models with execution providers: {self.config.execution_providers}")
            loaded_models = {
                item.name: TextEmbedding(
                    model_name=item.value["model_name"],
                    providers=self.config.execution_providers,
                )
                for item in self.selected_models
            }
        else:
            logger.info("Dynamic model loading enabled - models will be loaded on demand")

        return loaded_models

    def calculate_embeddings(self, input, intent, model_name) -> List[List[float]]:
        input_array = [input] if isinstance(input, str) else input
        input_with_instructions = []

        for text in input_array:
            if intent == "query":
                input_with_instructions.append(f"{MODEL_QUERY_INSTRUCTIONS_MAP[model_name]}{text}")
            elif intent == "passage":
                input_with_instructions.append(f"{MODEL_PASSAGE_INSTRUCTIONS_MAP[model_name]}{text}")
            else:
                raise ValueError(f"Unknown intent: {intent}. Supported intents are 'query' and 'passage'.")

        if model_name not in self.selected_model_names:
            raise ValueError(
                f"Model {model_name} is not supported:\n Supported models {', '.join(self.selected_model_names)}"
            )

        if model_name in self.loaded_models:
            return list(embed_alternative(self.loaded_models[model_name], input_with_instructions))

        if self.config.dynamically_load_models:
            with self.model_loading_lock:
                if model_name not in self.loaded_models:
                    self.loaded_models[model_name] = TextEmbedding(
                        model_name=OramaModelInfo[model_name].value["model_name"],
                        providers=self.config.execution_providers,
                    )

                return list(embed_alternative(self.loaded_models[model_name], input_with_instructions))
        else:
            raise ValueError(f"Model {model_name} is not loaded")
