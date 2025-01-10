import threading
from typing import List
from fastembed import TextEmbedding
from src.utils import OramaAIConfig
from fastembed.text.onnx_embedding import supported_onnx_models
from fastembed.text.e5_onnx_embedding import supported_multilingual_e5_models
from src.embeddings.embeddings import embed_alternative, ModelGroups, OramaModelInfo


class EmbeddingsModels:
    def __init__(self, config: OramaAIConfig, selected_models: List[OramaModelInfo]):
        self.config = config
        self.selected_models = selected_models
        self.selected_model_names = [item.name for item in selected_models]
        self.loaded_models = self.load_models()
        self.model_loading_lock = threading.RLock()
        self.model_last_used = {}

        self.models_with_intent_prefix = [s.name for s in ModelGroups.multilingual.value]
        self.models_with_intent_prefix.append(OramaModelInfo.MultilingualE5LargeRaw.name)

    def load_models(self):
        loaded_models = {}

        if not self.config.dynamically_load_models:
            loaded_models = {
                item.name: TextEmbedding(
                    model_name=item.value["model_name"],
                    providers=self.config.execution_providers,
                )
                for item in self.selected_models
            }

        return loaded_models

    def calculate_embeddings(self, input, intent, model_name) -> List[float]:
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
            return embed_alternative(self.loaded_models[model_name], input_strings)

        if self.config.embeddings.dynamically_load_models:
            with self.model_loading_lock:
                if not model_name in self.loaded_models:
                    self.loaded_models[model_name] = TextEmbedding(
                        model_name=OramaModelInfo[model_name].value["model_name"],
                        providers=self.config.embeddings.execution_providers,
                    )

                return embed_alternative(self.loaded_models[model_name], input_strings)
        else:
            raise ValueError(f"Model {model_name} is not loaded")


def extend_fastembed_supported_models():
    supported_onnx_models.extend(
        [
            {
                "model": "intfloat/multilingual-e5-small",
                "dim": 384,
                "description": "Text embeddings, Unimodal (text), Multilingual (~100 languages), 512 input tokens truncation, Prefixes for queries/documents: necessary, 2024 year.",
                "license": "mit",
                "size_in_GB": 0.4,
                "sources": {
                    "hf": "intfloat/multilingual-e5-small",
                },
                "model_file": "onnx/model.onnx",
            },
            {
                "model": "intfloat/multilingual-e5-base",
                "dim": 768,
                "description": "Text embeddings, Unimodal (text), Multilingual (~100 languages), 512 input tokens truncation, Prefixes for queries/documents: necessary, 2024 year.",
                "license": "mit",
                "size_in_GB": 1.11,
                "sources": {
                    "hf": "intfloat/multilingual-e5-base",
                },
                "model_file": "onnx/model.onnx",
            },
            {
                "model": "BAAI/bge-small-en-v1.5-raw",
                "dim": 384,
                "description": "Text embeddings, Unimodal (text), English, 512 input tokens truncation, Prefixes for queries/documents: not so necessary, 2023 year.",
                "license": "mit",
                "size_in_GB": 0.4,
                "sources": {
                    "hf": "BAAI/bge-small-en-v1.5",
                },
                "model_file": "onnx/model.onnx",
            },
            {
                "model": "BAAI/bge-base-en-v1.5-raw",
                "dim": 768,
                "description": "Text embeddings, Unimodal (text), English, 512 input tokens truncation, Prefixes for queries/documents: not so necessary, 2023 year.",
                "license": "mit",
                "size_in_GB": 1.11,
                "sources": {
                    "hf": "BAAI/bge-base-en-v1.5",
                },
                "model_file": "onnx/model.onnx",
            },
            {
                "model": "BAAI/bge-large-en-v1.5-raw",
                "dim": 1024,
                "description": "Text embeddings, Unimodal (text), English, 512 input tokens truncation, Prefixes for queries/documents: not so necessary, 2023 year.",
                "license": "mit",
                "size_in_GB": 1.20,
                "sources": {
                    "hf": "BAAI/bge-large-en-v1.5",
                },
                "model_file": "onnx/model.onnx",
            },
        ]
    )
    supported_multilingual_e5_models.append(
        {
            "model": "intfloat/multilingual-e5-large-raw",
            "dim": 1024,
            "description": "Text embeddings, Unimodal (text), Multilingual (~100 languages), 512 input tokens truncation, Prefixes for queries/documents: necessary, 2024 year.",
            "license": "mit",
            "size_in_GB": 2.24,
            "sources": {
                "hf": "intfloat/multilingual-e5-large",
            },
            "model_file": "onnx/model.onnx",
            "additional_files": ["onnx/model.onnx_data"],
        }
    )
