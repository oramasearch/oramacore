import yaml
import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

BASE_MODEL = "Qwen/Qwen2.5-3B"
DEFAULT_GENERAL_MODEL = BASE_MODEL
DEFAULT_CONFIG_PATH = "../../config.yaml"


@dataclass
class EmbeddingsConfig:
    default_model_group: Optional[str] = "en"
    dynamically_load_models: Optional[bool] = False
    execution_providers: List[str] = field(default_factory=lambda: ["CUDAExecutionProvider"])
    total_threads: Optional[int] = 8

    def __post_init__(self):
        available_providers = [
            "CUDAExecutionProvider",
            "AzureExecutionProvider",
            "CPUExecutionProvider",
        ]
        self.execution_providers = [
            provider for provider in self.execution_providers if provider in available_providers
        ]
        if not self.execution_providers:
            self.execution_providers = ["CPUExecutionProvider"]


@dataclass
class RustServerConfig:
    host: Optional[str] = "0.0.0.0"
    port: Optional[int] = 8080


@dataclass
class SamplingParams:
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 512


@dataclass
class ModelConfig:
    id: str = DEFAULT_GENERAL_MODEL
    tensor_parallel_size: int = 1
    use_cpu: bool = False
    sampling_params: SamplingParams = field(default_factory=SamplingParams)


@dataclass
class LLMConfig:
    answer: Optional[ModelConfig] = None
    content_expansion: Optional[ModelConfig] = None
    google_query_translator: Optional[ModelConfig] = None
    party_planner: Optional[ModelConfig] = None
    action: Optional[ModelConfig] = None


@dataclass
class OramaAIConfig:
    models_cache_dir: Optional[str] = ".embeddings_models_cache"
    port: Optional[int] = 50051
    host: Optional[str] = "0.0.0.0"
    embeddings: Optional[EmbeddingsConfig] = field(default_factory=EmbeddingsConfig)
    LLMs: LLMConfig = field(default_factory=LLMConfig)
    total_threads: Optional[int] = 12

    rust_server_host: Optional[str] = "0.0.0.0"
    rust_server_port: Optional[int] = 8080

    default_model: Optional[str] = BASE_MODEL

    def __post_init__(self):
        if Path(DEFAULT_CONFIG_PATH).exists():
            self.update_from_yaml()
        elif Path("../../config.yml").exists():
            self.update_from_yaml("../../config.yml")

    def update_from_yaml(self, path: str = DEFAULT_CONFIG_PATH):
        with open(path) as f:
            config = yaml.safe_load(f)
            config = config.get("ai_server")
            rust_server_config = config.get("http", {})

            if rust_server_config:
                self.rust_server_host = rust_server_config.get("host", self.rust_server_host)
                self.rust_server_port = rust_server_config.get("port", self.rust_server_port)

            for k, v in config.items():
                if hasattr(self, k):
                    if k == "embeddings" and v is not None:
                        self.embeddings = EmbeddingsConfig(**v)
                    elif k == "LLMs" and v is not None:
                        llm_configs = {}
                        # Extract default_model if present
                        default_model = None
                        if "default_model" in v:
                            default_model = v.pop("default_model")
                            if isinstance(default_model, dict) and "id" in default_model:
                                self.default_model = default_model["id"]

                        for model_key, model_config in v.items():
                            if model_config:
                                sampling_params = SamplingParams(**model_config.pop("sampling_params", {}))
                                llm_configs[model_key] = ModelConfig(**model_config, sampling_params=sampling_params)
                        self.LLMs = LLMConfig(**llm_configs)
                    else:
                        setattr(self, k, v)


def json_to_md(data, level=0) -> str:
    if isinstance(data, str):
        data = json.loads(data)

    indent = "  " * level
    md = ""

    if isinstance(data, list) and data and isinstance(data[0], list):
        data = data[0]

    if isinstance(data, dict):
        for key, value in data.items():
            md += f"{indent}- **{key}**: "
            if isinstance(value, (dict, list)):
                md += "\n" + json_to_md(value, level + 1)
            else:
                md += f"`{value}`\n"
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                md += json_to_md(item, level)
            else:
                md += f"{indent}- `{item}`\n"

    return md
