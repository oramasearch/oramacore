import yaml
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field


DEFAULT_GENERAL_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_VISION_MODEL = "microsoft/Phi-3.5-vision-instruct"
DEFAULT_CONTENT_EXPANSION_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_GOOGLE_QUERY_TRANSLATOR_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ANSWER_MODEL = "Qwen/Qwen2.5-3B-Instruct"


@dataclass
class EmbeddingsConfig:
    default_model_group: Optional[str] = "en"
    dynamically_load_models: Optional[bool] = False
    execution_providers: Optional[List[str]] = field(default_factory=lambda: ["CUDAExecutionProvider"])
    total_threads: Optional[int] = 8

    def __post_init__(self):
        available_providers = ["CUDAExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"]
        self.execution_providers = [
            provider for provider in self.execution_providers if provider in available_providers
        ]
        if not self.execution_providers:
            self.execution_providers = ["CPUExecutionProvider"]


@dataclass
class SamplingParams:
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512


@dataclass
class ModelConfig:
    id: str = DEFAULT_GENERAL_MODEL
    tensor_parallel_size: int = 1
    sampling_params: SamplingParams = field(default_factory=SamplingParams)


@dataclass
class LLMs:
    answer: Optional[ModelConfig] = field(
        default_factory=lambda: ModelConfig(
            id=DEFAULT_ANSWER_MODEL,
            tensor_parallel_size=1,
            sampling_params=SamplingParams(temperature=0.0, top_p=0.95, max_tokens=2048),
        )
    )
    content_expansion: Optional[ModelConfig] = field(
        default_factory=lambda: ModelConfig(
            id=DEFAULT_CONTENT_EXPANSION_MODEL,
            tensor_parallel_size=1,
            sampling_params=SamplingParams(temperature=0.2, top_p=0.95, max_tokens=512),
        )
    )
    # vision: Optional[ModelConfig] = field(
    #     default_factory=lambda: ModelConfig(
    #         id=DEFAULT_VISION_MODEL,
    #         tensor_parallel_size=1,
    #         sampling_params=SamplingParams(temperature=0.2, top_p=0.95, max_tokens=512),
    #     )
    # )
    google_query_translator: Optional[ModelConfig] = field(
        default_factory=lambda: ModelConfig(
            id=DEFAULT_GOOGLE_QUERY_TRANSLATOR_MODEL,
            tensor_parallel_size=1,
            sampling_params=SamplingParams(temperature=0.2, top_p=0.95, max_tokens=20),
        )
    )


@dataclass
class OramaAIConfig:
    models_cache_dir: Optional[str] = ".embeddings_models_cache"
    grpc_port: Optional[int] = 50051
    host: Optional[str] = "0.0.0.0"
    embeddings: Optional[EmbeddingsConfig] = field(default_factory=EmbeddingsConfig)
    LLMs: Optional[LLMs] = field(default_factory=LLMs)  # type: ignore
    total_threads: Optional[int] = 12

    def __post_init__(self):
        if Path("config.yaml").exists():
            self.update_from_yaml()

    def update_from_yaml(self, path: str = "config.yaml"):
        with open(path) as f:
            config = yaml.safe_load(f)

            for k, v in config.items():
                if hasattr(self, k):
                    if k == "embeddings" and v is not None:
                        self.embeddings = EmbeddingsConfig(**v)
                    elif k == "LLMs" and v is not None:
                        llm_configs = {}
                        for model_key, model_config in v.items():
                            if model_config:
                                sampling_params = SamplingParams(**model_config.pop("sampling_params", {}))
                                llm_configs[model_key] = ModelConfig(**model_config, sampling_params=sampling_params)
                        self.LLMs = LLMs(**llm_configs)
                    else:
                        setattr(self, k, v)
