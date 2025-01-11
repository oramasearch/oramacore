import yaml
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class EmbeddingsConfig:
    default_model_group: Optional[str] = "en"
    dynamically_load_models: Optional[bool] = False
    execution_providers: Optional[List[str]] = field(default_factory=lambda: ["CPUExecutionProvider"])
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
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 512


@dataclass
class ModelConfig:
    id: str = "microsoft/Phi-3.5-mini-instruct"
    tensor_parallel_size: int = 1
    sampling_params: SamplingParams = field(default_factory=SamplingParams)


@dataclass
class LLMs:
    content_expansion: Optional[ModelConfig] = field(
        default_factory=lambda: ModelConfig(
            id="microsoft/Phi-3.5-mini-instruct",
            tensor_parallel_size=1,
            sampling_params=SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512),
        )
    )
    vision: Optional[ModelConfig] = field(
        default_factory=lambda: ModelConfig(
            id="microsoft/Phi-3.5-vision-instruct",
            tensor_parallel_size=1,
            sampling_params=SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512),
        )
    )
    google_query_translator: Optional[ModelConfig] = field(
        default_factory=lambda: ModelConfig(
            id="microsoft/Phi-3.5-mini-instruct",
            tensor_parallel_size=1,
            sampling_params=SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20),
        )
    )


@dataclass
class OramaAIConfig:
    models_cache_dir: Optional[str] = ".embeddins_models_cache"
    api_key: Optional[str] = None
    grpc_port: Optional[int] = 50051
    host: Optional[str] = "0.0.0.0"
    embeddings: Optional[EmbeddingsConfig] = field(default_factory=EmbeddingsConfig)
    LLMs: Optional[LLMs] = field(default_factory=LLMs)
    total_threads: Optional[int] = 12

    def __post_init__(self):
        if Path("config.yaml").exists():
            self.update_from_yaml()

    def update_from_yaml(self, path: str = "config.yaml"):
        with open(path) as f:
            config = yaml.safe_load(f)
            for k, v in config.items():
                if hasattr(self, k) and v is not None:
                    setattr(self, k, v)
