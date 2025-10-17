import os
import yaml
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

DEFAULT_CONFIG_PATH = "../../../config.yaml"


@dataclass
class EmbeddingsConfig:
    default_model_group: Optional[str] = "all"
    dynamically_load_models: Optional[bool] = False
    execution_providers: List[str] = field(default_factory=lambda: ["CUDAExecutionProvider"])
    total_threads: Optional[int] = 8
    automatic_embeddings_selector: None = None

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
class SamplingParams:
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 512


@dataclass
class OramaAIConfig:
    models_cache_dir: Optional[str] = "/tmp/fastembed_cache"
    embeddings: Optional[EmbeddingsConfig] = field(default_factory=EmbeddingsConfig)
    total_threads: Optional[int] = 12
    config_path = os.getenv('CONFIG_PATH', DEFAULT_CONFIG_PATH)

    def __post_init__(self):
        if Path(self.config_path).exists():
            self.update_from_yaml()
        elif Path("../../config.yml").exists():
            self.update_from_yaml("../../config.yml")

    def update_from_yaml(self, path: str = config_path):
        with open(path) as f:
            config = yaml.safe_load(f)
            config = config.get("ai_server")

            for k, v in config.items():
                if hasattr(self, k):
                    if k == "embeddings" and v is not None:
                        self.embeddings = EmbeddingsConfig(**v)
                    else:
                        setattr(self, k, v)
