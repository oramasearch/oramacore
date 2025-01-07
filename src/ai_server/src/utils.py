import yaml
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class OramaAIConfig:
    api_key: Optional[str] = None
    embeddings_grpc_port: int = 50051
    http_port: int = 5000
    host: str = "0.0.0.0"
    execution_providers: List[str] = field(default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    total_threads: int = 8
    default_model_group: str = "en"
    dynamically_load_models: bool = False

    def __post_init__(self):
        if Path("config.yaml").exists():
            self.update_from_yaml()

    def update_from_yaml(self, path: str = "config.yaml"):
        with open(path) as f:
            config = yaml.safe_load(f)
            for k, v in config.items():
                if hasattr(self, k) and v is not None:
                    setattr(self, k, v)
