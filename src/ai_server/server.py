from src.utils import OramaAIConfig
from src.service.embedding import EmbeddingService
from src.models.main import ModelsManager

if __name__ == "__main__":
    config = OramaAIConfig()
    service = EmbeddingService(config)
    service.start()
