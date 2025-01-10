import sys
import signal

from src.grpc.server import serve
from src.utils import OramaAIConfig
from src.models.main import ModelsManager
from src.service.embedding import EmbeddingService


def handle_shutdown(signum, frame):
    print("\nShutting down gracefully...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    config = OramaAIConfig()

    embeddings_service = EmbeddingService(config)
    models_manager = ModelsManager(config)

    try:
        serve(config, embeddings_service.embeddings_service, models_manager)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        sys.exit(0)
