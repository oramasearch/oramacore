import sys
import signal
import logging

from src.grpc.server import serve
from src.utils import OramaAIConfig
from src.service.embedding import EmbeddingService
from src.service.models import ModelsManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def handle_shutdown(signum, frame):
    logger.info("\nShutting down gracefully...")
    sys.exit(0)


if __name__ == "__main__":
    # signal.signal(signal.SIGINT, handle_shutdown)
    # signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info("Initializing config...")
    config = OramaAIConfig()

    manager = ModelsManager(config)

    a = manager.chat(
        model_id="google_query_translator",
        history=[],
        prompt="I just installed a Ryzen 9 9900X but I fear I bent some pins. What should I do? I'm scared AF.",
        context=None,
        stream=True,
    )

    print(a)

    # logger.info("Initializing embedding service...")
    # embeddings_service = EmbeddingService(config)

    # try:
    #     logger.info(f"Starting gRPC server on port {config.grpc_port}...")
    #     serve(config, embeddings_service.embeddings_service)
    # except KeyboardInterrupt:
    #     logger.info("\nShutting down gracefully...")
    #     sys.exit(0)
    # except Exception as e:
    #     logger.error(f"Error starting server: {e}", exc_info=True)
    #     sys.exit(1)
