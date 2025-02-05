import os
import logging
from concurrent.futures import ThreadPoolExecutor

from src.embeddings.models import (
    EmbeddingsModels,
    ModelGroups,
)
from src.embeddings.embeddings import initialize_thread_executor

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, config):
        logger.info("Starting EmbeddingService initialization...")

        logger.info("Setting up thread executor...")
        self.config = config
        self.thread_executor = ThreadPoolExecutor(max_workers=config.total_threads // 2)

        logger.info("Initializing embeddings service...")
        self.embeddings_service = self._initialize_embeddings_service()
        logger.info("EmbeddingService initialization complete")

    def _initialize_embeddings_service(self):
        logger.info("Setting ONNXRUNTIME_PROVIDERS...")
        os.environ["ONNXRUNTIME_PROVIDERS"] = "CUDAExecutionProvider"

        logger.info("Initializing thread executor...")
        initialize_thread_executor(max_workers=self.config.total_threads // 2)

        logger.info("Creating EmbeddingsModels instance...")
        try:
            model_group = self.config.embeddings.default_model_group
            logger.info(f"Using model group: {model_group}")
            selected_models = ModelGroups[model_group].value
            logger.info(f"Selected models: {[model.name for model in selected_models]}")

            return EmbeddingsModels(
                self.config,
                selected_models=selected_models,
            )
        except Exception as e:
            logger.error(f"Error initializing EmbeddingsModels: {str(e)}", exc_info=True)
            raise
