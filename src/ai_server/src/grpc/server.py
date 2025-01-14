import grpc
import base64
import logging
from PIL import Image
from io import BytesIO
from grpc_reflection.v1alpha import reflection
from concurrent.futures import ThreadPoolExecutor

import service_pb2
import service_pb2_grpc
from service_pb2 import (
    OramaModel as ProtoOramaModel,
    OramaIntent as ProtoOramaIntent,
    Embedding as EmbeddingProto,
    EmbeddingResponse as EmbeddingResponseProto,
    LLMType,
    LLMResponse,
    LLMStreamResponse,
    VisionResponse,
    HealthCheckResponse,
)

from src.embeddings.models import OramaModelInfo


class HealthCheckService(service_pb2_grpc.HealthCheckServiceServicer):
    def CheckHealth(self, request, context):
        return HealthCheckResponse(status="OK")


class CalculateEmbeddingService(service_pb2_grpc.CalculateEmbeddingsServiceServicer):
    def __init__(self, embeddings_service):
        self.embeddings_service = embeddings_service

    def GetEmbedding(self, request, context):
        model_name = ProtoOramaModel.Name(request.model)
        embeddings = self.embeddings_service.calculate_embeddings(
            request.input, ProtoOramaIntent.Name(request.intent), model_name
        )

        return EmbeddingResponseProto(
            embeddings_result=[EmbeddingProto(embeddings=e.tolist()) for e in embeddings],
            dimensions=OramaModelInfo[model_name].value["dimensions"],
        )


class LLMService(service_pb2_grpc.LLMServiceServicer):
    def __init__(self, models_manager):
        self.models_manager = models_manager

    def CallLLM(self, request, context):
        """Generate text using a model."""
        try:
            model_key = LLMType.Name(request.model).lower()
            response_text = self.models_manager.generate_text(model_key=model_key, prompt=request.prompt)
            return LLMResponse(text=response_text)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing LLM request: {str(e)}")
            return LLMResponse()

    def CallLLMStream(self, request, context):
        """Generate text using a model with streaming response."""
        try:
            model_key = LLMType.Name(request.model).lower()

            try:
                for text_chunk in self.models_manager.generate_text_stream(model_key=model_key, prompt=request.prompt):
                    yield LLMStreamResponse(text_chunk=text_chunk, is_final=False)
                # Send final chunk
                yield LLMStreamResponse(text_chunk="", is_final=True)

            except Exception as e:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Error during stream generation: {str(e)}")
                return

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing streaming LLM request: {str(e)}")
            return


class VisionService(service_pb2_grpc.VisionServiceServicer):
    def __init__(self, models_manager):
        self.models_manager = models_manager
        self.logger = logging.getLogger(__name__)

    def preprocess_image(self, image: Image.Image, max_size: int = 256) -> Image.Image:
        """
        Preprocess image by:
        1. Resizing to a smaller size while maintaining aspect ratio
        2. Converting to RGB mode if needed
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if larger than max_size
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    def CallVision(self, request, context):
        try:
            if not request.image:
                self.logger.error("No image data provided")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Image data is required")
                return VisionResponse()

            try:
                image = Image.open(BytesIO(request.image))
                image.load()
                self.logger.info(f"Original image: format={image.format}, size={image.size}")

                processed_image = self.preprocess_image(image)
                self.logger.info(f"Processed image size: {processed_image.size}")

                img_byte_arr = BytesIO()
                processed_image.save(img_byte_arr, format="JPEG", quality=80, optimize=True)
                img_byte_arr = img_byte_arr.getvalue()
                self.logger.info(f"Compressed image size in bytes: {len(img_byte_arr)}")

            except Exception as e:
                self.logger.error(f"Image processing failed: {e}")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Image processing failed: {str(e)}")
                return VisionResponse()

            image_b64 = base64.b64encode(img_byte_arr).decode("utf-8")
            user_text = request.text if request.text else "What is in this image?"

            prompt = f"<|user|>\n<|image_1|>\n{user_text}<|end|>\n<|assistant|>\n"

            prompt_data = {"prompt": prompt, "multi_modal_data": {"image": processed_image}}

            self.logger.info("Sending prompt to vision model")
            response_text = self.models_manager.generate_text(model_key="vision", prompt=prompt_data)
            self.logger.info(f"Received response from vision model: {response_text}")

            return VisionResponse(text=response_text)

        except Exception as e:
            self.logger.error(f"Error processing Vision request: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing Vision request: {str(e)}")
            return VisionResponse()


def serve(config, embeddings_service, models_manager):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting gRPC server on port {config.grpc_port}")
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    logger.info("gRPC server created")

    embedding_service = CalculateEmbeddingService(embeddings_service)
    llm_service = LLMService(models_manager)
    vision_service = VisionService(models_manager)
    health_check_service = HealthCheckService()

    service_pb2_grpc.add_LLMServiceServicer_to_server(llm_service, server)
    service_pb2_grpc.add_VisionServiceServicer_to_server(vision_service, server)
    service_pb2_grpc.add_CalculateEmbeddingsServiceServicer_to_server(embedding_service, server)
    service_pb2_grpc.add_HealthCheckServiceServicer_to_server(health_check_service, server)

    SERVICE_NAMES = (
        service_pb2.DESCRIPTOR.services_by_name["CalculateEmbeddingsService"].full_name,
        service_pb2.DESCRIPTOR.services_by_name["LLMService"].full_name,
        service_pb2.DESCRIPTOR.services_by_name["VisionService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    logger.info(f"Available gRPC services: {SERVICE_NAMES}")

    server.add_insecure_port(f"[::]:{config.grpc_port}")
    server.start()
    server.wait_for_termination()
