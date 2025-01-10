import grpc
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
)

from src.embeddings.models import OramaModelInfo


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
        try:
            model_key = LLMType.Name(request.model).lower()
            response_text = self.models_manager.generate_text(model_key=model_key, prompt=request.prompt)
            return LLMResponse(text=response_text)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing LLM request: {str(e)}")
            return LLMResponse()

    def CallLLMStream(self, request, context):
        try:
            timeout = 60  # in seconds
            context.set_timeout(timeout)
            model_data = self.models_manager.get_model(LLMType.Name(request.model).lower())
            if not model_data:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Model not available")
                return

            model = model_data["model"]
            sampling_params = model_data["config"]["sampling_params"]

            outputs = model.generate([request.prompt], sampling_params, stream=True)

            for output in outputs:
                for token in output.outputs[0].token_texts:
                    yield LLMStreamResponse(text_chunk=token, is_final=False)

            # Send final chunk
            yield LLMStreamResponse(text_chunk="", is_final=True)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error in stream: {str(e)}")
            return


class VisionService(service_pb2_grpc.VisionServiceServicer):
    def __init__(self, models_manager):
        self.models_manager = models_manager

    def CallVision(self, request, context):
        try:
            if not request.image:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Image data is required")
                return VisionResponse()

            prompt = f"[IMAGE]\n{request.text}"
            response_text = self.models_manager.generate_text(model_key="vision", prompt=prompt)
            return VisionResponse(text=response_text)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing Vision request: {str(e)}")
            return VisionResponse()


def serve(config, embeddings_service, models_manager):
    print(f"Starting gRPC server on port {config.grpc_port}")
    server = grpc.server(ThreadPoolExecutor(max_workers=10))

    embedding_service = CalculateEmbeddingService(embeddings_service)
    llm_service = LLMService(models_manager)
    vision_service = VisionService(models_manager)

    service_pb2_grpc.add_LLMServiceServicer_to_server(llm_service, server)
    service_pb2_grpc.add_VisionServiceServicer_to_server(vision_service, server)
    service_pb2_grpc.add_CalculateEmbeddingsServiceServicer_to_server(embedding_service, server)

    SERVICE_NAMES = (
        service_pb2.DESCRIPTOR.services_by_name["CalculateEmbeddingsService"].full_name,
        service_pb2.DESCRIPTOR.services_by_name["LLMService"].full_name,
        service_pb2.DESCRIPTOR.services_by_name["VisionService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    server.add_insecure_port(f"[::]:{config.grpc_port}")
    server.start()
    server.wait_for_termination()
