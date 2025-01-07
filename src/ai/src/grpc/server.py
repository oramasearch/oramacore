from concurrent.futures import ThreadPoolExecutor
import grpc
from grpc_reflection.v1alpha import reflection

import service_pb2
import service_pb2_grpc
from service_pb2 import (
    OramaModel as ProtoOramaModel,
    OramaIntent as ProtoOramaIntent,
    Embedding as EmbeddingProto,
    EmbeddingResponse as EmbeddingResponseProto,
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


def serve(config, embeddings_service):
    print(f"Starting gRPC server on port {config.embeddings_grpc_port}")
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    service = CalculateEmbeddingService(embeddings_service)
    service_pb2_grpc.add_CalculateEmbeddingsServiceServicer_to_server(service, server)

    SERVICE_NAMES = (
        service_pb2.DESCRIPTOR.services_by_name["CalculateEmbeddingsService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    server.add_insecure_port(f"[::]:{config.embeddings_grpc_port}")
    server.start()
    server.wait_for_termination()
