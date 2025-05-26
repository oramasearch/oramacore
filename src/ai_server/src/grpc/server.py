import grpc
import json
import logging
from textwrap import dedent
from json_repair import repair_json
from grpc_reflection.v1alpha import reflection
from concurrent.futures import ThreadPoolExecutor

import service_pb2
import service_pb2_grpc
from service_pb2 import (
    OramaModel as ProtoOramaModel,
    OramaIntent as ProtoOramaIntent,
    Embedding as EmbeddingProto,
    EmbeddingResponse as EmbeddingResponseProto,
    HealthCheckResponse,
    NLPQueryTriggerRequest as NLPQueryTriggerRequestProto,
    NLPQueryTriggerResponse as NLPQueryTriggerResponseProto,
)
from src.utils import OramaAIConfig


class LLMService(service_pb2_grpc.LLMServiceServicer):
    def __init__(self, embeddings_service, search_intent_detector, config: OramaAIConfig):
        self.config = config
        self.embeddings_service = embeddings_service
        self.search_intent_detector = search_intent_detector

    def CheckHealth(self, request, context):
        return HealthCheckResponse(status="OK")

    def GetEmbedding(self, request, context):
        try:
            model_name = ProtoOramaModel.Name(request.model)
            intent_name = ProtoOramaIntent.Name(request.intent)

            embeddings = self.embeddings_service.calculate_embeddings(request.input, intent_name, model_name)

            return EmbeddingResponseProto(
                embeddings_result=[EmbeddingProto(embeddings=e.tolist()) for e in embeddings],
                dimensions=embeddings[0].shape[0] if embeddings else 0,
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error calculating embeddings: {str(e)}")
            return EmbeddingResponseProto()

    def NLPQueryTrigger(self, request, context):
        language = request.language
        query = request.query
        autodetect_language = language == "auto"
        result = self.search_intent_detector.process_search_query(query, None if autodetect_language else language)

        return NLPQueryTriggerResponseProto(
            should_search=result.get("should_search", False),
            searchable_content=result.get("searchable_content", ""),
            detected_language=result.get("detected_language", ""),
            original_text=result.get("original_text", ""),
            processing_time_ms=result.get("processing_time_ms", 0.0),
            model_used=result.get("model_used", ""),
        )


class AuthInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, handler_call_details):

        # Health check and embeddings won't require authentication.
        # This server should never be exposed to the public and it's meant for internal use only.
        allowed_methods = ["CheckHealth", "GetEmbedding", "ServerReflection", "NLPQueryTrigger"]
        if any(x in handler_call_details.method for x in allowed_methods):
            return continuation(handler_call_details)

        # The current gRPC server is a proxy for the Rust server, which requires an API key.
        # There's no API key validation in the Python server, so we just check if the API key is present.
        metadata = dict(handler_call_details.invocation_metadata)
        if "x-api-key" not in metadata:
            return grpc.unary_unary_rpc_method_handler(
                lambda req, ctx: ctx.abort(grpc.StatusCode.UNAUTHENTICATED, "Missing API key")
            )
        return continuation(handler_call_details)


def serve(config, embeddings_service, search_intent_detector):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting gRPC server on port {config.port}")
    server = grpc.server(ThreadPoolExecutor(max_workers=10), interceptors=[AuthInterceptor()])
    logger.info("gRPC server created")

    llm_service = LLMService(embeddings_service, search_intent_detector, config)
    service_pb2_grpc.add_LLMServiceServicer_to_server(llm_service, server)

    SERVICE_NAMES = (
        service_pb2.DESCRIPTOR.services_by_name["LLMService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    logger.info(f"Available gRPC services: {SERVICE_NAMES}")

    server.add_insecure_port(f"{config.host}:{config.port}")
    server.start()
    server.wait_for_termination()
