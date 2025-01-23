import grpc
import logging
from grpc_reflection.v1alpha import reflection
from concurrent.futures import ThreadPoolExecutor

import service_pb2
import service_pb2_grpc
from service_pb2 import (
    OramaModel as ProtoOramaModel,
    OramaIntent as ProtoOramaIntent,
    Embedding as EmbeddingProto,
    Role as ProtoRole,
    EmbeddingResponse as EmbeddingResponseProto,
    ChatResponse,
    ChatStreamResponse,
    HealthCheckResponse,
    LLMType,
    PlannedAnswerResponse,
)
from src.prompts.party_planner import PartyPlannerActions
from src.prompts.main import PROMPT_TEMPLATES


class LLMService(service_pb2_grpc.LLMServiceServicer):
    def __init__(self, embeddings_service, models_manager):
        self.embeddings_service = embeddings_service
        self.models_manager = models_manager
        self.party_planner_actions = PartyPlannerActions()

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

    def Chat(self, request, context):
        try:
            model_name = LLMType.Name(request.model)
            history = (
                [
                    {"role": ProtoRole.Name(message.role).lower(), "content": message.content}
                    for message in request.conversation.messages
                ]
                if request.conversation.messages
                else []
            )

            response = self.models_manager.chat(model_id=model_name.lower(), history=history, prompt=request.prompt)
            return ChatResponse(text=response)
        except Exception as e:
            logging.error(f"Error in Chat: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error in chat: {str(e)}")
            return ChatResponse()

    def ChatStream(self, request, context):
        try:
            model_name = LLMType.Name(request.model)
            history = (
                [
                    {"role": ProtoRole.Name(message.role).lower(), "content": message.content}
                    for message in request.conversation.messages
                ]
                if request.conversation.messages
                else []
            )

            for text_chunk in self.models_manager.chat_stream(
                model_id=model_name.lower(), history=history, prompt=request.prompt, context=request.context
            ):
                yield ChatStreamResponse(text_chunk=text_chunk, is_final=False)
            yield ChatStreamResponse(text_chunk="", is_final=True)
        except Exception as e:
            logging.error(f"Error in ChatStream: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error in chat stream: {str(e)}")

    def PlannedAnswer(self, request, context):
        try:
            model_name = "party_planner"
            history = []
            response = self.models_manager.chat(
                model_id=model_name.lower(),
                history=history,
                prompt=request.input,
                context=self.party_planner_actions.get_actions(),
            )
            return PlannedAnswerResponse(plan=response)

        except Exception as e:
            logging.error(f"Error in PlannedAnswer: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error in planned answer stream: {str(e)}")
            return PlannedAnswerResponse()


def serve(config, embeddings_service, models_manager):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting gRPC server on port {config.port}")
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    logger.info("gRPC server created")

    llm_service = LLMService(embeddings_service, models_manager)
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
