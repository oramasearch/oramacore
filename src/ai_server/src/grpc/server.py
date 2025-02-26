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
    Role as ProtoRole,
    EmbeddingResponse as EmbeddingResponseProto,
    ChatResponse,
    ChatStreamResponse,
    HealthCheckResponse,
    LLMType,
    PlannedAnswerResponse,
    SegmentResponse as ProtoSegmentResponse,
    TriggerResponse as ProtoTriggerResponse,
    AutoQueryResponse as ProtoAutoQueryResponse,
)
from src.utils import OramaAIConfig
from src.prompts.party_planner import PartyPlannerActions
from src.actions.party_planner import PartyPlanner
from src.prompts.main import PROMPT_TEMPLATES


class LLMService(service_pb2_grpc.LLMServiceServicer):
    def __init__(self, embeddings_service, models_manager, config: OramaAIConfig):
        self.config = config
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
                    {
                        "role": ProtoRole.Name(message.role).lower(),
                        "content": message.content,
                    }
                    for message in request.conversation.messages
                ]
                if request.conversation.messages
                else []
            )

            if request.HasField("segment"):
                history[-1]["content"] += dedent(
                    f"""
                    ### Persona
                    - **Name**: {request.segment.name}
                    - **Description**: {request.segment.description}
                    - **Goal**: {request.segment.goal}                        
                """
                )

            if request.HasField("trigger"):
                history[-1]["content"] += dedent(
                    f"""
                    ### Trigger
                    - **Name**: {request.trigger.name}
                    - **Description**: {request.trigger.description}
                    - **Response**: {request.trigger.response}
                """
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
                    {
                        "role": ProtoRole.Name(message.role).lower(),
                        "content": message.content,
                    }
                    for message in request.conversation.messages
                ]
                if request.conversation.messages
                else []
            )

            segment, trigger = get_segment_and_trigger(request)

            if segment:
                history[-1]["content"] += segment

            if trigger:
                history[-1]["content"] += trigger

            for text_chunk in self.models_manager.chat_stream(
                model_id=model_name.lower(),
                history=history,
                prompt=request.prompt,
                context=request.context,
            ):
                yield ChatStreamResponse(text_chunk=text_chunk, is_final=False)
            yield ChatStreamResponse(text_chunk="", is_final=True)
        except Exception as e:
            logging.error(f"Error in ChatStream: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error in chat stream: {str(e)}")

    def PlannedAnswer(self, request, context):
        metadata = dict(context.invocation_metadata())
        api_key = metadata.get("x-api-key") or ""

        try:
            history = (
                [
                    {
                        "role": ProtoRole.Name(message.role).lower(),
                        "content": message.content,
                    }
                    for message in request.conversation.messages
                ]
                if request.conversation.messages
                else []
            )

            segment, trigger = get_segment_and_trigger(request)

            party_planner = PartyPlanner(self.config, self.models_manager, history)

            for message in party_planner.run(
                collection_id=request.collection_id,
                input=request.input,
                api_key=api_key,
                segment=segment,
                trigger=trigger,
            ):
                yield PlannedAnswerResponse(data=message, finished=False)

            yield PlannedAnswerResponse(data="", finished=True)

        except Exception as e:
            logging.error(f"Error in PlannedAnswer: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error in planned answer stream: {str(e)}")
            return PlannedAnswerResponse()

    def GetSegment(self, request, context):

        if not request.segments:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("At least one segment must be provided")
            return ProtoSegmentResponse()

        try:
            model_name = "answer"  # @todo: make this configurable

            full_conversation = ""
            for message in request.conversation.messages:
                role = ProtoRole.Name(message.role).lower()
                full_conversation += f"Role: {role}\nContent: {message.content}\n"

            segments_data = [
                {
                    "id": segment.id,
                    "name": segment.name,
                    "description": segment.description,
                    "goal": segment.goal if segment.HasField("goal") else None,
                }
                for segment in request.segments
            ]

            history = [
                {
                    "role": "system",
                    "content": PROMPT_TEMPLATES["segmenter:system"],
                },
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATES["segmenter:user"](segments_data, full_conversation),
                },
            ]

            response = self.models_manager.chat(model_id=model_name.lower(), history=history, prompt="")

            repaired_response = repair_json(response)
            repaired_response_json = json.loads(repaired_response)

            selected_segment = (
                repaired_response_json[0] if isinstance(repaired_response_json, list) else repaired_response
            )

            return ProtoSegmentResponse(
                id=selected_segment.get("id", ""),
                name=selected_segment.get("name", ""),
                probability=selected_segment.get("probability", 0.0),
            )

        except Exception as e:
            logging.error(f"Error in GetSegment: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error in segment classification: {str(e)}")
            return ProtoSegmentResponse()

    def GetTrigger(self, request, context):

        if not request.triggers:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("At least one trigger must be provided")
            return ProtoTriggerResponse()

        try:
            model_name = "answer"  # @todo: make this configurable

            full_conversation = ""
            for message in request.conversation.messages:
                role = ProtoRole.Name(message.role).lower()
                full_conversation += f"Role: {role}\nContent: {message.content}\n"

            triggers_data = [
                {
                    "id": trigger.id,
                    "name": trigger.name,
                    "description": trigger.description,
                    "response": trigger.response,
                }
                for trigger in request.triggers
            ]

            history = [
                {
                    "role": "system",
                    "content": PROMPT_TEMPLATES["trigger:system"],
                },
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATES["trigger:user"](triggers_data, full_conversation),
                },
            ]

            response = self.models_manager.chat(model_id=model_name.lower(), history=history, prompt="")

            repaired_response = repair_json(response)
            repaired_response_json = json.loads(repaired_response)

            selected_trigger = (
                repaired_response_json[0] if isinstance(repaired_response_json, list) else repaired_response
            )

            return ProtoTriggerResponse(
                id=selected_trigger.get("id", ""),
                name=selected_trigger.get("name", ""),
                response=selected_trigger.get("response", ""),
                probability=selected_trigger.get("probability", 0.0),
            )

        except Exception as e:
            logging.error(f"Error in GetSegment: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error in segment classification: {str(e)}")
            return ProtoTriggerResponse()

    def AutoQuery(self, request, context):
        try:
            model_name = "answer"  # @todo: make this configurable

            history = [
                {
                    "role": "system",
                    "content": PROMPT_TEMPLATES["autoquery:system"],
                },
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATES["autoquery:user"](query=request.query, _context=""),
                },
            ]

            response = self.models_manager.chat(model_id=model_name.lower(), history=history, prompt="")
            repaired_response = repair_json(response)
            repaired_response_json = json.loads(repaired_response)

            return ProtoAutoQueryResponse(
                mode=repaired_response_json.get("mode", ""),
            )

        except Exception as e:
            logging.error(f"Error in GetSegment: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error in segment classification: {str(e)}")
            return ProtoAutoQueryResponse()


class AuthInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, handler_call_details):

        # Health check and embeddings won't require authentication.
        # This server should never be exposed to the public and it's meant for internal use only.
        allowed_methods = ["CheckHealth", "GetEmbedding", "ServerReflection", "GetSegment", "GetTrigger"]
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


def get_segment_and_trigger(request) -> tuple[str | None, str | None]:
    segment = None
    trigger = None

    if request.HasField("segment"):
        segment = dedent(
            f"""
            ### Persona
            - **Name**: {request.segment.name}
            - **Description**: {request.segment.description}
            - **Goal**: {request.segment.goal}                        
        """
        )

    if request.HasField("trigger"):
        trigger = dedent(
            f"""
            ### Trigger
            - **Name**: {request.trigger.name}
            - **Description**: {request.trigger.description}
            - **Response**: {request.trigger.response}
        """
        )

    return segment, trigger


def serve(config, embeddings_service, models_manager):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting gRPC server on port {config.port}")
    server = grpc.server(ThreadPoolExecutor(max_workers=10), interceptors=[AuthInterceptor()])
    logger.info("gRPC server created")

    llm_service = LLMService(embeddings_service, models_manager, config)
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
