import os
import sys
import grpc
import pytest
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import service_pb2
import service_pb2_grpc
from src.grpc.server import CalculateEmbeddingService, LLMService, VisionService


# Mock context for gRPC calls
@pytest.fixture
def mock_context():
    context = Mock()
    context.set_code = Mock()
    context.set_details = Mock()
    return context


# Mock embeddings service
@pytest.fixture
def mock_embeddings_service():
    service = Mock()
    service.calculate_embeddings = Mock(return_value=[np.array([0.1, 0.2, 0.3])])
    return service


# Mock models manager
@pytest.fixture
def mock_models_manager():
    manager = Mock()
    manager.generate_text = Mock(return_value="Generated text response")
    manager.generate_text_stream = Mock(return_value=iter(["Chunk 1", "Chunk 2"]))
    return manager


class TestCalculateEmbeddingService:
    def test_get_embedding_success(self, mock_context, mock_embeddings_service):
        service = CalculateEmbeddingService(mock_embeddings_service)
        request = service_pb2.EmbeddingRequest(
            model=service_pb2.OramaModel.BGESmall,
            input=["test input"],
            intent=service_pb2.OramaIntent.query,
        )

        response = service.GetEmbedding(request, mock_context)

        assert len(response.embeddings_result) == 1
        assert len(response.embeddings_result[0].embeddings) == 3
        mock_embeddings_service.calculate_embeddings.assert_called_once()

    def test_get_embedding_with_multiple_inputs(self, mock_context, mock_embeddings_service):

        service = CalculateEmbeddingService(mock_embeddings_service)
        request = service_pb2.EmbeddingRequest(
            model=service_pb2.OramaModel.BGESmall,
            input=["input1", "input2", "input3"],
            intent=service_pb2.OramaIntent.passage,
        )

        response = service.GetEmbedding(request, mock_context)

        assert len(response.embeddings_result) == 1
        mock_embeddings_service.calculate_embeddings.assert_called_once_with(
            ["input1", "input2", "input3"], "passage", "BGESmall"
        )


class TestLLMService:
    def test_call_llm_success(self, mock_context, mock_models_manager):

        service = LLMService(mock_models_manager)
        request = service_pb2.LLMRequest(model=service_pb2.LLMType.content_expansion, prompt="Test prompt")

        response = service.CallLLM(request, mock_context)

        assert response.text == "Generated text response"
        mock_models_manager.generate_text.assert_called_once_with(model_key="content_expansion", prompt="Test prompt")

    def test_call_llm_stream_success(self, mock_context, mock_models_manager):

        service = LLMService(mock_models_manager)
        request = service_pb2.LLMRequest(model=service_pb2.LLMType.google_query_translator, prompt="Test prompt")

        responses = list(service.CallLLMStream(request, mock_context))

        assert len(responses) == 3  # Two content chunks plus final empty chunk
        assert all(isinstance(r, service_pb2.LLMStreamResponse) for r in responses)

        # Check content chunks
        content_chunks = [r.text_chunk for r in responses[:-1]]
        assert content_chunks == ["Chunk 1", "Chunk 2"]

        # Check final chunk
        assert responses[-1].text_chunk == ""
        assert responses[-1].is_final == True

        mock_models_manager.generate_text_stream.assert_called_once()

    def test_call_llm_handles_error(self, mock_context, mock_models_manager):

        mock_models_manager.generate_text.side_effect = Exception("Test error")
        service = LLMService(mock_models_manager)
        request = service_pb2.LLMRequest(model=service_pb2.LLMType.content_expansion, prompt="Test prompt")

        response = service.CallLLM(request, mock_context)

        mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
        mock_context.set_details.assert_called_once()
        assert response == service_pb2.LLMResponse()


class TestVisionService:
    def test_call_vision_success(self, mock_context, mock_models_manager):

        service = VisionService(mock_models_manager)

        # Create a small test image
        img = Image.new("RGB", (100, 100), color="red")
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        request = service_pb2.VisionRequest(image=img_byte_arr, text="What's in this image?")

        response = service.CallVision(request, mock_context)

        assert response.text == "Generated text response"
        mock_models_manager.generate_text.assert_called_once()

    def test_call_vision_no_image(self, mock_context, mock_models_manager):

        service = VisionService(mock_models_manager)
        request = service_pb2.VisionRequest(image=b"", text="What's in this image?")

        response = service.CallVision(request, mock_context)

        mock_context.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
        mock_context.set_details.assert_called_once_with("Image data is required")
        assert response == service_pb2.VisionResponse()

    def test_call_vision_invalid_image(self, mock_context, mock_models_manager):

        service = VisionService(mock_models_manager)
        request = service_pb2.VisionRequest(image=b"invalid image data", text="What's in this image?")

        response = service.CallVision(request, mock_context)

        mock_context.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)
        assert "Image processing failed" in mock_context.set_details.call_args[0][0]
        assert response == service_pb2.VisionResponse()

    def test_call_vision_handles_error(self, mock_context, mock_models_manager):

        mock_models_manager.generate_text.side_effect = Exception("Test error")
        service = VisionService(mock_models_manager)

        # Create a small test image
        img = Image.new("RGB", (100, 100), color="red")
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        request = service_pb2.VisionRequest(image=img_byte_arr, text="What's in this image?")

        response = service.CallVision(request, mock_context)

        mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
        assert "Error processing Vision request" in mock_context.set_details.call_args[0][0]
        assert response == service_pb2.VisionResponse()
