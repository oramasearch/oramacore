from enum import Enum
from pydantic import BaseModel
from typing import List, Union, Optional
from service_pb2 import OramaIntent as ProtoOramaIntent


def create_model_types(embeddings_service):
    FastApiRequestOramaModelType = Enum(
        "FastApiRequestOramaModelType", {name: name for name in embeddings_service.selected_model_names}
    )
    FastApiRequestIntentType = Enum("FastApiRequestIntentType", {name: name for name in ProtoOramaIntent.keys()})

    class FastApiEmbeddingRequest(BaseModel):
        input: Union[List[str], str]
        intent: Optional[FastApiRequestIntentType] = FastApiRequestIntentType["query"]
        model: FastApiRequestOramaModelType

    return FastApiEmbeddingRequest
