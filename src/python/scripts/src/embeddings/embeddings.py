import numpy as np
from enum import Enum
from fastembed import TextEmbedding
from fastembed.common.utils import iter_batch
from concurrent.futures import ThreadPoolExecutor
from fastembed.text.pooled_embedding import PooledEmbedding

_thread_executor = None


def initialize_thread_executor(max_workers=None):
    global _thread_executor
    if _thread_executor is None:
        _thread_executor = ThreadPoolExecutor(max_workers=max_workers)


def set_thread_executor(executor):
    global _thread_executor
    _thread_executor = executor


def shutdown_thread_executor():
    global _thread_executor
    if _thread_executor is not None:
        _thread_executor.shutdown()
        _thread_executor = None


def process_mean_pooling(args):
    model_output, attention_mask = args
    mean_polled = PooledEmbedding.mean_pooling(model_output, attention_mask)
    return mean_polled.astype(np.float32)[0]


def embed_alternative(model, input_strings, batch_size=256):
    if _thread_executor is None:
        raise RuntimeError("Thread executor not initialized. Call `initialize_thread_executor` first.")

    for batch in iter_batch(input_strings, batch_size):
        onnx_output = model.model.onnx_embed(batch)
        model_output = onnx_output.model_output
        attention_mask = onnx_output.attention_mask

        args_list = [(model_output[i], attention_mask[i]) for i in range(len(model_output))]

        for result in _thread_executor.map(process_mean_pooling, args_list):
            yield result


def extend_supported_models():
    from fastembed.text.onnx_embedding import supported_onnx_models

    new_models = [
        {
            "model": "intfloat/multilingual-e5-small",
            "dim": 384,
            "description": "Text embeddings, Multilingual (~100 languages)",
            "license": "mit",
            "size_in_GB": 0.4,
            "sources": {"hf": "intfloat/multilingual-e5-small"},
            "model_file": "onnx/model.onnx",
        },
        {
            "model": "intfloat/multilingual-e5-base",
            "dim": 768,
            "description": "Text embeddings, Multilingual (~100 languages)",
            "license": "mit",
            "size_in_GB": 1.11,
            "sources": {"hf": "intfloat/multilingual-e5-base"},
            "model_file": "onnx/model.onnx",
        },
        {
            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "dim": 768,
            "description": "Text embeddings in many languages",
            "license": "apache-2.0",
            "size_in_GB": 0.4,
            "sources": {"hf": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"},
            "model_file": "onnx/model.onnx",
        },
        {
            "model": "jinaai/jina-embeddings-v2-base-code",
            "dim": 768,
            "description": "Text embeddings for code and english",
            "license": "apache-2.0",
            "size_in_GB": 0.6,
            "sources": {"hf": "jinaai/jina-embeddings-v2-base-code"},
            "model_file": "onnx/model.onnx",
        },
    ]

    for model in new_models:
        if model not in supported_onnx_models:
            supported_onnx_models.append(model)


def get_supported_models_info():
    extend_supported_models()

    supported_models = {
        m["model"]: {"dimensions": m["dim"], "model_name": m["model"]} for m in TextEmbedding.list_supported_models()
    }

    return supported_models


class OramaModelInfo(Enum):
    SUPPORTED_MODELS_INFO = get_supported_models_info()

    # Regular models
    BGESmall = SUPPORTED_MODELS_INFO["BAAI/bge-small-en-v1.5"]
    BGEBase = SUPPORTED_MODELS_INFO["BAAI/bge-base-en-v1.5"]
    BGELarge = SUPPORTED_MODELS_INFO["BAAI/bge-large-en-v1.5"]

    # Multilingual models
    MultilingualE5Small = SUPPORTED_MODELS_INFO["intfloat/multilingual-e5-small"]
    MultilingualE5Base = SUPPORTED_MODELS_INFO["intfloat/multilingual-e5-base"]
    MultilingualE5Large = SUPPORTED_MODELS_INFO["intfloat/multilingual-e5-large"]
    MultilingualMiniLML12V2 = SUPPORTED_MODELS_INFO["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]

    # Code models
    JinaEmbeddingsV2BaseCode = SUPPORTED_MODELS_INFO["jinaai/jina-embeddings-v2-base-code"]


class ModelGroups(Enum):
    en = [OramaModelInfo.BGEBase, OramaModelInfo.BGESmall, OramaModelInfo.BGELarge]
    multilingual = [
        OramaModelInfo.MultilingualE5Large,
        OramaModelInfo.MultilingualE5Small,
        OramaModelInfo.MultilingualE5Base,
        OramaModelInfo.MultilingualMiniLML12V2,
    ]
    small = [
        OramaModelInfo.BGESmall,
        OramaModelInfo.MultilingualE5Small,
    ]
    code = [OramaModelInfo.JinaEmbeddingsV2BaseCode]
    all = [
        OramaModelInfo.BGESmall,
        OramaModelInfo.BGEBase,
        OramaModelInfo.BGELarge,
        OramaModelInfo.MultilingualE5Small,
        OramaModelInfo.MultilingualE5Base,
        OramaModelInfo.MultilingualE5Large,
        OramaModelInfo.MultilingualMiniLML12V2,
        OramaModelInfo.JinaEmbeddingsV2BaseCode,
    ]
