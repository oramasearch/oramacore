use std::ffi::CStr;

pub mod embeddings;

// @todo: we will have to move all the python stuff elsewhere.
// Also, we should ensure that we're rinning in the correct venv and Python version.
pub static VENV_DIR: &str = "src/ai_server/.venv/lib/python3.11/site-packages";

pub static INIT_THREAD_EXECUTOR: &CStr = c"
from src.embeddings.embeddings import initialize_thread_executor
initialize_thread_executor()
";

pub static EMBEDDINGS_CONFIG_CODE: &CStr = c"
class EmbeddingsConfig:
    def __init__(self):
        # Set to True to avoid loading models during test initialization
        self.dynamically_load_models = True
        self.embeddings = type('obj', (object,), {
            'execution_providers': ['CPUExecutionProvider'],
            'dynamically_load_models': True
        })()

config = EmbeddingsConfig()
";

pub static EMBEDDINGS_LOADING_CODE: &CStr = c"
class ModelInfo:
    def __init__(self, name, model_name):
        self.name = name
        self.value = {'model_name': model_name}

models = [
    ModelInfo('BGESmall', 'BAAI/bge-small-en-v1.5'),
    ModelInfo('JinaEmbeddingsV2BaseCode', 'jinaai/jina-embeddings-v2-base-code'),
    ModelInfo('MultilingualE5Small', 'intfloat/multilingual-e5-small'),
]
";
