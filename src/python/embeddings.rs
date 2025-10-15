use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

// @todo: we will have to move all the python stuff elsewhere.
// Also, we should ensure that we're rinning in the correct venv and Python version.
static VENV_DIR: &str = "src/ai_server/.venv/lib/python3.11/site-packages";

#[derive(Serialize, Deserialize, Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum Model {
    BGESmall,
    BGEBase,
    BGELarge,
    JinaEmbeddingsV2BaseCode,
    MultilingualE5Small,
    MultilingualE5Base,
    MultilingualE5Large,
    MultilingualMiniLML12V2,
}

impl Display for Model {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::BGESmall => write!(f, "BGESmall"),
            Model::BGEBase => write!(f, "BGEBase"),
            Model::BGELarge => write!(f, "BGELarge"),
            Model::JinaEmbeddingsV2BaseCode => write!(f, "JinaEmbeddingsV2BaseCode"),
            Model::MultilingualE5Small => write!(f, "MultilingualE5Small"),
            Model::MultilingualE5Base => write!(f, "MultilingualE5Base"),
            Model::MultilingualE5Large => write!(f, "MultilingualE5Large"),
            Model::MultilingualMiniLML12V2 => write!(f, "MultilingualMiniLML12V2"),
        }
    }
}

impl Model {
    pub fn from_str(model_name: &str) -> Option<Self> {
        match model_name {
            "BGESmall" => Some(Model::BGESmall),
            "BGEBase" => Some(Model::BGEBase),
            "BGELarge" => Some(Model::BGELarge),
            "JinaEmbeddingsV2BaseCode" => Some(Model::JinaEmbeddingsV2BaseCode),
            "MultilingualE5Small" => Some(Model::MultilingualE5Small),
            "MultilingualE5Base" => Some(Model::MultilingualE5Base),
            "MultilingualE5Large" => Some(Model::MultilingualE5Large),
            "MultilingualMiniLML12V2" => Some(Model::MultilingualMiniLML12V2),
            _ => None,
        }
    }

    pub fn dimensions(&self) -> usize {
        match self {
            Model::BGESmall => 384,
            Model::BGEBase => 768,
            Model::BGELarge => 1024,
            Model::JinaEmbeddingsV2BaseCode => 768,
            Model::MultilingualE5Small => 384,
            Model::MultilingualE5Base => 768,
            Model::MultilingualE5Large => 1024,
            Model::MultilingualMiniLML12V2 => 384,
        }
    }

    pub fn overlap(&self) -> usize {
        self.dimensions() * 2 / 100
    }
}

#[derive(Serialize, Deserialize)]
pub enum Intent {
    Passage,
    Query,
}

impl Display for Intent {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Intent::Passage => write!(f, "passage"),
            Intent::Query => write!(f, "query"),
        }
    }
}

impl Intent {
    pub fn from_str(intent_name: &str) -> Option<Self> {
        match intent_name {
            "passage" => Some(Intent::Passage),
            "query" => Some(Intent::Query),
            _ => None,
        }
    }
}

pub struct EmbeddingsService {
    instance: Py<PyAny>,
}

impl EmbeddingsService {
    pub fn new() -> PyResult<Self> {
        // @todo: move this to lib.rs and call it only once during the application startup.
        Python::initialize();

        Python::attach(|py| {
            Self::initialize_python_env(py)?;

            let utils_module = py.import("src.utils")?;
            let config_class = utils_module.getattr("OramaAIConfig")?;
            let config = config_class.call0()?;
            let embeddings_config = config.getattr("embeddings")?;

            config.setattr("dynamically_load_models", true)?;
            embeddings_config.setattr("dynamically_load_models", true)?;

            // @todo: make this configurable via the config.yaml file. We should support both CPU and CUDA execution providers.
            let execution_providers = py.eval(c"['CPUExecutionProvider']", None, None)?;
            embeddings_config.setattr("execution_providers", execution_providers)?;

            let models_module = py.import("src.embeddings.models")?;
            let embeddings_class = models_module.getattr("EmbeddingsModels")?;
            let instance = embeddings_class.call1((config,))?;

            Ok(EmbeddingsService {
                instance: instance.unbind(),
            })
        })
    }

    pub fn initialize_python_env(py: Python<'_>) -> PyResult<()> {
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;

        if std::path::Path::new(VENV_DIR).exists() {
            path.call_method1("insert", (0, VENV_DIR))?;
        }

        path.call_method1("insert", (0, "src/ai_server"))?;

        let embeddings_module = py.import("src.embeddings.embeddings")?;
        let init_fn = embeddings_module.getattr("initialize_thread_executor")?;
        init_fn.call0()?;

        Ok(())
    }

    pub fn calculate_embeddings(
        &self,
        input: Vec<String>,
        intent: Option<Intent>,
        model: Model,
    ) -> PyResult<Vec<Vec<f32>>> {
        Python::attach(|py| {
            let instance = self.instance.bind(py);
            let intent_str = intent.map(|i| i.to_string());

            let result = instance.call_method1(
                "calculate_embeddings",
                (input, intent_str, model.to_string()),
            )?;

            result.extract()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_embeddings_with_single_string() -> PyResult<()> {
        let embeddings = EmbeddingsService::new()?;
        let result = embeddings.calculate_embeddings(
            vec!["Hello world".to_string()],
            None,
            Model::BGESmall,
        )?;

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 384);

        Ok(())
    }

    #[test]
    fn test_calculate_embeddings_with_multiple_strings() -> PyResult<()> {
        let embeddings = EmbeddingsService::new()?;

        let result = embeddings.calculate_embeddings(
            vec![
                "Hello world".to_string(),
                "The quick brown fox jumps over the lazy dog".to_string(),
            ],
            None,
            Model::BGESmall,
        )?;

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 384);
        assert_eq!(result[1].len(), 384);

        Ok(())
    }

    #[test]
    fn test_calculate_embeddings_with_multiple_models() -> PyResult<()> {
        let embeddings = EmbeddingsService::new()?;

        let result1 = embeddings.calculate_embeddings(
            vec!["Hello world".to_string()],
            None,
            Model::BGESmall,
        )?;

        let result2 = embeddings.calculate_embeddings(
            vec!["Hello world".to_string()],
            None,
            Model::JinaEmbeddingsV2BaseCode,
        )?;

        assert_eq!(result1.len(), 1);
        assert_eq!(result1[0].len(), 384);

        assert_eq!(result2.len(), 1);
        assert_eq!(result2[0].len(), 768);

        Ok(())
    }

    #[test]
    fn test_calculate_embeddings_with_different_intent() -> PyResult<()> {
        let embeddings = EmbeddingsService::new()?;

        let result1 = embeddings.calculate_embeddings(
            vec!["Hello world".to_string()],
            Some(Intent::Passage),
            Model::MultilingualE5Small,
        )?;

        let result2 = embeddings.calculate_embeddings(
            vec!["Hello world".to_string()],
            Some(Intent::Query),
            Model::MultilingualE5Small,
        )?;

        assert_eq!(result1.len(), 1);
        assert_eq!(result2.len(), 1);
        assert_ne!(result1, result2);

        Ok(())
    }
}
