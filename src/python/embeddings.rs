use include_dir::{include_dir, Dir};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::path::PathBuf;
use tracing::info;

// @todo: we will have to move all the python stuff elsewhere.
// Also, we should ensure that we're rinning in the correct venv and Python version.
static VENV_DIR: &str = ".venv/lib/python3.11/site-packages";
static PYTHON_SCRIPTS: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/src/python/scripts");

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

    pub fn sequence_length(&self) -> usize {
        match self {
            Model::BGESmall => 512,
            Model::BGEBase => 512,
            Model::BGELarge => 512,
            Model::JinaEmbeddingsV2BaseCode => 512,
            Model::MultilingualE5Small => 512,
            Model::MultilingualE5Base => 512,
            Model::MultilingualE5Large => 512,
            Model::MultilingualMiniLML12V2 => 128,
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
        self.sequence_length() * 2 / 100
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
    python_scripts_dir: PathBuf,
}

impl EmbeddingsService {
    pub fn new() -> PyResult<Self> {
        // @todo: move this to lib.rs and call it only once during the application startup.
        Python::initialize();

        // Extract Python scripts directory before attaching to Python
        let python_scripts_dir = Self::extract_python_scripts().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to extract Python scripts: {}",
                e
            ))
        })?;

        Python::attach(|py| {
            let sys = py.import("sys")?;
            let version = sys.getattr("version")?.extract::<String>()?;
            let version_number = version.split_whitespace().next().unwrap_or(&version);
            info!("Detected local Python version: v{}", version_number);

            Self::initialize_python_env(py, &python_scripts_dir)?;

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
                python_scripts_dir,
            })
        })
    }

    pub fn initialize_python_env(py: Python<'_>, python_scripts_dir: &PathBuf) -> PyResult<()> {
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;

        if std::path::Path::new(VENV_DIR).exists() {
            path.call_method1("insert", (0, VENV_DIR))?;
        }

        // Use the provided Python scripts directory
        info!("Using Python scripts from: {:?}", python_scripts_dir);
        path.call_method1("insert", (0, python_scripts_dir.to_string_lossy().as_ref()))?;

        let embeddings_module = py.import("src.embeddings.embeddings")?;
        let init_fn = embeddings_module.getattr("initialize_thread_executor")?;
        init_fn.call0()?;

        Ok(())
    }

    fn extract_python_scripts() -> anyhow::Result<PathBuf> {
        let temp_base = std::env::temp_dir();
        let scripts_dir =
            temp_base.join(format!("oramacore_python_scripts_{}", std::process::id()));

        info!("Extracting embedded Python scripts to: {:?}", scripts_dir);

        // Try to extract embedded Python scripts (for distributed binary)
        match PYTHON_SCRIPTS.extract(&scripts_dir) {
            Ok(_) => {
                info!("Python scripts extracted successfully");
                Ok(scripts_dir)
            }
            Err(e) => {
                // Fall back to local development path if extraction fails
                info!(
                    "Failed to extract embedded Python scripts ({}), falling back to local path",
                    e
                );
                if std::path::Path::new("src/python/scripts").exists() {
                    Ok(PathBuf::from("src/python/scripts"))
                } else {
                    anyhow::bail!("Python scripts not found. Extraction failed: {}", e);
                }
            }
        }
    }

    pub fn calculate_embeddings(
        &self,
        input: Vec<String>,
        intent: Intent,
        model: Model,
    ) -> PyResult<Vec<Vec<f32>>> {
        Python::attach(|py| {
            let instance = self.instance.bind(py);

            let result = instance.call_method1(
                "calculate_embeddings",
                (input, intent.to_string(), model.to_string()),
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
            Intent::Passage,
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
            Intent::Passage,
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
            Intent::Passage,
            Model::BGESmall,
        )?;

        let result2 = embeddings.calculate_embeddings(
            vec!["Hello world".to_string()],
            Intent::Passage,
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
            vec!["The process of photosynthesis in plants converts carbon dioxide and water into glucose and oxygen using sunlight energy, primarily occurring in chloroplasts through light-dependent and light-independent reactions.".to_string()],
            Intent::Passage,
            Model::MultilingualE5Small,
        )?;

        let result2 = embeddings.calculate_embeddings(
            vec!["The process of photosynthesis in plants converts carbon dioxide and water into glucose and oxygen using sunlight energy, primarily occurring in chloroplasts through light-dependent and light-independent reactions.".to_string()],
            Intent::Query,
            Model::MultilingualE5Small,
        )?;

        assert_eq!(result1.len(), 1);
        assert_eq!(result2.len(), 1);
        assert_ne!(result1, result2);

        Ok(())
    }
}
