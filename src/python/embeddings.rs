use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Display, Formatter},
    str::FromStr,
};

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

    // Some models need a rescale because they produce "similar" embeddings.
    #[inline]
    pub fn rescale_score(&self, score: f32) -> f32 {
        let is_e5_model = matches!(
            self,
            Model::MultilingualE5Small | Model::MultilingualE5Base | Model::MultilingualE5Large
        );

        if is_e5_model {
            // For instance, E5 models produce similarity scores that rarely go below
            // 0.7, making the effective range much narrower.
            // So, cosine similarity scores are usually in a narrow range like [0.7, 1.0],
            // instead of the full [0.0, 1.0] range.
            // This rescaling helps normalize the scores to use the full [0.0, 1.0]
            // range for better search ranking.
            const MIN: f32 = 0.7;
            const MAX: f32 = 1.0;
            const DELTA: f32 = MAX - MIN;
            let clamped_score = score.clamp(MIN, MAX);
            (clamped_score - MIN) / DELTA
        } else {
            score
        }
    }
}

impl FromStr for Model {
    type Err = ();

    fn from_str(model_name: &str) -> Result<Self, Self::Err> {
        match model_name {
            "BGESmall" => Ok(Model::BGESmall),
            "BGEBase" => Ok(Model::BGEBase),
            "BGELarge" => Ok(Model::BGELarge),
            "JinaEmbeddingsV2BaseCode" => Ok(Model::JinaEmbeddingsV2BaseCode),
            "MultilingualE5Small" => Ok(Model::MultilingualE5Small),
            "MultilingualE5Base" => Ok(Model::MultilingualE5Base),
            "MultilingualE5Large" => Ok(Model::MultilingualE5Large),
            "MultilingualMiniLML12V2" => Ok(Model::MultilingualMiniLML12V2),
            _ => Err(()),
        }
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

impl FromStr for Intent {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "passage" => Ok(Intent::Passage),
            "query" => Ok(Intent::Query),
            _ => Err(()),
        }
    }
}

pub struct EmbeddingsService {
    instance: Py<PyAny>,
}

impl EmbeddingsService {
    pub fn new() -> PyResult<Self> {
        Python::attach(|py| {
            let utils_module = py.import("src.utils")?;
            let config_class = utils_module.getattr("OramaAIConfig")?;
            let config = config_class.call0()?;
            let embeddings_config = config.getattr("embeddings")?;

            config.setattr("dynamically_load_models", true)?;
            embeddings_config.setattr("dynamically_load_models", true)?;

            let models_module = py.import("src.embeddings.models")?;
            let embeddings_class = models_module.getattr("EmbeddingsModels")?;
            let instance = embeddings_class.call1((config,))?;

            Ok(EmbeddingsService {
                instance: instance.unbind(),
            })
        })
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
    use crate::python::PythonService;
    use std::sync::{Arc, LazyLock};

    use super::*;

    static PYTHON_SERVICE: LazyLock<Arc<PythonService>> = LazyLock::new(|| {
        Arc::new(PythonService::new().expect("Failed to initialize PythonService for tests"))
    });

    fn get_embeddings_service() -> Arc<EmbeddingsService> {
        PYTHON_SERVICE.embeddings_service.clone()
    }

    #[test]
    fn test_calculate_embeddings_with_single_string() -> PyResult<()> {
        let embeddings = get_embeddings_service();

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
        let embeddings = get_embeddings_service();

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
        let embeddings = get_embeddings_service();

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
        let embeddings = get_embeddings_service();

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
