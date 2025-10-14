use pyo3::{prelude::*, types::PyDict};
use std::sync::{Arc, Mutex};

use crate::python::{
    EMBEDDINGS_CONFIG_CODE, EMBEDDINGS_LOADING_CODE, INIT_THREAD_EXECUTOR, VENV_DIR,
};

#[derive(Clone)]
pub struct Embeddings {
    instance: Arc<Mutex<Py<PyAny>>>,
}

impl Embeddings {
    pub fn new(py: Python<'_>) -> PyResult<Self> {
        let config = Self::create_config(py)?;
        let models = Self::create_models(py)?;
        let models_module = py.import("src.embeddings.models")?;
        let embeddings_class = models_module.getattr("EmbeddingsModels")?;
        let instance = embeddings_class.call1((config, models))?;

        Ok(Embeddings {
            instance: Arc::new(Mutex::new(instance.unbind())),
        })
    }

    pub fn initialize_python_env(py: Python<'_>) -> PyResult<()> {
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;

        if std::path::Path::new(VENV_DIR).exists() {
            path.call_method1("insert", (0, VENV_DIR))?;
        }

        path.call_method1("insert", (0, "src/ai_server"))?;

        py.run(INIT_THREAD_EXECUTOR, None, None)?;

        Ok(())
    }

    pub fn calculate_embeddings(
        &self,
        py: Python<'_>,
        input: Vec<String>,
        intent: Option<String>,
        model_name: &str,
    ) -> PyResult<Vec<Vec<f32>>> {
        let instance_guard = self.instance.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to acquire lock: {}",
                e
            ))
        })?;

        let instance = instance_guard.bind(py);
        let result = instance.call_method1("calculate_embeddings", (input, intent, model_name))?;

        result.extract()
    }

    fn create_config(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        py.run(EMBEDDINGS_CONFIG_CODE, None, None)?;
        let locals = PyDict::new(py);
        py.run(EMBEDDINGS_CONFIG_CODE, None, Some(&locals))?;
        Ok(locals.get_item("config")?.unwrap())
    }

    fn create_models(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let locals = PyDict::new(py);
        py.run(EMBEDDINGS_LOADING_CODE, None, Some(&locals))?;
        Ok(locals.get_item("models")?.unwrap())
    }
}

#[cfg(test)]
mod tests {
    use crate::python::{EMBEDDINGS_CONFIG_CODE, EMBEDDINGS_LOADING_CODE};

    use super::*;
    use pyo3::types::PyDict;

    fn create_mock_config(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        py.run(EMBEDDINGS_CONFIG_CODE, None, None)?;
        let locals = PyDict::new(py);
        py.run(EMBEDDINGS_CONFIG_CODE, None, Some(&locals))?;
        Ok(locals.get_item("config")?.unwrap())
    }

    fn create_mock_models(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let locals = PyDict::new(py);
        py.run(EMBEDDINGS_LOADING_CODE, None, Some(&locals))?;
        Ok(locals.get_item("models")?.unwrap())
    }

    #[test]
    fn test_embeddings_creation() -> PyResult<()> {
        Python::initialize();

        Python::attach(|py| {
            Embeddings::initialize_python_env(py)?;

            let embeddings = Embeddings::new(py)?;
            let instance_guard = embeddings.instance.lock().unwrap();
            let instance = instance_guard.bind(py);

            assert!(instance.hasattr("calculate_embeddings")?);

            Ok(())
        })
    }

    #[test]
    fn test_calculate_embeddings_with_single_string() -> PyResult<()> {
        Python::initialize();

        Python::attach(|py| {
            Embeddings::initialize_python_env(py)?;

            let embeddings = Embeddings::new(py)?;

            let result = embeddings.calculate_embeddings(
                py,
                vec!["Hello world".to_string()],
                None,
                "BGESmall",
            )?;

            assert_eq!(result.len(), 1);
            assert_eq!(result[0].len(), 384);

            Ok(())
        })
    }

    #[test]
    fn test_calculate_embeddings_with_multiple_strings() -> PyResult<()> {
        Python::initialize();

        Python::attach(|py| {
            Embeddings::initialize_python_env(py)?;

            let embeddings = Embeddings::new(py)?;

            let result = embeddings.calculate_embeddings(
                py,
                vec![
                    "Hello world".to_string(),
                    "The quick brown fox jumps over the lazy dog".to_string(),
                ],
                None,
                "BGESmall",
            )?;

            assert_eq!(result.len(), 2);
            assert_eq!(result[0].len(), 384);
            assert_eq!(result[1].len(), 384);

            Ok(())
        })
    }

    #[test]
    fn test_calculate_embeddings_with_multiple_models() -> PyResult<()> {
        Python::initialize();

        Python::attach(|py| {
            Embeddings::initialize_python_env(py)?;

            let embeddings = Embeddings::new(py)?;

            let result1 = embeddings.calculate_embeddings(
                py,
                vec!["Hello world".to_string()],
                None,
                "BGESmall",
            )?;

            let result2 = embeddings.calculate_embeddings(
                py,
                vec!["Hello world".to_string()],
                None,
                "JinaEmbeddingsV2BaseCode",
            )?;

            assert_eq!(result1.len(), 1);
            assert_eq!(result1[0].len(), 384);

            assert_eq!(result2.len(), 1);
            assert_eq!(result2[0].len(), 768);

            Ok(())
        })
    }

    #[test]
    fn test_calculate_embeddings_with_different_intent() -> PyResult<()> {
        Python::initialize();

        Python::attach(|py| {
            Embeddings::initialize_python_env(py)?;

            let embeddings = Embeddings::new(py)?;

            let result1 = embeddings.calculate_embeddings(
                py,
                vec!["Hello world".to_string()],
                Some("passage".to_string()),
                "MultilingualE5Small",
            )?;

            let result2 = embeddings.calculate_embeddings(
                py,
                vec!["Hello world".to_string()],
                Some("query".to_string()),
                "MultilingualE5Small",
            )?;

            assert_eq!(result1.len(), 1);
            assert_eq!(result2.len(), 1);
            assert_ne!(result1, result2);

            Ok(())
        })
    }
}
