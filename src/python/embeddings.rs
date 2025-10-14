use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Embeddings {
    instance: Arc<Mutex<Py<PyAny>>>,
}

impl Embeddings {
    pub fn initialize_python_env(py: Python<'_>) -> PyResult<()> {
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;

        let venv_site_packages = "src/ai_server/.venv/lib/python3.11/site-packages";
        if std::path::Path::new(venv_site_packages).exists() {
            path.call_method1("insert", (0, venv_site_packages))?;
        }

        path.call_method1("insert", (0, "src/ai_server"))?;

        py.run(
            c"
from src.embeddings.embeddings import initialize_thread_executor
initialize_thread_executor()
",
            None,
            None,
        )?;

        Ok(())
    }

    pub fn new(
        py: Python<'_>,
        config: Bound<'_, PyAny>,
        selected_models: Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let models_module = py.import("src.embeddings.models")?;
        let embeddings_class = models_module.getattr("EmbeddingsModels")?;
        let instance = embeddings_class.call1((config, selected_models))?;

        Ok(Embeddings {
            instance: Arc::new(Mutex::new(instance.unbind())),
        })
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;

    fn create_mock_config(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let config_code = c"
class MockConfig:
    def __init__(self):
        # Set to True to avoid loading models during test initialization
        self.dynamically_load_models = True
        self.embeddings = type('obj', (object,), {
            'execution_providers': ['CPUExecutionProvider'],
            'dynamically_load_models': True
        })()

config = MockConfig()
";
        py.run(config_code, None, None)?;
        let locals = PyDict::new(py);
        py.run(config_code, None, Some(&locals))?;
        Ok(locals.get_item("config")?.unwrap())
    }

    fn create_mock_models(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let models_code = c"
class MockModelInfo:
    def __init__(self, name, model_name):
        self.name = name
        self.value = {'model_name': model_name}

models = [
    MockModelInfo('BGESmall', 'BAAI/bge-small-en-v1.5')
]
";
        let locals = PyDict::new(py);
        py.run(models_code, None, Some(&locals))?;
        Ok(locals.get_item("models")?.unwrap())
    }

    #[test]
    fn test_embeddings_creation() -> PyResult<()> {
        Python::initialize();

        Python::attach(|py| {
            Embeddings::initialize_python_env(py)?;

            let config = create_mock_config(py)?;
            let models = create_mock_models(py)?;

            let embeddings = Embeddings::new(py, config, models)?;

            // Test that we can access the instance
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

            let config = create_mock_config(py)?;
            let models = create_mock_models(py)?;
            let embeddings = Embeddings::new(py, config, models)?;

            let result = embeddings.calculate_embeddings(
                py,
                vec!["Hello world".to_string()],
                None,
                "BGESmall",
            )?;

            assert_eq!(result.len(), 1);
            assert!(result[0].len() > 0);

            Ok(())
        })
    }
}
