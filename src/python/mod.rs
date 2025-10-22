use include_dir::{include_dir, Dir};
use pyo3::prelude::*;
use std::{path::PathBuf, sync::Arc};
use tracing::info;

pub mod embeddings;
pub mod mcp;

// @todo: we should ensure that we're running in the correct venv and Python version.
static VENV_DIR: &str = ".venv/lib/python3.11/site-packages";
static PYTHON_SCRIPTS: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/src/python/scripts");

pub struct PythonService {
    pub embeddings_service: Arc<embeddings::EmbeddingsService>,
}

impl PythonService {
    pub fn new() -> PyResult<Self> {
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

            Ok(PythonService {
                embeddings_service: Arc::new(embeddings::EmbeddingsService::new()?),
            })
        })
    }

    pub fn initialize_python_env(py: Python<'_>, python_scripts_dir: &PathBuf) -> PyResult<()> {
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;

        if std::path::Path::new(VENV_DIR).exists() {
            path.call_method1("insert", (0, VENV_DIR))?;
        }

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

        info!(
            "Attempting to extract embedded Python scripts to: {:?}",
            scripts_dir
        );

        if let Err(e) = std::fs::create_dir_all(&scripts_dir) {
            tracing::warn!("Failed to create directory for Python scripts: {}", e);
        }

        match PYTHON_SCRIPTS.extract(&scripts_dir) {
            Ok(_) => {
                info!("[PYTHON] Python scripts extracted successfully from embedded binary");
                info!("[PYTHON] Location: {:?}", scripts_dir);
                info!("[PYTHON] Source: Binary embedded");
                Ok(scripts_dir)
            }
            Err(e) => {
                // Check if we should allow fallback to local files
                let allow_local_fallback = std::env::var("ORAMACORE_ALLOW_LOCAL_PYTHON_SCRIPTS")
                    .unwrap_or_else(|_| "true".to_string())
                    .to_lowercase()
                    == "true";

                if !allow_local_fallback {
                    anyhow::bail!(
                        "Failed to extract embedded Python scripts and local fallback is disabled. Error: {}",
                        e
                    );
                }

                // Fall back to local development path if extraction fails
                tracing::warn!(
                    "Failed to extract embedded Python scripts: {}. Falling back to local development path",
                    e
                );

                let local_path = PathBuf::from("src/python/scripts");
                if local_path.exists() {
                    tracing::warn!("[PYTHON] Using local Python scripts from: {:?}", local_path);
                    tracing::warn!("[PYTHON] Source: Local filesystem (development mode)");
                    tracing::error!("[PYTHON] This should NOT happen in production!");
                    Ok(local_path)
                } else {
                    anyhow::bail!("Python scripts not found. Extraction failed: {}, and local path does not exist", e);
                }
            }
        }
    }
}
