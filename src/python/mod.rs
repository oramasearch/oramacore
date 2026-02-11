use include_dir::{include_dir, Dir};
use pyo3::{prelude::*, types::PyDict};
use std::{path::PathBuf, sync::Arc};
use tracing::{info, warn};

use crate::ai::AIServiceConfig;

pub mod embeddings;
pub mod mcp;

// @todo: we should ensure that we're running in the correct venv and Python version.
static DEFAULT_VENV_DIR: &str = ".venv/lib/python3.11/site-packages";
static PYTHON_SCRIPTS: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/src/python/scripts");

pub struct PythonService {
    pub embeddings_service: Arc<embeddings::EmbeddingsService>,
}

impl PythonService {
    pub fn new(orama_config: AIServiceConfig) -> PyResult<Self> {
        // This initialization could not be required because it is already called in main function
        // `initialize` has to be called in the main thread.
        // Don't call it here to avoid issues.
        // Don't uncomment the next line even if you know what you're doing.
        // Python::initialize();

        let python_scripts_dir = Self::extract_python_scripts().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to extract Python scripts: {e}"
            ))
        })?;

        Python::attach(|py| {
            let sys = py.import("sys")?;
            let version = sys.getattr("version")?.extract::<String>()?;
            let version_number = version.split_whitespace().next().unwrap_or(&version);
            info!("Detected local Python version: v{}", version_number);

            Self::initialize_python_env(py, &python_scripts_dir)?;

            Self::set_global_logging_level(py)?;

            let embeddings_serivce = embeddings::EmbeddingsService::new(orama_config)?;

            Ok(PythonService {
                embeddings_service: Arc::new(embeddings_serivce),
            })
        })
    }

    fn initialize_python_env(py: Python<'_>, python_scripts_dir: &PathBuf) -> PyResult<()> {
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;

        let venv_dir = std::env::var("ORAMACORE_PYTHON_VENV_DIR")
            .unwrap_or_else(|_| DEFAULT_VENV_DIR.to_string());

        if std::path::Path::new(&venv_dir).exists() {
            path.call_method1("insert", (0, &venv_dir))?;

            // Fix sys.executable: PyO3 sets it to the host Rust binary, but
            // Python's multiprocessing "spawn" method (default on macOS) uses
            // sys.executable to create worker processes. Without this fix, it
            // re-executes the Rust binary instead of Python, causing a process
            // cascade. See: https://github.com/PyO3/pyo3/issues/4215
            //
            // Note: on first model download, a harmless "leaked semaphore"
            // warning may appear at shutdown from multiprocessing's
            // resource_tracker. This is a one-time cosmetic issue — the OS
            // cleans up the semaphore automatically.
            if let Some(venv_base) = std::path::Path::new(&venv_dir).ancestors().nth(3) {
                let python_exe = venv_base.join("bin").join("python3");
                if python_exe.exists() {
                    sys.setattr("executable", python_exe.to_string_lossy().as_ref())?;
                    info!("Set sys.executable to: {:?}", python_exe);
                }
            }
        } else {
            warn!(
                "Specified Python venv directory does not exist: {}. Continuing without it.",
                venv_dir
            );
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
                        "Failed to extract embedded Python scripts and local fallback is disabled. Error: {e}"
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
                    anyhow::bail!("Python scripts not found. Extraction failed: {e}, and local path does not exist");
                }
            }
        }
    }

    /// Set the global logging level for Python modules to WARN
    /// Which mean we will not see dependency logs unless they are errors or warnings
    fn set_global_logging_level(py: Python<'_>) -> PyResult<()> {
        let logging_module = py.import("logging")?;
        let basic_config = logging_module.getattr("basicConfig")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("level", logging_module.getattr("WARN")?)?;
        basic_config.call((), Some(&kwargs))?;

        Ok(())
    }
}
