use crate::{HookOperation, HookType};
use fs::*;
use orama_js_pool::{JSExecutor, JSRunnerError};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HookWriterError {
    #[error("Cannot perform operation on FS: {0:?}")]
    FSError(#[from] std::io::Error),
    #[error("Unknown error: {0:?}")]
    Generic(#[from] anyhow::Error),

    #[error("Export error: {0}")]
    ExportError(String),
    #[error("Compilation error: {0}")]
    CompilationError(String),
}

// Type alias for the hook operation callback
pub type HookOperationCallback =
    Box<dyn Fn(HookOperation) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

pub struct HookWriter {
    base_dir: PathBuf,
    before_retrieval_presence: AtomicBool,
    f: HookOperationCallback,
}

impl HookWriter {
    pub fn try_new(base_dir: PathBuf, f: HookOperationCallback) -> Result<Self, HookWriterError> {
        create_if_not_exists(&base_dir)?;

        let before_retrieval_file = base_dir.join(HookType::BeforeRetrieval.get_file_name());
        let before_retrieval_presence = BufferedFile::exists_as_file(&before_retrieval_file);

        Ok(Self {
            base_dir,
            before_retrieval_presence: AtomicBool::new(before_retrieval_presence),
            f,
        })
    }

    pub async fn insert_hook(
        &self,
        hook_type: HookType,
        code: String,
    ) -> Result<(), HookWriterError> {
        // We don't care about the input and output. We just want to check is the hook function is there
        let executor = JSExecutor::<(), ()>::try_new(
            code.to_string(),
            Some(vec![]),
            Duration::from_millis(100),
            true,
            hook_type.get_function_name().to_string(),
        )
        .await;
        match executor {
            Err(JSRunnerError::DefaultExportIsNotAnObject) => {
                return Err(HookWriterError::ExportError(
                    "Default export is not an object".to_string(),
                ))
            }
            Err(JSRunnerError::NoExportedFunction(fn_name)) => {
                return Err(HookWriterError::ExportError(format!(
                    "Default export doesn't contain `{fn_name}` property"
                )))
            }
            Err(JSRunnerError::ExportedElementNotAFunction) => {
                return Err(HookWriterError::ExportError(format!(
                    "Default exported `{}` should be a function",
                    hook_type.get_function_name()
                )))
            }
            Err(JSRunnerError::CompilationError(e)) => {
                return Err(HookWriterError::CompilationError(e.exception_message))
            }
            Err(e) => {
                return Err(HookWriterError::Generic(anyhow::anyhow!(
                    "Unknown JS error {e:?}"
                )))
            }
            Ok(_) => {}
        };

        match hook_type {
            HookType::BeforeRetrieval => {
                self.before_retrieval_presence
                    .store(true, Ordering::Relaxed);
            }
        };
        let path = self.base_dir.join(hook_type.get_file_name());
        BufferedFile::create_or_overwrite(path)?.write_text_data(&code)?;

        (self.f)(HookOperation::Insert(hook_type, code)).await;

        Ok(())
    }

    pub async fn delete_hook(&self, hook_type: HookType) -> Result<(), HookWriterError> {
        let path = self.base_dir.join(hook_type.get_file_name());
        match std::fs::remove_file(&path) {
            Ok(_) => match hook_type {
                HookType::BeforeRetrieval => {
                    self.before_retrieval_presence
                        .store(false, Ordering::Relaxed);
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // File does not exist, treat as success
            }
            Err(e) => return Err(HookWriterError::FSError(e)),
        };

        (self.f)(HookOperation::Delete(hook_type)).await;

        Ok(())
    }

    pub fn list_hooks(&self) -> Result<Vec<(HookType, Option<String>)>, HookWriterError> {
        // For each HookType variant, check if the corresponding file exists
        // Currently only BeforeRetrieval exists, but this is future-proofed
        let hook_type = HookType::BeforeRetrieval;
        let path = self.base_dir.join(hook_type.get_file_name());

        let before_retrieval_content = BufferedFile::open(path)
            .and_then(|f| f.read_text_data())
            .ok();

        Ok(vec![(HookType::BeforeRetrieval, before_retrieval_content)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt;
    use std::sync::{Arc, RwLock};

    #[tokio::test]
    async fn test_hook_writer_lifecycle() {
        let base_dir = generate_new_path();

        // Shared vector to record HookOperation invocations
        let ops: Arc<RwLock<Vec<HookOperation>>> = Arc::new(RwLock::new(Vec::new()));

        let ops_clone = ops.clone();
        let dummy_f = Box::new(move |op: HookOperation| {
            let ops_inner = ops_clone.clone();
            async move {
                ops_inner.write().unwrap().push(op);
            }
            .boxed()
        });

        let writer =
            HookWriter::try_new(base_dir.clone(), dummy_f).expect("Failed to create HookWriter");

        // Initially, no hook file should exist
        let hooks = writer.list_hooks().expect("list_hooks failed");
        assert_eq!(hooks.len(), 1);
        assert!(hooks[0].1.is_none());

        // Insert a hook
        let code = r#"
const beforeRetrieval = function () { }
export default { beforeRetrieval }
"#
        .to_string();
        writer
            .insert_hook(HookType::BeforeRetrieval, code.clone())
            .await
            .expect("insert_hook failed");

        // list_hooks should return the code
        let hooks = writer.list_hooks().expect("list_hooks failed");
        assert_eq!(hooks.len(), 1);
        assert_eq!(hooks[0].1.as_deref(), Some(code.as_str()));

        // Delete the hook
        writer
            .delete_hook(HookType::BeforeRetrieval)
            .await
            .expect("delete_hook failed");

        // list_hooks should return None for content
        let hooks = writer.list_hooks().expect("list_hooks failed");
        assert_eq!(hooks.len(), 1);
        assert!(hooks[0].1.is_none());

        // Assert closure invocations at the end
        let ops = ops.read().unwrap();
        assert_eq!(ops.len(), 2);
        assert!(
            matches!(ops[0], HookOperation::Insert(HookType::BeforeRetrieval, ref c) if c == &code)
        );
        assert!(matches!(
            ops[1],
            HookOperation::Delete(HookType::BeforeRetrieval)
        ));
    }

    #[tokio::test]
    async fn test_hook_writer_error() {
        let base_dir = generate_new_path();

        let dummy_f = Box::new(move |_: HookOperation| async move {}.boxed());

        let writer =
            HookWriter::try_new(base_dir.clone(), dummy_f).expect("Failed to create HookWriter");

        let code = r#"console.log('hello');"#.to_string();
        let err: HookWriterError = writer
            .insert_hook(HookType::BeforeRetrieval, code.clone())
            .await
            .unwrap_err();
        let HookWriterError::ExportError(s) = err else {
            panic!("No HookWriterError::ExportError");
        };
        assert_eq!(s, "Default export is not an object".to_string());

        let code = r#"
export default function () {}
        "#
        .to_string();
        let err: HookWriterError = writer
            .insert_hook(HookType::BeforeRetrieval, code.clone())
            .await
            .unwrap_err();
        let HookWriterError::ExportError(s) = err else {
            panic!("No HookWriterError::ExportError");
        };
        assert_eq!(s, "Default export is not an object".to_string());

        let code = r#"
export default { }
        "#
        .to_string();
        let err: HookWriterError = writer
            .insert_hook(HookType::BeforeRetrieval, code.clone())
            .await
            .unwrap_err();
        let HookWriterError::ExportError(s) = err else {
            panic!("No HookWriterError::ExportError");
        };
        assert_eq!(
            s,
            "Default export doesn't contain `beforeRetrieval` property".to_string()
        );

        let code = r#"
const beforeRetrieval = 42
export default { beforeRetrieval }
        "#
        .to_string();
        let err: HookWriterError = writer
            .insert_hook(HookType::BeforeRetrieval, code.clone())
            .await
            .unwrap_err();
        let HookWriterError::ExportError(s) = err else {
            panic!("No HookWriterError::ExportError");
        };
        assert_eq!(
            s,
            "Default exported `beforeRetrieval` should be a function".to_string()
        );
    }
}
