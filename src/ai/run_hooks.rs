use std::{sync::Arc, time::Duration};

use orama_js_pool::{
    DomainPermission, ExecOptions, OutputChannel, RuntimeError, TryIntoFunctionParameters, Worker,
};
use oramacore_lib::hook_storage::{HookReader, HookReaderError, HookType};
use tracing::info;

use crate::{
    collection_manager::sides::system_prompts::SystemPrompt, types::SearchParams, HooksConfig,
};

pub async fn run_before_retrieval(
    hook_reader: &HookReader,
    input: SearchParams,
    log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    exec_options: ExecOptions,
    hooks_config: &HooksConfig,
) -> Result<SearchParams, HookReaderError> {
    run_hook_with_fallback(
        hook_reader,
        input,
        log_sender,
        exec_options,
        HookType::BeforeRetrieval,
        hooks_config,
    )
    .await
}

pub async fn run_before_answer(
    hook_reader: &HookReader,
    input: (Vec<(String, String)>, Option<SystemPrompt>),
    log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    exec_options: ExecOptions,
    hooks_config: &HooksConfig,
) -> Result<(Vec<(String, String)>, Option<SystemPrompt>), HookReaderError> {
    run_hook_with_fallback(
        hook_reader,
        input,
        log_sender,
        exec_options,
        HookType::BeforeAnswer,
        hooks_config,
    )
    .await
}

async fn run_hook_with_fallback<
    Params: TryIntoFunctionParameters + serde::de::DeserializeOwned + Clone + Send + 'static,
>(
    hook_reader: &HookReader,
    input: Params,
    log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    exec_options: ExecOptions,
    hook_type: HookType,
    hooks_config: &HooksConfig,
) -> Result<Params, HookReaderError> {
    let content = hook_reader.get_hook_content(hook_type)?;
    if let Some(code) = content {
        info!("Running hook: {:?}", hook_type);

        let mut worker = Worker::builder()
            .with_domain_permission(DomainPermission::Allow(hooks_config.allowed_hosts.clone()))
            .with_evaluation_timeout(Duration::from_millis(hooks_config.builder_timeout_ms))
            .add_module(hook_type.get_function_name(), code)
            .build()
            .await
            .map_err(|e: RuntimeError| HookReaderError::Generic(e.into()))?;

        let exec_opts = if let Some(sender) = log_sender {
            exec_options.with_stdout_sender(sender)
        } else {
            exec_options
        };

        let output: Option<Params> = worker
            .exec(
                hook_type.get_function_name(),
                hook_type.get_function_name(),
                input.clone(),
                exec_opts,
            )
            .await
            .map_err(|e: RuntimeError| HookReaderError::Generic(e.into()))?;

        info!("Hook change something {:?}", output.is_some());
        Ok(output.unwrap_or(input))
    } else {
        info!("No code for hook {:?}. Skip invocation.", hook_type);
        Ok(input)
    }
}
