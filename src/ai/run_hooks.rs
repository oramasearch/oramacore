use std::sync::Arc;

use orama_js_pool::{ExecOptions, OutputChannel, Pool, RuntimeError, TryIntoFunctionParameters};
use oramacore_lib::hook_storage::HookType;
use tracing::info;

use crate::{collection_manager::sides::system_prompts::SystemPrompt, types::SearchParams};

pub async fn run_before_retrieval(
    js_pool: &Pool,
    input: SearchParams,
    log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    exec_options: ExecOptions,
) -> Result<SearchParams, RuntimeError> {
    run_hook_with_fallback(
        js_pool,
        input,
        log_sender,
        exec_options,
        HookType::BeforeRetrieval,
    )
    .await
}

pub async fn run_before_answer(
    js_pool: &Pool,
    input: (Vec<(String, String)>, Option<SystemPrompt>),
    log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    exec_options: ExecOptions,
) -> Result<(Vec<(String, String)>, Option<SystemPrompt>), RuntimeError> {
    run_hook_with_fallback(
        js_pool,
        input,
        log_sender,
        exec_options,
        HookType::BeforeAnswer,
    )
    .await
}

async fn run_hook_with_fallback<
    Params: TryIntoFunctionParameters + serde::de::DeserializeOwned + Clone + Send + 'static,
>(
    js_pool: &Pool,
    input: Params,
    log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    exec_options: ExecOptions,
    hook_type: HookType,
) -> Result<Params, RuntimeError> {
    info!("Attempting to run hook: {:?}", hook_type);

    let exec_opts = if let Some(sender) = log_sender {
        exec_options.with_stdout_sender(sender)
    } else {
        exec_options
    };

    let result: Result<Option<Params>, RuntimeError> = js_pool
        .exec(
            hook_type.get_function_name(),
            hook_type.get_function_name(),
            input.clone(),
            exec_opts,
        )
        .await;

    match result {
        Ok(output) => {
            if output.is_some() {
                info!("Hook {:?} transformed the input", hook_type);
            } else {
                info!("Hook {:?} returned no transformation", hook_type);
            }
            Ok(output.unwrap_or(input))
        }
        Err(RuntimeError::MissingModule(_)) => {
            info!("No hook {:?} found in pool. Skip invocation.", hook_type);
            Ok(input)
        }
        Err(e) => Err(e),
    }
}
