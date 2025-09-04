use std::{sync::Arc, time::Duration};

use oramacore_lib::hook_storage::{HookReader, HookReaderError, HookType};
use orama_js_pool::{ExecOption, JSExecutor, OutputChannel, TryIntoFunctionParameters};
use tracing::info;

use crate::{collection_manager::sides::system_prompts::SystemPrompt, types::SearchParams};
use crate::types::DocumentId;

pub async fn run_before_retrieval(
    hook_reader: &HookReader,
    input: SearchParams,
    log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    exec_option: ExecOption,
) -> Result<SearchParams, HookReaderError> {
    run_hook_with_fallback(
        hook_reader,
        input,
        log_sender,
        exec_option,
        HookType::BeforeRetrieval,
    )
    .await
}

pub async fn run_before_answer(
    hook_reader: &HookReader,
    input: (Vec<(String, String)>, Option<SystemPrompt>),
    log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    exec_option: ExecOption,
) -> Result<(Vec<(String, String)>, Option<SystemPrompt>), HookReaderError> {
    run_hook_with_fallback(
        hook_reader,
        input,
        log_sender,
        exec_option,
        HookType::BeforeAnswer,
    )
    .await
}

async fn run_hook_with_fallback<
    Params: TryIntoFunctionParameters + serde::de::DeserializeOwned + Clone + 'static,
>(
    hook_reader: &HookReader,
    input: Params,
    log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    exec_option: ExecOption,
    hook_type: HookType,
) -> Result<Params, HookReaderError> {
    let content = hook_reader.get_hook_content(hook_type)?;
    if let Some(code) = content {
        info!("Running hook: {:?}", hook_type);
        let mut a: JSExecutor<Params, Option<Params>> = JSExecutor::try_new(
            code,
            Some(vec![]),
            Duration::from_millis(200),
            true,
            hook_type.get_function_name().to_string(),
        )
        .await?;

        let output: Option<Params> = a.exec(input.clone(), log_sender, exec_option).await?;
        info!("Hook change something {:?}", output.is_some());
        Ok(output.unwrap_or(input))
    } else {
        info!("No code for hook {:?}. Skip invocation.", hook_type);
        Ok(input)
    }
}
