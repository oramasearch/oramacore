use std::{sync::Arc, time::Duration};

use hook_storage::{HookReader, HookReaderError, HookType};
use orama_js_pool::{ExecOption, JSExecutor, OutputChannel, TryIntoFunctionParameters};

pub async fn run_before_retrieval<
    Params: TryIntoFunctionParameters + serde::de::DeserializeOwned + Clone + 'static,
>(
    hook_reader: &HookReader,
    input: Params,
    log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    exec_option: ExecOption,
) -> Result<Params, HookReaderError> {
    let content = hook_reader.get_hook_content(HookType::BeforeRetrieval)?;

    if let Some(code) = content {
        let mut a: JSExecutor<Params, Option<Params>> = JSExecutor::try_new(
            code,
            Some(vec![]),
            Duration::from_millis(200),
            true,
            HookType::BeforeRetrieval.get_function_name().to_string(),
        )
        .await?;

        let output: Option<Params> = a.exec(input.clone(), log_sender, exec_option).await?;

        Ok(output.unwrap_or(input))
    } else {
        Ok(input)
    }
}
