use anyhow::Result;
use anyhow::{Context, Error};
use deno_core::{JsRuntime, RuntimeOptions};
use strum_macros::Display;
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

use crate::metrics::{Empty, JAVASCRIPT_REQUEST_GAUDGE};

pub struct JavaScript {
    sender: mpsc::Sender<Job>,
}

#[derive(Debug, Display)]
pub enum Operation {
    Anonymous,
    SelectEmbeddingsProperties,
    DynamicDocumentRanking,
}

#[derive(Debug)]
struct Job {
    code: String,
    input: serde_json::Value,
    response: oneshot::Sender<Result<String, Error>>,
    operation: Operation,
}

impl JavaScript {
    pub async fn new(channel_limit: usize) -> Self {
        let (sender, mut receiver) = mpsc::channel::<Job>(channel_limit);

        std::thread::spawn(move || {
            let mut runtime = JsRuntime::new(RuntimeOptions::default());

            let mut buff = Vec::with_capacity(channel_limit);
            
            loop {
                info!("Waiting for jobs...");
                // recv_many returns 0 if the channel is closed
                let count = receiver.blocking_recv_many(&mut buff, channel_limit);
                if count == 0 {
                    break;
                }

                info!("Received job {:?}", count);

                for job in buff.drain(..count) {
                    // @todo: based on the `Operation`, we can perform custom checks and custom script
                    // operations on the incoming data.
                    let full_script = format!(
                        r#"
                            (() => {{
                                const input = {};
                                const func = {};
                                const output = func(input);
                                return JSON.stringify(output);
                            }})()
                        "#,
                        job.input, job.code
                    );

                    let script_name = format!("{}_script.js", job.operation);
                    let b = Box::into_raw(Box::new(script_name));
                    let c: &'static str = unsafe { &*b };

                    info!("Running script in Deno: {}", full_script);
                    let result = runtime
                        .execute_script(c, full_script)
                        .with_context(|| {
                            format!(
                                "Failed to run script in Deno in operation '{}'",
                                job.operation
                            )
                        })
                        .and_then(|value| {
                            let scope = &mut runtime.handle_scope();
                            let local = value.open(scope);
                            if let Some(js_string) = local.to_string(scope) {
                                Ok(js_string.to_rust_string_lossy(scope))
                            } else {
                                Err(Error::msg("Failed to convert JavaScript value to string"))
                            }
                        })
                        .map_err(|err| Error::msg(format!("JavaScript error: {:?}", err)));
                    info!("Deno result: {:?}", result);

                    // let _ = unsafe { Box::from_raw(b) };
                    let _ = job.response.send(result);

                    JAVASCRIPT_REQUEST_GAUDGE.create(Empty {}).decrement_by(1);
                }
            }
            warn!("JS runtime thread finished");
        });

        Self { sender }
    }

    pub async fn eval<T: serde::Serialize, R: serde::de::DeserializeOwned>(
        &self,
        operation: Operation,
        code: String,
        input: T,
    ) -> Result<R> {
        JAVASCRIPT_REQUEST_GAUDGE.create(Empty {}).increment_by(1);

        let input_json = serde_json::to_value(input)?;
        let (response_tx, response_rx) = oneshot::channel();
        let job = Job {
            code,
            operation,
            input: input_json,
            response: response_tx,
        };

        info!("Sending job to JavaScript runtime... {:?}", job);
        self.sender
            .send(job)
            .await
            .map_err(|_| Error::msg("Runtime thread disconnected"))?;
        let res = response_rx.await??;
        info!("Received response from JavaScript runtime {}", res);

        serde_json::from_str(&res).context("Unable to deserialize string into valid data structure")
    }
}
