use anyhow::{Context, Error};
use deno_core::{JsRuntime, RuntimeOptions};
use std::sync::mpsc;
use std::thread;
use strum_macros::Display;
use tokio::sync::oneshot;

pub struct JavaScript {
    sender: mpsc::Sender<Job>,
}

#[derive(Display)]
pub enum Operation {
    Anonymous,
    SelectEmbeddingsProperties,
    DynamicDocumentRanking,
}

struct Job {
    code: String,
    input: serde_json::Value,
    response: oneshot::Sender<Result<String, Error>>,
    operation: Operation,
}

impl Default for JavaScript {
    fn default() -> Self {
        Self::new()
    }
}

impl JavaScript {
    pub fn new() -> Self {
        // @todo: use crossbeam to create a thread pool (multi-consumer, multi-producer)
        let (sender, receiver) = mpsc::channel::<Job>();

        thread::spawn(move || {
            let mut runtime = JsRuntime::new(RuntimeOptions::default());

            for job in receiver {
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

                let _ = unsafe { Box::from_raw(b) };
                let _ = job.response.send(result);
            }
        });

        Self { sender }
    }

    pub async fn eval<T: serde::Serialize>(
        &self,
        operation: Operation,
        code: String,
        input: T,
    ) -> Result<String, Error> {
        let input_json = serde_json::to_value(input)?;
        let (response_tx, response_rx) = oneshot::channel();
        let job = Job {
            code,
            operation,
            input: input_json,
            response: response_tx,
        };

        self.sender
            .send(job)
            .map_err(|_| Error::msg("Runtime thread disconnected"))?;
        response_rx.await?
    }
}
