use anyhow::Error;
use deno_core::{JsRuntime, RuntimeOptions};
use std::sync::mpsc;
use std::thread;
use tokio::sync::oneshot;

pub struct JavaScript {
    sender: mpsc::Sender<Job>,
}

struct Job {
    code: String,
    input: serde_json::Value,
    response: oneshot::Sender<Result<String, Error>>,
}

impl JavaScript {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::channel::<Job>();

        thread::spawn(move || {
            let mut runtime = JsRuntime::new(RuntimeOptions::default());

            for job in receiver {
                let full_script = format!(
                    r#"
                        (() => {{
                            const input = {};
                            const func = {};
                            const output = func(input);
                            return JSON.stringify(output);
                        }})()
                    "#,
                    job.input.to_string(),
                    job.code
                );

                let result = runtime.execute_script("user_function.js", full_script)
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


                let _ = job.response.send(result);
            }
        });

        Self { sender }
    }

    pub async fn eval<T: serde::Serialize>(
        &self,
        code: String,
        input: T,
    ) -> Result<String, Error> {
        let input_json = serde_json::to_value(input)?;
        let (response_tx, response_rx) = oneshot::channel();
        let job = Job {
            code,
            input: input_json,
            response: response_tx,
        };

        self.sender.send(job).map_err(|_| Error::msg("Runtime thread disconnected"))?;
        response_rx.await?
    }
}
